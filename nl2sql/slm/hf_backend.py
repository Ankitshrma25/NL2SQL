"""Offline Hugging Face Transformers / PyTorch backend.

* Loads weights with ``local_files_only=True`` from a local cache
  directory (populate it once with ``scripts/download_model.py``) and sets
  ``HF_HUB_OFFLINE=1`` so no network call is ever made at inference.
* Supports both **causal chat models** (e.g. ``Qwen/Qwen2.5-Coder-1.5B-Instruct``,
  uses the tokenizer's chat template) and **seq2seq text-to-SQL models**
  (T5-style), detected from the model config.
* Greedy decoding by default -> deterministic output.
* ``torch`` / ``transformers`` are imported lazily so the rest of the
  system (validator, fallback, tests, UI) works without them.
"""
from __future__ import annotations

import logging
import os
import threading

from ..config import ModelConfig
from ..prompts import Prompt
from .base import SLMError, SQLGenerator

log = logging.getLogger(__name__)


class HuggingFaceSQLGenerator(SQLGenerator):
    def __init__(self, cfg: ModelConfig | None = None, eager: bool = False):
        self.cfg = cfg or ModelConfig()
        self.name = f"hf:{self.cfg.model_name_or_path}"
        self._model = None
        self._tok = None
        self._seq2seq = False
        self._device = "cpu"
        self._lock = threading.Lock()
        self._load_error: str | None = None
        if eager:
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self.cfg.local_files_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                import torch
                from transformers import (AutoConfig, AutoModelForCausalLM,
                                          AutoModelForSeq2SeqLM, AutoTokenizer)
            except ImportError as e:  # pragma: no cover - env dependent
                self._load_error = f"transformers/torch not installed: {e}"
                raise SLMError(self._load_error) from e

            src = self.cfg.model_name_or_path
            kw = dict(cache_dir=self.cfg.cache_dir, local_files_only=self.cfg.local_files_only)
            try:
                conf = AutoConfig.from_pretrained(src, **kw)
                self._seq2seq = bool(getattr(conf, "is_encoder_decoder", False))
                self._tok = AutoTokenizer.from_pretrained(src, **kw)
                self._tok.truncation_side = "left"  # never cut off the question at the end of the prompt
                dev = self.cfg.device
                if dev == "auto":
                    dev = "cuda" if torch.cuda.is_available() else (
                        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
                self._device = dev
                # dtype = torch.float16 if dev == "cuda" else torch.float32
                if dev == "cuda":
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                else:
                    dtype = torch.float32
                cls = AutoModelForSeq2SeqLM if self._seq2seq else AutoModelForCausalLM
                import transformers
                major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
                dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
                self._model = cls.from_pretrained(src, **{dtype_kw: dtype}, **kw).to(dev).eval()
                log.info("Loaded %s (%s) on %s", src, "seq2seq" if self._seq2seq else "causal", dev)
            except Exception as e:  # missing weights, corrupted cache, ...
                self._load_error = (
                    f"Could not load model '{src}' from '{self.cfg.cache_dir}' offline: {e}. "
                    "Run `python scripts/download_model.py` once while online."
                )
                raise SLMError(self._load_error) from e

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except SLMError:
            return False

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------------
    def _render(self, prompt: Prompt) -> str:
        if self._seq2seq:
            # T5 text-to-SQL checkpoints expect a single flat string
            return prompt.user.replace("\n", " ")
        if getattr(self._tok, "chat_template", None):
            return self._tok.apply_chat_template(prompt.as_messages(), tokenize=False, add_generation_prompt=True)
        return prompt.as_text()

    def generate(self, prompt: Prompt) -> str:
        self._load()
        import torch

        text_in = self._render(prompt)
        inputs = self._tok(text_in, return_tensors="pt", truncation=True, max_length=4096).to(self._device)
        gen_kw = dict(max_new_tokens=self.cfg.max_new_tokens, pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        if self.cfg.temperature and self.cfg.temperature > 0:
            gen_kw.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kw.update(do_sample=False)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kw)
        if self._seq2seq:
            return self._tok.decode(out[0], skip_special_tokens=True)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True)
