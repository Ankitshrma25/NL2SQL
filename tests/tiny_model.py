"""Builds a tiny, randomly initialised causal LM + tokenizer on disk (no download).

Used to exercise the real Transformers/PyTorch backend code path offline in tests.
Its output is meaningless text, which is exactly what we want for testing that the
pipeline never trusts model output.
"""
from __future__ import annotations

from pathlib import Path


def build_tiny_causal_model(out_dir: str) -> str:
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    path = Path(out_dir) / "tiny-sql-lm"
    path.mkdir(parents=True, exist_ok=True)
    corpus = ["SELECT city, SUM(amount) FROM transactions GROUP BY city ORDER BY 2 DESC LIMIT 5",
              "CREATE TABLE t (a INTEGER, b TEXT); question: how many rows", "WHERE AND OR COUNT(*) AVG MIN MAX"] * 20
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=400, min_frequency=1, special_tokens=["<pad>", "<eos>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<eos>", pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant: {% endif %}")
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    return str(path)
