"""Real offline SLM backend.

Weight-dependent tests are skipped automatically when torch/transformers
or the cached weights are not present (run scripts/download_model.py once).
"""
import importlib.util

import pytest

from nl2sql.config import ModelConfig
from nl2sql.slm.hf_backend import HuggingFaceSQLGenerator

HAS_STACK = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def test_backend_reports_missing_weights_without_network(tmp_path):
    gen = HuggingFaceSQLGenerator(ModelConfig(model_name_or_path="nonexistent/model", cache_dir=str(tmp_path),
                                              local_files_only=True))
    assert gen.available is False
    assert gen.load_error


def test_pipeline_uses_fallback_when_weights_missing(make_bot, tmp_path):
    gen = HuggingFaceSQLGenerator(ModelConfig(model_name_or_path="nonexistent/model", cache_dir=str(tmp_path)))
    resp = make_bot(gen).ask("How many transactions are there?")
    assert resp.ok and resp.source == "template_fallback"
    assert resp.slm_calls == 1


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    if not HAS_STACK:
        pytest.skip("torch/transformers not installed")
    from tiny_model import build_tiny_causal_model
    d = tmp_path_factory.mktemp("tiny")
    return build_tiny_causal_model(str(d)), str(d)


def test_real_backend_loads_and_generates_offline(tiny_model_dir, monkeypatch):
    """Exercises the actual Transformers/PyTorch code path with a locally built model (no network)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    path, cache = tiny_model_dir
    gen = HuggingFaceSQLGenerator(ModelConfig(model_name_or_path=path, cache_dir=cache, local_files_only=True,
                                              device="cpu", max_new_tokens=8))
    assert gen.available, gen.load_error
    from nl2sql.prompts import Prompt
    out = gen.generate(Prompt("generate", "sys", "CREATE TABLE t (a INT)\n### Question\nq\n### SQL\n"))
    assert isinstance(out, str)


def test_untrained_real_model_is_never_trusted(tiny_model_dir, make_bot, reference, same_results):
    """A real (random-weight) model emits garbage: the loop spends its budget, then the fallback answers."""
    path, cache = tiny_model_dir
    gen = HuggingFaceSQLGenerator(ModelConfig(model_name_or_path=path, cache_dir=cache, local_files_only=True,
                                              device="cpu", max_new_tokens=8))
    resp = make_bot(gen, max_retries=2).ask("What is the total amount by city?")
    assert resp.slm_calls == 3
    assert [a.stage for a in resp.attempts] == ["generate", "repair", "repair", "fallback"]
    assert not any(a.valid for a in resp.attempts[:3])
    assert resp.source == "template_fallback"
    same_results(resp.result.rows, reference("SELECT city, SUM(amount) FROM transactions GROUP BY city"))


@pytest.mark.slow
@pytest.mark.skipif(not HAS_STACK, reason="torch/transformers not installed")
def test_real_model_generates_valid_sql(make_bot):
    gen = HuggingFaceSQLGenerator(ModelConfig())
    if not gen.available:
        pytest.skip(f"model weights not cached: {gen.load_error}")
    resp = make_bot(gen).ask("What is the total amount by city?")
    assert resp.ok
    assert resp.source in ("slm", "slm_repaired", "template_fallback")
