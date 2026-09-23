"""Self-correction loop, retry budget and fallback - with a fake SLM.

These tests prove the central architectural guarantees:
  * invalid SQL is never executed,
  * validator errors are fed back to the SLM in a repair prompt,
  * the loop is strictly bounded,
  * the deterministic fallback takes over only after the budget is spent.
"""
import sqlite3

import pytest

from nl2sql.slm.base import SLMError
from nl2sql.slm.testing import ScriptedSLM

QUESTION = "What is the total amount by city?"
REF = "SELECT city, SUM(amount) FROM transactions GROUP BY city"

BAD_COLUMN = "SELECT city, SUM(amout) FROM transactions GROUP BY city"              # attempt 1: invalid column
TYPE_ERROR = "SELECT city, SUM(city) FROM transactions GROUP BY city"               # attempt 2: SUM over categorical
GOOD = "SELECT city, SUM(amount) AS total_amount FROM transactions GROUP BY city"   # attempt 3: correct


def test_successful_self_repair_after_two_different_errors(make_bot, reference, same_results):
    slm = ScriptedSLM([BAD_COLUMN, TYPE_ERROR, GOOD])
    resp = make_bot(slm, max_retries=2).ask(QUESTION)

    # outcome
    assert resp.ok and resp.source == "slm_repaired"
    assert resp.sql == GOOD
    same_results(resp.result.rows, reference(REF))

    # attempt history shows the evolution invalid -> invalid -> valid
    assert slm.calls == 3 and len(resp.attempts) == 3
    stages = [a.stage for a in resp.attempts]
    assert stages == ["generate", "repair", "repair"]
    assert [a.valid for a in resp.attempts] == [False, False, True]
    assert resp.attempts[0].validation.error_codes == ["UNKNOWN_COLUMN"]
    assert resp.attempts[1].validation.error_codes == ["INVALID_AGGREGATION"]

    # repair prompts carry question + schema + previous SQL + exact validator error
    p1, p2, p3 = slm.prompts
    assert p1.kind == "generate"
    assert p2.kind == "repair" and p3.kind == "repair"
    for p in (p2, p3):
        assert QUESTION in p.user
        assert "CREATE TABLE transactions" in p.user
    assert BAD_COLUMN in p2.user and "[UNKNOWN_COLUMN]" in p2.user and "amout" in p2.user
    assert "Did you mean 'amount'" in p2.user
    assert TYPE_ERROR in p3.user and "[INVALID_AGGREGATION]" in p3.user
    assert BAD_COLUMN in p3.user  # earlier failure kept in history


def test_repeated_failure_hits_retry_limit_then_falls_back(make_bot, reference, same_results):
    slm = ScriptedSLM([BAD_COLUMN, TYPE_ERROR, "SELECT FROM WHERE", GOOD])  # GOOD would come too late
    resp = make_bot(slm, max_retries=2).ask(QUESTION)

    assert slm.calls == 3, "exactly 1 generation + 2 repairs, never more"
    assert resp.source == "template_fallback"
    assert [a.stage for a in resp.attempts] == ["generate", "repair", "repair", "fallback"]
    assert resp.attempts[2].validation.error_codes == ["SYNTAX_ERROR"]
    assert resp.attempts[3].valid
    same_results(resp.result.rows, reference(REF))


@pytest.mark.parametrize("retries", [0, 1, 2, 3])
def test_retry_budget_strictly_enforced(make_bot, retries):
    slm = ScriptedSLM([BAD_COLUMN])  # always wrong
    resp = make_bot(slm, max_retries=retries).ask(QUESTION)
    assert slm.calls == 1 + retries
    assert resp.slm_calls == 1 + retries
    assert resp.source == "template_fallback"


def test_retry_hard_cap_prevents_runaway_config(make_bot):
    slm = ScriptedSLM([BAD_COLUMN])
    resp = make_bot(slm, max_retries=10_000).ask(QUESTION)
    assert slm.calls == 1 + 5  # PipelineConfig.retry_hard_cap
    assert resp.ok


def test_no_fallback_returns_structured_error(make_bot):
    slm = ScriptedSLM([BAD_COLUMN])
    resp = make_bot(slm, max_retries=1, enable_fallback=False).ask(QUESTION)
    assert not resp.ok and resp.result is None
    assert "UNKNOWN_COLUMN" in resp.error
    assert slm.calls == 2


@pytest.mark.parametrize("bad_sql", [
    "DROP TABLE transactions",
    "DELETE FROM transactions",
    "SELECT * FROM transactions; DROP TABLE transactions",
    "UPDATE transactions SET amount = 0",
])
def test_destructive_slm_output_is_never_executed(make_bot, db_path, bad_sql):
    before = sqlite3.connect(db_path).execute("SELECT COUNT(*), SUM(amount) FROM transactions").fetchone()
    slm = ScriptedSLM([bad_sql, bad_sql, bad_sql])
    resp = make_bot(slm).ask(QUESTION)
    after = sqlite3.connect(db_path).execute("SELECT COUNT(*), SUM(amount) FROM transactions").fetchone()
    assert before == after
    assert all("NOT_READ_ONLY" in a.validation.error_codes for a in resp.attempts[:3])
    assert resp.source == "template_fallback"


def test_type_mismatch_is_repaired(make_bot, reference, same_results):
    slm = ScriptedSLM([
        "SELECT COUNT(*) FROM transactions WHERE amount = 'Mumbai'",
        "SELECT COUNT(*) FROM transactions WHERE city = 'Mumbai'",
    ])
    resp = make_bot(slm).ask("How many transactions in Mumbai?")
    assert resp.source == "slm_repaired" and slm.calls == 2
    assert resp.attempts[0].validation.error_codes == ["TYPE_MISMATCH"]
    assert "[TYPE_MISMATCH]" in slm.prompts[1].user
    same_results(resp.result.rows, reference("SELECT COUNT(*) FROM transactions WHERE city = 'Mumbai'"))


def test_db_level_error_is_repaired(make_bot):
    slm = ScriptedSLM(["SELECT median(amount) FROM transactions", "SELECT AVG(amount) FROM transactions"])
    resp = make_bot(slm).ask("average amount")
    assert resp.attempts[0].validation.error_codes == ["DB_ERROR"]
    assert "median" in slm.prompts[1].user
    assert resp.source == "slm_repaired"


def test_model_exception_counts_as_attempt(make_bot):
    slm = ScriptedSLM([RuntimeError("CUDA OOM"), GOOD])
    resp = make_bot(slm).ask(QUESTION)
    assert resp.source == "slm_repaired" and slm.calls == 2
    assert "CUDA OOM" in resp.attempts[0].error
    assert "[MODEL_ERROR]" in slm.prompts[1].user


def test_missing_weights_skip_straight_to_fallback(make_bot):
    slm = ScriptedSLM([SLMError("Could not load model 'x' offline: not found")])
    resp = make_bot(slm).ask(QUESTION)
    assert slm.calls == 1  # retrying cannot fix missing weights
    assert resp.source == "template_fallback" and resp.ok


def test_first_attempt_success_does_not_repair(make_bot):
    slm = ScriptedSLM([GOOD, BAD_COLUMN])
    resp = make_bot(slm).ask(QUESTION)
    assert slm.calls == 1 and resp.source == "slm"
