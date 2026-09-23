"""Prompt construction, SQL extraction and the execution-layer safety net."""
import sqlite3

import pytest

from nl2sql.executor import DatabaseExecutor, make_engine
from nl2sql.prompts import build_generation_prompt, build_repair_prompt, extract_sql


@pytest.mark.parametrize("raw,expected", [
    ("SELECT * FROM t", "SELECT * FROM t"),
    ("```sql\nSELECT a FROM t;\n```", "SELECT a FROM t"),
    ("Here is the query:\nSELECT a\nFROM t\nWHERE b = 1;\n\nThis returns all a.", "SELECT a\nFROM t\nWHERE b = 1"),
    ("SQL: SELECT 1", "SELECT 1"),
    ("WITH x AS (SELECT 1 AS a) SELECT a FROM x", "WITH x AS (SELECT 1 AS a) SELECT a FROM x"),
    ("DROP TABLE t; SELECT 1", "DROP TABLE t; SELECT 1"),  # kept so the validator rejects it
    ("", ""),
    (None, ""),
])
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


def test_generation_prompt_is_grounded(semantic):
    p = build_generation_prompt(semantic.build_context("average amount per city"))
    assert p.kind == "generate"
    assert "CREATE TABLE transactions" in p.user
    assert "average amount per city" in p.user
    assert "Hints from question analysis" in p.user


def test_repair_prompt_contains_all_required_parts(semantic):
    ctx = semantic.build_context("average amount per city")
    p = build_repair_prompt(ctx, "SELECT AVG(amout) FROM transactions", "- [UNKNOWN_COLUMN] Column 'amout' does not exist.",
                            attempt=1, history=[("SELECT AVG(amout) FROM transactions", "x")])
    assert p.kind == "repair" and p.attempt == 1
    assert "average amount per city" in p.user                   # original question
    assert "CREATE TABLE transactions" in p.user                 # schema / semantic metadata
    assert "SELECT AVG(amout) FROM transactions" in p.user       # previous SQL
    assert "[UNKNOWN_COLUMN] Column 'amout' does not exist." in p.user  # exact validator error


def test_executor_only_accepts_validated_queries(executor):
    with pytest.raises(TypeError):
        executor.execute("SELECT * FROM transactions")  # raw string is refused


def test_executor_connection_is_read_only(db_url, validator):
    ex = DatabaseExecutor(make_engine(db_url))
    # Even if something slipped past validation, the connection itself refuses writes.
    with ex.engine.connect() as conn, pytest.raises(Exception):
        conn.exec_driver_sql("DELETE FROM transactions")


def test_executor_truncates_large_results(engine, validator):
    ex = DatabaseExecutor(engine, max_rows=10)
    res = ex.execute(validator.validate("SELECT * FROM transactions").query)
    assert len(res.rows) == 10 and res.truncated


def test_dry_run_does_not_execute(db_path, executor):
    before = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert executor.dry_run("SELECT * FROM transactions") is None
    assert executor.dry_run("SELECT bogus_fn(1)") is not None
    after = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert before == after
