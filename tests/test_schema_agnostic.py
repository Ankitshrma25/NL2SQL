"""The same pipeline code works on a completely different, multi-table schema."""
import sqlite3

import pytest

from nl2sql.config import PipelineConfig
from nl2sql.orchestrator import NL2SQLChatbot
from nl2sql.slm.testing import ScriptedSLM, UnavailableSLM


@pytest.fixture(scope="module")
def hr_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("hr") / "hr.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, dept_name TEXT NOT NULL, floor INTEGER);
        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY, full_name TEXT, dept_id INTEGER REFERENCES departments(dept_id),
            job_title TEXT, salary REAL, hire_date TEXT);
        INSERT INTO departments VALUES (1,'Engineering',3),(2,'Sales',1),(3,'Finance',2);
        INSERT INTO employees VALUES
          (1,'Ann Lee',1,'Engineer',120000,'2020-03-01'),(2,'Bob Stone',1,'Engineer',110000,'2021-06-15'),
          (3,'Cara Diaz',1,'Manager',150000,'2019-01-10'),(4,'Dan Wu',2,'Sales Rep',70000,'2022-02-01'),
          (5,'Eve Kim',2,'Manager',95000,'2018-11-20'),(6,'Fay Ola',3,'Analyst',80000,'2023-05-05'),
          (7,'Gus Roy',2,'Sales Rep',72000,'2023-08-08');
    """)
    con.commit()
    con.close()
    return path


@pytest.fixture
def hr_bot(hr_db):
    def _make(slm):
        cfg = PipelineConfig(db_url=f"sqlite:///{hr_db}", metadata_path=None)
        return NL2SQLChatbot.from_config(cfg, slm=slm, use_hf_model=False)
    return _make


def rows(db, sql):
    con = sqlite3.connect(db)
    try:
        return sorted(con.execute(sql).fetchall())
    finally:
        con.close()


def test_schema_discovered_without_metadata(hr_bot):
    bot = hr_bot(UnavailableSLM())
    s = bot.semantic.schema
    assert {t.name for t in s.tables} == {"departments", "employees"}
    emp = s.table("employees")
    assert emp.column("job_title").is_categorical
    assert emp.column("salary").logical_type == "numeric"
    assert emp.column("dept_id").is_identifier
    assert emp.foreign_keys[0].ref_table == "departments"


@pytest.mark.parametrize("question,ref", [
    ("average salary per job title", "SELECT job_title, AVG(salary) FROM employees GROUP BY job_title"),
    ("how many employees are Engineers", "SELECT COUNT(*) FROM employees WHERE job_title = 'Engineer'"),
    ("top 2 employees by salary", "SELECT * FROM employees ORDER BY salary DESC LIMIT 2"),
    ("employees with salary above 100000", "SELECT * FROM employees WHERE salary > 100000"),
])
def test_fallback_on_new_schema(hr_bot, hr_db, question, ref):
    resp = hr_bot(UnavailableSLM()).ask(question)
    assert resp.ok, resp.error
    assert sorted(resp.result.rows) == rows(hr_db, ref)


def test_join_generated_by_slm_is_validated_and_executed(hr_bot, hr_db):
    good = ("SELECT d.dept_name, SUM(e.salary) FROM employees e JOIN departments d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_name")
    slm = ScriptedSLM([
        "SELECT dept_name, SUM(salary) FROM employees GROUP BY dept_name",  # dept_name not in employees
        "SELECT d.dept_name, SUM(e.salary) FROM employees e JOIN departments d ON e.dept_id = d.dept_id GROUP BY dept_id",  # ambiguous
        good,
    ])
    resp = hr_bot(slm).ask("total salary by department name")
    assert resp.source == "slm_repaired"
    assert resp.attempts[0].validation.error_codes == ["UNKNOWN_COLUMN"]
    assert resp.attempts[1].validation.error_codes == ["AMBIGUOUS_COLUMN"]
    assert "departments" in slm.prompts[0].user  # FK-related table included in grounding
    assert sorted(resp.result.rows) == rows(hr_db, good)


def test_no_hardcoded_schema_names_in_core():
    """Guard: core modules must not mention the sample schema's names."""
    from pathlib import Path
    core = Path(__file__).resolve().parent.parent / "nl2sql"
    banned = ["transactions", "customer_name", "payment_method", "transaction_date"]
    for f in core.rglob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{f.name} hard-codes '{word}'"
