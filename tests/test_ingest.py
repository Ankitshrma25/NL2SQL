"""Runtime table loading: CSV + plain-English column descriptions -> working chatbot, no code changes."""
import json
import sqlite3
from pathlib import Path

import pytest

from nl2sql.ingest import load_table, parse_schema_text, sanitize_identifier
from nl2sql.orchestrator import NL2SQLChatbot
from nl2sql.prompts import build_generation_prompt
from nl2sql.slm.testing import RuleSLM, UnavailableSLM

DATA = Path(__file__).resolve().parent.parent / "data"
BANK_CSV = DATA / "task2_bank_transactions_sample.csv"
BANK_SCHEMA = (DATA / "task2_bank_schema.txt").read_text(encoding="utf-8")
SPEC = json.loads((DATA / "task2_validation.json").read_text(encoding="utf-8"))


# ---------------- schema text parsing ----------------
def test_parse_plain_lines_with_colons_in_description():
    descs, types = parse_schema_text(BANK_SCHEMA)
    assert len(descs) == 10
    assert descs["payment_mode"] == "Channel used: UPI / NEFT / IMPS / Card / Cash / Cheque."
    assert types == {}


@pytest.mark.parametrize("text", [
    "amount: Transaction amount\ncity: Where it happened",
    "amount - Transaction amount\ncity - Where it happened",
    "| Column | Description |\n|---|---|\n| amount | Transaction amount |\n| city | Where it happened |",
    "amount\tTransaction amount\ncity\tWhere it happened",
    '{"amount": "Transaction amount", "city": "Where it happened"}',
    '{"columns": [{"name": "amount", "description": "Transaction amount"}, {"name": "city", "description": "Where it happened"}]}',
])
def test_parse_accepts_common_formats(text):
    descs, _ = parse_schema_text(text)
    assert descs == {"amount": "Transaction amount", "city": "Where it happened"}


def test_parse_optional_types():
    descs, types = parse_schema_text("amount (REAL): value\nbooked_on (DATE): when")
    assert types == {"amount": "REAL", "booked_on": "DATE"}


@pytest.mark.parametrize("raw,clean", [("Amount (INR)", "amount_inr"), ("  Customer Name ", "customer_name"),
                                       ("2024 sales", "col_2024_sales"), ("a-b.c", "a_b_c")])
def test_sanitize_identifier(raw, clean):
    assert sanitize_identifier(raw) == clean


# ---------------- loading ----------------
def test_load_bank_csv(tmp_path):
    ld = load_table(BANK_CSV, BANK_SCHEMA, "transactions", workdir=tmp_path)
    assert ld.table_name == "transactions" and ld.row_count == 50 and len(ld.columns) == 10
    assert all(ld.descriptions.values()) and ld.warnings == []
    con = sqlite3.connect(ld.db_path)
    assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 50
    types = {r[1]: r[2] for r in con.execute("PRAGMA table_info(transactions)")}
    assert types["amount"] == "REAL" and types["payment_mode"] == "TEXT"


def test_loaded_database_is_read_only(tmp_path):
    ld = load_table(BANK_CSV, BANK_SCHEMA, "transactions", workdir=tmp_path)
    with ld.engine.connect() as conn, pytest.raises(Exception):
        conn.exec_driver_sql("DELETE FROM transactions")


def test_table_name_defaults_to_file_name(tmp_path):
    assert load_table(BANK_CSV, None, workdir=tmp_path).table_name == "task2_bank_transactions_sample"


def test_warnings_for_missing_and_unknown_descriptions(tmp_path):
    ld = load_table(BANK_CSV, "amount: Transaction amount\nnot_a_column: ???", "t", workdir=tmp_path)
    text = " ".join(ld.warnings)
    assert "not_a_column" in text and "No description for" in text and "payment_mode" in text


def test_messy_csv_headers_are_sanitised_and_descriptions_still_match(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("Order ID,Unit Price ($),Ship City\n1,9.5,Pune\n2,20,Delhi\n")
    ld = load_table(csv, "Order ID: id\nUnit Price ($): price per unit\nShip City: destination city", workdir=tmp_path)
    assert ld.columns == ["order_id", "unit_price", "ship_city"]
    assert ld.descriptions == {"order_id": "id", "unit_price": "price per unit", "ship_city": "destination city"}


def test_schema_only_mode_generates_and_validates_sql(tmp_path):
    bot = NL2SQLChatbot.from_csv(None, "amount (REAL): Transaction amount\nmode: Payment channel", "payments",
                                 slm=RuleSLM(lambda q, p: "SELECT SUM(amount) FROM payments"), use_hf_model=False)
    resp = bot.ask("total amount")
    assert resp.ok and resp.source == "slm" and resp.result.rows == [(None,)]
    assert any("No data supplied" in w for w in bot.loaded.warnings)


def test_nothing_supplied_is_an_error():
    with pytest.raises(ValueError):
        load_table(None, "")


# ---------------- descriptions reach the model ----------------
def test_descriptions_and_values_are_in_the_prompt():
    bot = NL2SQLChatbot.from_csv(BANK_CSV, BANK_SCHEMA, "transactions", use_hf_model=False)
    prompt = build_generation_prompt(bot.semantic.build_context("total amount credited")).user
    assert "Whether the transaction was a Credit or a Debit" in prompt
    assert "'Credit', 'Debit'" in prompt
    assert "'Transfer'" in prompt and "'Utilities'" in prompt  # every categorical value listed


# ---------------- the 10 assessment questions ----------------
@pytest.fixture(scope="module")
def bank_bot():
    answers = {q["question"]: q["expected_sql"] for q in SPEC["questions"]}
    slm = RuleSLM(lambda q, _p: answers[q])  # stands in for a model that answers correctly
    return NL2SQLChatbot.from_csv(BANK_CSV, BANK_SCHEMA, "transactions", slm=slm, use_hf_model=False)


@pytest.mark.parametrize("case", SPEC["questions"], ids=[f"Q{q['id']}" for q in SPEC["questions"]])
def test_validation_questions_pass_through_pipeline(bank_bot, same_results, case):
    """Expected SQL passes the validator unchanged and returns the reference rows."""
    resp = bank_bot.ask(case["question"])
    assert resp.ok and resp.source == "slm", resp.error
    con = sqlite3.connect(bank_bot.loaded.db_path)
    same_results(resp.result.rows, con.execute(case["expected_sql"]).fetchall(), case["ordered"])


def test_template_only_baseline_is_tracked():
    """The deterministic path alone answers all 10 (the safety net if the SLM fails)."""
    bot = NL2SQLChatbot.from_csv(BANK_CSV, BANK_SCHEMA, "transactions", slm=UnavailableSLM(), use_hf_model=False)
    con = sqlite3.connect(bot.loaded.db_path)
    passed = 0
    for q in SPEC["questions"]:
        r = bot.ask(q["question"])
        exp = con.execute(q["expected_sql"]).fetchall()
        norm = lambda rows: sorted([tuple(round(v, 4) if isinstance(v, float) else v for v in x) for x in rows], key=repr)  # noqa: E731
        passed += bool(r.ok and norm(r.result.rows) == norm(exp))
    assert passed == 10


# ---------------- swapping in a different table: no code changes ----------------
def test_swap_in_unrelated_table(tmp_path):
    csv = tmp_path / "flights.csv"
    csv.write_text("flight_no,airline,origin,delay_minutes,seats_sold\n"
                   "AI101,Air India,DEL,15,180\nAI202,Air India,BOM,0,150\n6E301,IndiGo,DEL,45,176\n"
                   "6E402,IndiGo,BLR,5,160\nUK501,Vistara,DEL,30,140\n")
    schema = ("flight_no: Flight number\nairline: Operating airline\norigin: Departure airport code\n"
              "delay_minutes: Departure delay in minutes\nseats_sold: Tickets sold")
    bot = NL2SQLChatbot.from_csv(csv, schema, "flights", slm=UnavailableSLM(), use_hf_model=False)
    r = bot.ask("average delay minutes per airline")
    assert r.ok and "GROUP BY airline" in r.sql
    assert dict(r.result.rows)["IndiGo"] == 25
    assert bot.ask("how many flights from DEL").result.rows == [(3,)]


# ---------------- hints the SLM receives (a wrong hint misleads the model) ----------------
@pytest.fixture(scope="module")
def bank_semantic():
    return NL2SQLChatbot.from_csv(BANK_CSV, BANK_SCHEMA, "transactions", use_hf_model=False).semantic


def _hints(sem, q):
    return " | ".join(sem.build_context(q).hints)


def test_hint_filter_uses_the_named_column(bank_semantic):
    h = _hints(bank_semantic, "List all transactions where the balance after the transaction is below 50000.")
    assert "balance_after_transaction < 50000" in h and "account_number" not in h


def test_hint_inflected_value(bank_semantic):
    assert "transaction_type = 'Credit'" in _hints(bank_semantic, "What is the total amount credited across all transactions?")


def test_hint_sorted_by_column_and_full_rows(bank_semantic):
    h = _hints(bank_semantic, "List all transactions done through Cheque, sorted by date.")
    assert "Sort by: transaction_date ASC" in h and "SELECT *" in h and "transaction_id" not in h


def test_hint_top_n_sorts_by_measure_not_identifier(bank_semantic):
    h = _hints(bank_semantic, "Show the top 5 highest value transactions.")
    assert "Sort by: amount DESC" in h and "Limit: 5" in h


def test_codes_are_identifiers_but_still_categorical(bank_semantic):
    t = bank_semantic.schema.table("transactions")
    assert t.column("account_number").is_identifier and t.column("branch_code").is_identifier
    assert t.column("branch_code").is_categorical and "BLR001" in t.column("branch_code").sample_values
    assert not t.column("transaction_id").is_categorical  # unique per row


@pytest.mark.parametrize("question,column,values", [
    ("all the transaction type", "transaction_type", [("Credit",), ("Debit",)]),
    ("list all payment modes", "payment_mode", [("Card",), ("Cash",), ("Cheque",), ("IMPS",), ("NEFT",), ("UPI",)]),
])
def test_asking_for_a_columns_values_returns_distinct_values(question, column, values):
    bot = NL2SQLChatbot.from_csv(BANK_CSV, BANK_SCHEMA, "transactions", slm=UnavailableSLM(), use_hf_model=False)
    r = bot.ask(question)
    assert f"SELECT DISTINCT {column}" in r.sql and r.result.rows == values
    assert f"DISTINCT values of {column}" in " ".join(bot.semantic.build_context(question).hints)


def test_filter_on_the_column_keeps_a_row_lookup():
    bot = NL2SQLChatbot.from_csv(BANK_CSV, BANK_SCHEMA, "transactions", slm=UnavailableSLM(), use_hf_model=False)
    assert bot.ask("List all UPI transactions.").sql.startswith("SELECT *")