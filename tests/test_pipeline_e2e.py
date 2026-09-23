"""End-to-end tests: question -> pipeline -> SQL executed on the sample DB,
comparing ACTUAL RESULT SETS with results of hand-written reference SQL.
"""
import pytest

from nl2sql.slm.testing import RuleSLM, UnavailableSLM

# (category, question, reference SQL, ordered?)
CASES = [
    ("normal", "Show transactions in Mumbai",
     "SELECT * FROM transactions WHERE city = 'Mumbai'", False),
    ("normal", "list distinct cities",
     "SELECT DISTINCT city FROM transactions", False),
    ("aggregation", "How many transactions are there?",
     "SELECT COUNT(*) FROM transactions", False),
    ("aggregation", "What is the total revenue?",
     "SELECT SUM(amount) FROM transactions", False),
    ("aggregation", "What is the average quantity for books?",
     "SELECT AVG(quantity) FROM transactions WHERE category = 'Books'", False),
    ("aggregation", "What is the maximum amount?",
     "SELECT MAX(amount) FROM transactions", False),
    ("filter", "How many transactions were paid with UPI?",
     "SELECT COUNT(*) FROM transactions WHERE payment_method = 'UPI'", False),
    ("filter", "Show transactions with amount greater than 50000",
     "SELECT * FROM transactions WHERE amount > 50000", False),
    ("filter", "Total amount of Electronics transactions in 2024",
     "SELECT SUM(amount) FROM transactions WHERE category = 'Electronics' AND strftime('%Y', transaction_date) = '2024'", False),
    ("filter", "refunded transactions not in Mumbai",
     "SELECT * FROM transactions WHERE status = 'Refunded' AND city <> 'Mumbai'", False),
    ("filter", "transactions with quantity between 2 and 3 in Delhi",
     "SELECT * FROM transactions WHERE city = 'Delhi' AND quantity BETWEEN 2 AND 3", False),
    ("group_by", "What is the total amount by city?",
     "SELECT city, SUM(amount) FROM transactions GROUP BY city", False),
    ("group_by", "Average amount per category",
     "SELECT category, AVG(amount) FROM transactions GROUP BY category", False),
    ("group_by", "Number of transactions per payment method",
     "SELECT payment_method, COUNT(*) FROM transactions GROUP BY payment_method", False),
    ("group_by", "total sales by city for completed transactions",
     "SELECT city, SUM(amount) FROM transactions WHERE status = 'Completed' GROUP BY city", False),
    ("top_n", "Top 5 transactions by amount",
     "SELECT * FROM transactions ORDER BY amount DESC LIMIT 5", True),
    ("top_n", "top 3 customers by total amount",
     "SELECT customer_name, SUM(amount) AS s FROM transactions GROUP BY customer_name ORDER BY s DESC LIMIT 3", True),
    ("top_n", "Which city has the highest total sales?",
     "SELECT city, SUM(amount) AS s FROM transactions GROUP BY city ORDER BY s DESC LIMIT 1", True),
    ("top_n", "lowest 2 categories by average amount",
     "SELECT category, AVG(amount) AS a FROM transactions GROUP BY category ORDER BY a ASC LIMIT 2", True),
]
IDS = [f"{c[0]}:{c[1]}" for c in CASES]


@pytest.mark.parametrize("category,question,ref_sql,ordered", CASES, ids=IDS)
def test_deterministic_backend_end_to_end(make_bot, reference, same_results, category, question, ref_sql, ordered):
    """SLM unavailable -> semantic layer + template backend must still answer correctly."""
    bot = make_bot(UnavailableSLM())
    resp = bot.ask(question)
    assert resp.ok, resp.error
    assert resp.source == "template_fallback"
    same_results(resp.result.rows, reference(ref_sql), ordered)


@pytest.mark.parametrize("category,question,ref_sql,ordered", CASES, ids=IDS)
def test_slm_path_end_to_end(make_bot, reference, same_results, category, question, ref_sql, ordered):
    """A well-behaved SLM: its SQL is validated once, executed, and results match."""
    answers = {q: sql for _, q, sql, _ in CASES}
    slm = RuleSLM(lambda q, _p: f"```sql\n{answers[q]};\n```")
    resp = make_bot(slm).ask(question)
    assert resp.ok, resp.error
    assert resp.source == "slm" and slm.calls == 1 and len(resp.attempts) == 1
    assert resp.attempts[0].validation.checks_passed[-1] == "db_dry_run"
    same_results(resp.result.rows, reference(ref_sql), ordered)


def test_response_exposes_sql_results_and_history(make_bot):
    resp = make_bot(RuleSLM(lambda q, p: "SELECT city, COUNT(*) AS n FROM transactions GROUP BY city")).ask("count by city")
    assert resp.sql.startswith("SELECT city")
    assert resp.result.columns == ["city", "n"]
    assert len(resp.result.as_records()) == 6
    h = resp.history()
    assert h[0]["valid"] and h[0]["stage"] == "generate"


def test_empty_question(make_bot):
    resp = make_bot(UnavailableSLM()).ask("   ")
    assert not resp.ok and resp.attempts == []
