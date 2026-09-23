"""Semantic layer: schema enrichment, schema linking and intent detection."""
import pytest


def test_schema_is_introspected_and_enriched(schema):
    t = schema.table("transactions")
    assert t is not None and t.row_count == 120
    city = t.column("city")
    assert city.is_categorical and "Mumbai" in city.sample_values
    assert city.description  # merged from metadata file
    amount = t.column("amount")
    assert amount.logical_type == "numeric" and not amount.is_categorical
    assert "revenue" in amount.synonyms
    assert t.column("transaction_id").is_identifier
    assert t.column("transaction_date").logical_type == "date"  # inferred from TEXT values


def test_context_contains_grounding(semantic):
    ctx = semantic.build_context("total amount by city")
    text = ctx.render()
    assert "CREATE TABLE transactions" in text
    assert "values: 'Bengaluru'" in text  # categorical values surfaced
    assert any("Group by: city" in h for h in ctx.hints)


@pytest.mark.parametrize("question,agg", [
    ("How many transactions are there", "COUNT"),
    ("number of orders", "COUNT"),
    ("total revenue", "SUM"),
    ("average spend", "AVG"),
    ("what is the maximum amount", "MAX"),
    ("minimum quantity", "MIN"),
])
def test_aggregation_intent(semantic, question, agg):
    assert semantic.linker.analyze(question).aggregation == agg


def test_synonym_and_fuzzy_linking(semantic):
    a = semantic.linker.analyze("total revenue per locaton")  # typo in 'location'
    assert a.measure.name == "amount"
    assert [c.name for c in a.group_by] == ["city"]


def test_value_linking_and_negation(semantic):
    a = semantic.linker.analyze("refunded transactions not in Mumbai paid with upi")
    got = {(f.column.name, f.op, f.value) for f in a.filters}
    assert ("status", "=", "Refunded") in got
    assert ("city", "!=", "Mumbai") in got
    assert ("payment_method", "=", "UPI") in got


def test_numeric_and_date_filters(semantic):
    a = semantic.linker.analyze("transactions with amount over 5000 and quantity at least 3 in 2024")
    got = {(f.column.name, f.op, f.value) for f in a.filters}
    assert ("amount", ">", 5000) in got
    assert ("quantity", ">=", 3) in got
    assert ("transaction_date", "YEAR", "2024") in got


def test_top_n(semantic):
    a = semantic.linker.analyze("top 3 customers by total amount")
    assert a.top_n == 3 and a.order_direction == "DESC"
    assert [c.name for c in a.group_by] == ["customer_name"]
    assert a.aggregation == "SUM" and a.measure.name == "amount"


def test_sort_by_is_not_group_by(semantic):
    a = semantic.linker.analyze("show transactions in Pune sorted by amount")
    assert a.group_by == []
