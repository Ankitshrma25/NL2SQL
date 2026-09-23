"""Multi-factor validator: each check returns a specific, structured error."""
import pytest

from nl2sql.validator import SQLValidator, ValidatedQuery


def codes(res):
    return res.error_codes


# ---------------- valid queries ----------------
@pytest.mark.parametrize("sql", [
    "SELECT * FROM transactions",
    "SELECT city, SUM(amount) AS total FROM transactions GROUP BY city ORDER BY total DESC LIMIT 3",
    "SELECT t.city FROM transactions AS t WHERE t.amount > 1000",
    "WITH c AS (SELECT city, amount FROM transactions) SELECT city, AVG(amount) FROM c GROUP BY city",
    "SELECT COUNT(*) FROM transactions WHERE strftime('%Y', transaction_date) = '2024'",
    "SELECT customer_name FROM transactions WHERE amount = (SELECT MAX(amount) FROM transactions)",
    "SELECT city FROM transactions UNION SELECT category FROM transactions",
    "SELECT COUNT(DISTINCT city) FROM transactions WHERE payment_method IN ('UPI', 'Cash')",
    "SELECT * FROM transactions WHERE amount BETWEEN 100 AND 500;",
])
def test_valid_queries_pass_all_checks(validator, sql):
    res = validator.validate(sql)
    assert res.is_valid, res.feedback()
    assert isinstance(res.query, ValidatedQuery)
    assert res.checks_passed[-1] == "db_dry_run"
    for check in ("syntax", "read_only", "tables", "columns", "types"):
        assert check in res.checks_passed


# ---------------- syntax / structure ----------------
@pytest.mark.parametrize("sql", ["", None, "   "])
def test_empty(validator, sql):
    assert codes(validator.validate(sql)) == ["EMPTY_SQL"]


@pytest.mark.parametrize("sql", ["SELECT FROM WHERE", "SELECT sum(amount FROM transactions", "SELEC * FROM transactions"])
def test_syntax_errors(validator, sql):
    res = validator.validate(sql)
    assert not res.is_valid
    assert codes(res) == ["SYNTAX_ERROR"]
    assert "line" in res.errors[0].message.lower() or "syntax" in res.errors[0].message.lower()


# ---------------- read-only / destructive ----------------
@pytest.mark.parametrize("sql", [
    "DROP TABLE transactions",
    "DELETE FROM transactions WHERE amount > 0",
    "UPDATE transactions SET amount = 0",
    "INSERT INTO transactions (transaction_id) VALUES (999)",
    "CREATE TABLE x (a INT)",
    "ALTER TABLE transactions ADD COLUMN x INT",
    "PRAGMA table_info(transactions)",
    "ATTACH DATABASE 'evil.db' AS evil",
    "REPLACE INTO transactions (transaction_id) VALUES (1)",
])
def test_destructive_statements_rejected(validator, sql):
    res = validator.validate(sql)
    assert not res.is_valid
    assert "NOT_READ_ONLY" in codes(res)


def test_stacked_statements_rejected(validator):
    res = validator.validate("SELECT * FROM transactions; DROP TABLE transactions")
    assert not res.is_valid
    assert "MULTIPLE_STATEMENTS" in codes(res)
    assert "NOT_READ_ONLY" in codes(res)


def test_destructive_inside_cte_rejected(validator):
    res = validator.validate("WITH d AS (DELETE FROM transactions RETURNING *) SELECT * FROM d")
    assert not res.is_valid


# ---------------- tables / columns / identifiers ----------------
def test_unknown_table_with_suggestion(validator):
    res = validator.validate("SELECT * FROM transaction")
    assert codes(res) == ["UNKNOWN_TABLE"]
    assert "transactions" in res.errors[0].suggestion


def test_unknown_column_with_suggestion(validator):
    res = validator.validate("SELECT SUM(amout) FROM transactions")
    assert codes(res) == ["UNKNOWN_COLUMN"]
    err = res.errors[0]
    assert err.identifier == "amout"
    assert "Did you mean 'amount'" in err.suggestion


def test_unknown_qualified_column(validator):
    res = validator.validate("SELECT t.revenue FROM transactions t")
    assert codes(res) == ["UNKNOWN_COLUMN"]
    assert "transactions" in res.errors[0].message


def test_unknown_alias(validator):
    res = validator.validate("SELECT x.city FROM transactions t")
    assert codes(res) == ["UNKNOWN_TABLE_ALIAS"]


def test_double_quoted_string_literal_is_caught(validator):
    res = validator.validate('SELECT * FROM transactions WHERE city = "Mumbai"')
    assert codes(res) == ["UNKNOWN_COLUMN"]
    assert "single quotes" in res.errors[0].suggestion


def test_select_alias_in_order_by_is_allowed(validator):
    assert validator.validate("SELECT city, SUM(amount) AS s FROM transactions GROUP BY city ORDER BY s").is_valid


# ---------------- type consistency ----------------
def test_numeric_compared_with_text(validator):
    res = validator.validate("SELECT * FROM transactions WHERE amount > 'Mumbai'")
    assert codes(res) == ["TYPE_MISMATCH"]
    assert res.errors[0].identifier == "amount"


def test_numeric_string_is_accepted(validator):
    assert validator.validate("SELECT * FROM transactions WHERE amount > '1000'").is_valid


def test_sum_of_categorical_column(validator):
    res = validator.validate("SELECT SUM(city) FROM transactions")
    assert codes(res) == ["INVALID_AGGREGATION"]
    assert "COUNT" in res.errors[0].suggestion


def test_avg_of_text_column(validator):
    assert codes(validator.validate("SELECT AVG(customer_name) FROM transactions")) == ["INVALID_AGGREGATION"]


def test_date_compared_with_number(validator):
    res = validator.validate("SELECT * FROM transactions WHERE transaction_date = 2024")
    assert codes(res) == ["TYPE_MISMATCH"]
    assert "strftime" in res.errors[0].suggestion


def test_text_range_compared_with_number(validator):
    assert codes(validator.validate("SELECT * FROM transactions WHERE city > 5")) == ["TYPE_MISMATCH"]


def test_wrong_case_categorical_value(validator):
    res = validator.validate("SELECT * FROM transactions WHERE city = 'mumbai'")
    assert codes(res) == ["UNKNOWN_VALUE"]
    assert "'Mumbai'" in res.errors[0].suggestion


def test_warnings_do_not_block(validator):
    res = validator.validate("SELECT city, amount FROM transactions GROUP BY city")
    assert res.is_valid
    assert any(w.code == "UNGROUPED_COLUMN" for w in res.warnings)


# ---------------- database dry run ----------------
def test_db_dry_run_catches_engine_errors(validator):
    res = validator.validate("SELECT no_such_function(amount) FROM transactions")
    assert codes(res) == ["DB_ERROR"]
    assert "no_such_function" in res.errors[0].message
    assert "types" in res.checks_passed  # static checks passed, DB caught it


def test_validator_without_db(schema):
    v = SQLValidator(schema, dry_run=None)
    res = v.validate("SELECT city FROM transactions")
    assert res.is_valid and "db_dry_run" not in res.checks_passed


def test_validated_query_cannot_be_forged():
    with pytest.raises(PermissionError):
        ValidatedQuery("DROP TABLE transactions", object())


def test_feedback_is_specific(validator):
    fb = validator.validate("SELECT SUM(city), amout FROM transactions").feedback()
    assert "amout" in fb and "UNKNOWN_COLUMN" in fb
