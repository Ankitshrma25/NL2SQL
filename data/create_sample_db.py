"""Create the deterministic sample SQLite database used by the app and tests.

    python data/create_sample_db.py            # writes data/sample.db

The data is generated with a fixed seed so tests can rely on exact results.
The pipeline itself never assumes this schema - see tests/test_schema_agnostic.py.
"""
from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCHEMA_SQL = """
CREATE TABLE transactions (
    transaction_id   INTEGER PRIMARY KEY,
    customer_name    TEXT    NOT NULL,
    city             TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    payment_method   TEXT    NOT NULL,
    status           TEXT    NOT NULL,
    quantity         INTEGER NOT NULL,
    amount           REAL    NOT NULL,
    transaction_date TEXT    NOT NULL
);
"""

CUSTOMERS = ["Aarav Shah", "Priya Nair", "Rohan Mehta", "Ananya Iyer", "Vikram Rao", "Sneha Kapoor",
             "Arjun Singh", "Kavya Reddy", "Rahul Verma", "Meera Joshi", "Karan Malhotra", "Isha Gupta"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Pune"]
CATEGORIES = {"Electronics": (2000, 60000), "Groceries": (200, 3000), "Clothing": (500, 8000),
              "Books": (150, 2000), "Home": (800, 20000)}
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
STATUSES = ["Completed"] * 8 + ["Refunded", "Pending"]


def build_rows(n: int = 120, seed: int = 42) -> list[tuple]:
    rng = random.Random(seed)
    start = date(2023, 1, 1)
    rows = []
    for i in range(1, n + 1):
        cat = rng.choice(list(CATEGORIES))
        lo, hi = CATEGORIES[cat]
        qty = rng.randint(1, 5)
        amount = round(rng.uniform(lo, hi), 2)
        d = start + timedelta(days=rng.randint(0, 729))  # 2023-01-01 .. 2024-12-30
        rows.append((i, rng.choice(CUSTOMERS), rng.choice(CITIES), cat, rng.choice(PAYMENTS),
                     rng.choice(STATUSES), qty, amount, d.isoformat()))
    return rows


def create(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_SQL)
    con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)", build_rows())
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sample.db"
    print("Created", create(out))
