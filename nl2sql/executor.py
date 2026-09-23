"""Database execution layer (SQLAlchemy, database-agnostic).

* ``execute`` only accepts a :class:`~nl2sql.validator.ValidatedQuery`,
  so unvalidated model output cannot reach the database by construction.
* SQLite files are opened **read-only** (``mode=ro`` URI) and every
  connection sets ``PRAGMA query_only`` - defence in depth behind the
  validator's read-only check.
* ``dry_run`` runs ``EXPLAIN <sql>`` which compiles the statement in the
  engine (catching unknown functions, misuse of aggregates, ...) without
  executing it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .validator import ValidatedQuery


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False
    elapsed_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_records(self) -> list[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]


def make_engine(db_url: str, read_only: bool = True) -> Engine:
    """Create an engine; SQLite files are opened read-only."""
    if db_url.startswith("sqlite:///") and read_only and ":memory:" not in db_url and "mode=ro" not in db_url:
        path = Path(db_url[len("sqlite:///"):]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        db_url = f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"
    engine = create_engine(db_url)
    if engine.dialect.name == "sqlite" and read_only:
        @event.listens_for(engine, "connect")
        def _query_only(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA query_only = ON")
            cur.close()
    return engine


class DatabaseExecutor:
    def __init__(self, engine: Engine, max_rows: int = 500):
        self.engine = engine
        self.max_rows = max_rows

    @classmethod
    def from_url(cls, db_url: str, max_rows: int = 500) -> "DatabaseExecutor":
        return cls(make_engine(db_url), max_rows)

    def dry_run(self, sql: str) -> str | None:
        """Compile the query in the database without running it. Returns an error message or None."""
        prefix = "EXPLAIN " if self.engine.dialect.name in ("sqlite", "mysql", "postgresql") else ""
        try:
            with self.engine.connect() as conn:
                conn.exec_driver_sql(prefix + sql).fetchall()
            return None
        except SQLAlchemyError as e:
            orig = getattr(e, "orig", None)
            return str(orig) if orig else str(e).splitlines()[0]

    def execute(self, query: ValidatedQuery) -> QueryResult:
        if not isinstance(query, ValidatedQuery):
            raise TypeError("DatabaseExecutor.execute() only accepts a ValidatedQuery produced by SQLValidator")
        t0 = time.perf_counter()
        with self.engine.connect() as conn:
            cur = conn.exec_driver_sql(query.sql)
            columns = list(cur.keys())
            rows = [tuple(r) for r in cur.fetchmany(self.max_rows + 1)]
            conn.rollback()
        truncated = len(rows) > self.max_rows
        return QueryResult(query.sql, columns, rows[: self.max_rows], truncated, (time.perf_counter() - t0) * 1000)
