import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import create_sample_db  # type: ignore  # noqa: E402  (found via sys.path above, at runtime)

from nl2sql.config import PipelineConfig  # noqa: E402
from nl2sql.executor import DatabaseExecutor, make_engine  # noqa: E402
from nl2sql.fallback import TemplateFallback  # noqa: E402
from nl2sql.orchestrator import NL2SQLChatbot  # noqa: E402
from nl2sql.semantic_layer import SemanticLayer  # noqa: E402
from nl2sql.validator import SQLValidator  # noqa: E402

METADATA = ROOT / "data" / "schema_metadata.json"


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    """A fresh copy of the deterministic sample DB for the test session."""
    return create_sample_db.create(tmp_path_factory.mktemp("db") / "sample.db")


@pytest.fixture(scope="session")
def db_url(db_path) -> str:
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="session")
def engine(db_url):
    return make_engine(db_url)


@pytest.fixture(scope="session")
def semantic(engine) -> SemanticLayer:
    return SemanticLayer.from_engine(engine, str(METADATA))


@pytest.fixture(scope="session")
def schema(semantic):
    return semantic.schema


@pytest.fixture(scope="session")
def executor(engine) -> DatabaseExecutor:
    return DatabaseExecutor(engine, max_rows=1000)


@pytest.fixture(scope="session")
def validator(schema, executor) -> SQLValidator:
    return SQLValidator(schema, dry_run=executor.dry_run)


@pytest.fixture
def make_bot(semantic, validator, executor, db_url):
    def _make(slm, max_retries: int = 2, enable_fallback: bool = True) -> NL2SQLChatbot:
        cfg = PipelineConfig(db_url=db_url, max_repair_retries=max_retries, enable_fallback=enable_fallback)
        return NL2SQLChatbot(semantic, validator, executor, slm, TemplateFallback(), cfg)
    return _make


@pytest.fixture(scope="session")
def reference(db_path):
    """Run hand-written ground-truth SQL directly with sqlite3 (independent of the pipeline)."""
    def _run(sql: str) -> list[tuple]:
        con = sqlite3.connect(db_path)
        try:
            return [tuple(r) for r in con.execute(sql).fetchall()]
        finally:
            con.close()
    return _run


def _norm(rows, ordered):
    out = [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows]
    return out if ordered else sorted(out, key=repr)


def assert_same_results(actual_rows, expected_rows, ordered=False):
    a, e = _norm(actual_rows, ordered), _norm(expected_rows, ordered)
    assert a == e, f"result sets differ\nactual:   {a[:10]}\nexpected: {e[:10]}"


@pytest.fixture
def same_results():
    return assert_same_results