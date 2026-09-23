# """Headless Streamlit UI test (streamlit.testing.AppTest) - the chat renders SQL, results and history."""
# from pathlib import Path

# import pytest

# AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
# APP = str(Path(__file__).resolve().parent.parent / "app.py")


# @pytest.fixture
# def app(db_url, monkeypatch):
#     monkeypatch.setenv("NL2SQL_DB_URL", db_url)
#     import importlib

#     import nl2sql.config as cfg
#     importlib.reload(cfg)  # pick up env var for PipelineConfig defaults
#     import nl2sql.orchestrator as orch
#     monkeypatch.setattr(orch, "PipelineConfig", cfg.PipelineConfig)
#     at = AppTest.from_file(APP, default_timeout=60)
#     yield at
#     importlib.reload(cfg)


# def _ask(at, mode, question):
#     at.run()
#     at.sidebar.radio[0].set_value(mode).run()
#     at.chat_input[0].set_value(question).run()
#     assert not at.exception, at.exception
#     return at


# def test_template_mode_answers(app):
#     at = _ask(app, "Template backend only", "total amount by city")
#     codes = [c.value for c in at.code]
#     assert any("GROUP BY city" in c for c in codes)
#     assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 6
#     assert any("fallback" in m.value.lower() for m in at.markdown)


# def test_demo_mode_shows_repair_loop(app):
#     at = _ask(app, "Demo: faulty SLM (shows repair loop)", "total amount by city")
#     md = " ".join(m.value for m in at.markdown)
#     assert "SLM after self-repair" in md
#     assert "UNKNOWN_COLUMN" in md and "INVALID_AGGREGATION" in md
#     assert len(at.dataframe) == 1

"""Headless Streamlit UI test (streamlit.testing.AppTest) - the chat renders SQL, results and history."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(db_url, monkeypatch):
    monkeypatch.setenv("NL2SQL_DB_URL", db_url)
    import importlib

    import nl2sql.config as cfg
    importlib.reload(cfg)  # pick up env var for PipelineConfig defaults
    import nl2sql.orchestrator as orch
    monkeypatch.setattr(orch, "PipelineConfig", cfg.PipelineConfig)
    at = AppTest.from_file(APP, default_timeout=60)
    yield at
    importlib.reload(cfg)


def _ask(at, mode, question, source="Built-in sample database"):
    at.run()
    at.sidebar.radio[0].set_value(mode).run()
    at.sidebar.radio[1].set_value(source).run()
    if source != "Built-in sample database":
        at.sidebar.button[0].click().run()  # "Load table"
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    return at


def test_template_mode_answers(app):
    at = _ask(app, "Template backend only", "total amount by city")
    codes = [c.value for c in at.code]
    assert any("GROUP BY city" in c for c in codes)
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 6
    assert any("fallback" in m.value.lower() for m in at.markdown)


def test_demo_mode_shows_repair_loop(app):
    at = _ask(app, "Demo: faulty SLM (shows repair loop)", "total amount by city")
    md = " ".join(m.value for m in at.markdown)
    assert "SLM after self-repair" in md
    assert "UNKNOWN_COLUMN" in md and "INVALID_AGGREGATION" in md
    assert len(at.dataframe) == 1


def test_bank_table_loaded_at_runtime(app):
    """Table + plain-English descriptions supplied in the UI; no code or config change."""
    at = _ask(app, "Template backend only", "How many debit transactions are there?",
              source="Assessment bank sample (CSV + descriptions)")
    assert any("Table `transactions`: 50 rows" in s.value for s in at.sidebar.success)
    assert any("transaction_type = 'Debit'" in c.value for c in at.code)
    assert at.dataframe[0].value.iloc[0, 0] == 28


def test_nothing_loaded_shows_instructions(app):
    app.run()
    app.sidebar.radio[0].set_value("Template backend only").run()
    assert not app.exception
    assert any("Load a table" in i.value for i in app.info)
