
"""Streamlit chatbot UI.

    streamlit run app.py

1. Pick a table: the built-in sample database, the bundled assessment bank
   sample, or upload any CSV and describe its columns in plain English.
2. Ask questions. Each answer shows the final SQL, where it came from (SLM,
   SLM after repair, or template fallback), the result table and the full
   attempt history (each candidate SQL with its validator verdict).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from nl2sql.config import DATA_DIR, PipelineConfig
from nl2sql.ingest import load_table, sanitize_identifier, schema_template
from nl2sql.orchestrator import NL2SQLChatbot
from nl2sql.prompts import Prompt
from nl2sql.slm.base import SQLGenerator

st.set_page_config(page_title="Offline NL→SQL Chatbot", page_icon="🗄️", layout="wide")

SOURCE_LABEL = {
    "slm": "✅ SLM (first attempt)",
    "slm_repaired": "🔧 SLM after self-repair",
    "template_fallback": "🛟 Deterministic template fallback",
    "none": "❌ No valid SQL",
}
MODE_SLM = "Offline SLM (Transformers)"
MODE_DEMO = "Demo: faulty SLM (shows repair loop)"
MODE_TEMPLATE = "Template backend only"

SRC_SAMPLE = "Built-in sample database"
SRC_BANK = "Assessment bank sample (CSV + descriptions)"
SRC_UPLOAD = "Upload your own CSV"
BANK_CSV = DATA_DIR / "task2_bank_transactions_sample.csv"
BANK_SCHEMA = DATA_DIR / "task2_bank_schema.txt"


class FaultInjectingDemoSLM(SQLGenerator):
    """DEMO ONLY: makes the repair loop visible without a real model.

    Attempt 1 returns SQL with a misspelled column, attempt 2 a SUM over a
    text column; repair attempts after that return the semantic layer's SQL.
    """

    name = "demo-faulty-slm"

    def __init__(self, bot_ref: dict):
        self.bot_ref = bot_ref

    def generate(self, prompt: Prompt) -> str:
        bot: NL2SQLChatbot = self.bot_ref["bot"]
        question = prompt.user.split("### Question\n", 1)[-1].split("\n", 1)[0]
        ctx = bot.semantic.build_context(question)
        good = bot.fallback.generate(ctx.analysis).sql or "SELECT 1"
        table = ctx.analysis.primary_table
        text_col = next((c.name for c in table.columns if c.logical_type == "text"), None) if table else None
        if prompt.attempt == 0:
            col = (ctx.analysis.measure or (table.columns[0] if table else None))
            return good.replace(col.name, col.name[:-1] + "x", 1) if col and col.name in good else good + " WHERE nosuchcol = 1"
        if prompt.attempt == 1 and table and text_col:
            return f"SELECT SUM({text_col}) FROM {table.name}"
        return f"```sql\n{good}\n```"


@st.cache_resource(show_spinner="Loading the offline model…")
def get_hf_model():
    """The model is loaded once per process and shared by every table the user loads."""
    from nl2sql.slm.hf_backend import HuggingFaceSQLGenerator
    gen = HuggingFaceSQLGenerator(PipelineConfig().model)
    gen.available  # trigger loading now (inside the spinner), not on the first question
    return gen


def make_bot(mode: str, retries: int, loaded=None) -> NL2SQLChatbot:
    cfg = PipelineConfig(max_repair_retries=retries)
    ref: dict = {}
    slm = get_hf_model() if mode == MODE_SLM else FaultInjectingDemoSLM(ref) if mode == MODE_DEMO else None
    if loaded is not None:
        bot = NL2SQLChatbot.from_loaded(loaded, cfg, slm=slm, use_hf_model=False)
        bot.loaded = loaded
    else:
        bot = NL2SQLChatbot.from_config(cfg, slm=slm, use_hf_model=False)
    ref["bot"] = bot
    return bot


# ---------------- sidebar: model settings ----------------
with st.sidebar:
    st.header("Model")
    mode = st.radio("SQL generator", [MODE_SLM, MODE_DEMO, MODE_TEMPLATE])
    retries = st.slider("Repair retries after first attempt", 0, 5, 2)
    show_prompts = st.checkbox("Show prompts sent to the SLM", value=False)

    # ---------------- sidebar: table supplied at runtime ----------------
    st.header("Table")
    source = st.radio("Data source", [SRC_BANK, SRC_UPLOAD, SRC_SAMPLE], label_visibility="collapsed")

    if source in (SRC_BANK, SRC_UPLOAD):
        csv_path = None
        if source == SRC_BANK:
            csv_path, default_schema, default_name = BANK_CSV, BANK_SCHEMA.read_text(encoding="utf-8"), "transactions"
            st.caption(f"`{BANK_CSV.name}` + its column descriptions. Edit anything below, then load.")
        else:
            up = st.file_uploader("CSV file", type=["csv"])
            default_schema, default_name = "", "data"
            if up is not None:
                tmp = Path(tempfile.gettempdir()) / f"nl2sql_upload_{sanitize_identifier(up.name)}.csv"
                tmp.write_bytes(up.getvalue())
                csv_path, default_schema = tmp, schema_template(tmp)
                default_name = sanitize_identifier(Path(up.name).stem, "t")
        key = f"{source}:{csv_path.name if csv_path else ''}"
        table_name = st.text_input("Table name", value=default_name, key=f"name:{key}")
        schema_text = st.text_area(
            "Column descriptions (one `column: description` per line)", value=default_schema, height=260,
            key=f"schema:{key}",
            help="Plain English. Optional type: `amount (REAL): Transaction amount`. JSON {column: description} also works.")
        if st.button("Load table", type="primary", disabled=csv_path is None and not schema_text.strip()):
            try:
                st.session_state[f"loaded:{source}"] = load_table(csv_path, schema_text, table_name)
                st.session_state.history = []
            except Exception as e:
                st.error(f"Could not load the table: {e}")
        loaded = st.session_state.get(f"loaded:{source}")
    else:
        loaded = None

# ---------------- build the chatbot for the current table ----------------
if source != SRC_SAMPLE and loaded is None:
    st.title("🗄️ Offline Natural Language → SQL")
    st.info("Load a table in the sidebar to start: the bank sample is pre-filled, or upload any CSV and describe its columns.")
    st.stop()

bot_key = (mode, retries, str(loaded.db_path) if (loaded is not None and source != SRC_SAMPLE) else "sample")
if st.session_state.get("bot_key") != bot_key:
    try:
        st.session_state.bot = make_bot(mode, retries, loaded if source != SRC_SAMPLE else None)
        st.session_state.bot_key = bot_key
    except Exception as e:  # e.g. sample database file missing
        st.error(f"Could not start the pipeline: {e}")
        st.info("Create the sample database with `python data/create_sample_db.py`.")
        st.stop()
bot: NL2SQLChatbot = st.session_state.bot

with st.sidebar:
    if mode == MODE_SLM and bot.slm is not None and not bot.slm.available:
        st.warning("Model weights not found – the template fallback will answer. "
                   "See docs/MODEL_SETUP.md to download them once.")
    if bot.loaded is not None:
        ld = bot.loaded
        st.success(f"Table `{ld.table_name}`: {ld.row_count} rows · {len(ld.columns)} columns · "
                   f"{sum(1 for d in ld.descriptions.values() if d)} described")
        for w in ld.warnings:
            st.caption("ℹ️ " + w)
    with st.expander("Schema the model sees", expanded=False):
        st.code(bot.semantic.schema.to_ddl(), language="sql")
        st.text(bot.semantic.schema.describe())

st.title("🗄️ Offline Natural Language → SQL")
st.caption("Semantic layer → offline SLM → multi-factor validator → bounded self-repair → template fallback → read-only execution")

if "history" not in st.session_state:
    st.session_state.history = []


def render(resp):
    st.markdown(f"**Source:** {SOURCE_LABEL.get(resp.source, resp.source)} · "
                f"{resp.slm_calls} SLM call(s) · {resp.elapsed_ms:.0f} ms")
    if resp.sql:
        st.code(resp.sql, language="sql", wrap_lines=True)
    if resp.ok:
        df = pd.DataFrame(resp.result.rows, columns=resp.result.columns)
        st.dataframe(df, use_container_width=True, hide_index=True)
        note = f"{len(df)} row(s)"
        if resp.result.truncated:
            note += " (truncated)"
        st.caption(note)
    else:
        st.error(resp.error)
    with st.expander(f"Attempt history ({len(resp.attempts)})", expanded=resp.source != "slm"):
        for a in resp.attempts:
            icon = "✅" if a.valid else "❌"
            st.markdown(f"{icon} **Attempt {a.number} – {a.stage}** · `{a.backend}`")
            st.code(a.sql or "<empty>", language="sql", wrap_lines=True)
            s = a.summary()
            for err in s["errors"]:
                st.markdown(f"- 🔴 {err}")
            for w in s["warnings"]:
                st.markdown(f"- 🟡 {w}")
            if s["checks_passed"]:
                st.caption("checks passed: " + ", ".join(s["checks_passed"]))
            if show_prompts and a.prompt:
                st.text_area("Prompt", a.prompt, height=200, key=f"p{id(resp)}{a.number}")
    if resp.context is not None:
        with st.expander("Semantic layer hints"):
            for h in resp.context.hints:
                st.markdown(f"- {h}")


for q, resp in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        render(resp)

question = st.chat_input("Ask a question about the table…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = bot.ask(question)
        render(resp)
    st.session_state.history.append((question, resp))
