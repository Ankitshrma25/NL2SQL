"""Runtime table ingestion: user-supplied CSV + column descriptions -> SQLite.

This is what lets a grader swap in *any* table at runtime with no code
changes:

    loaded = load_table("sales.csv", schema_text, table_name="sales")
    bot = NL2SQLChatbot.from_loaded(loaded)

* The CSV is written into a fresh SQLite file in a temp directory, then
  reopened **read-only** through the normal executor path.
* Column descriptions come as plain text, one ``column: description`` per
  line (``column (TYPE): description`` optionally sets a type), or as a
  ``{column: description}`` dict / JSON. They become metadata that the
  semantic layer puts in the prompt and uses for schema linking.
* With no CSV (schema only), an empty table is created from the schema, so
  the bot still produces and validates SQL; results are simply empty.
"""
from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from .executor import make_engine

# "column: description", "column - description", "column | description", "column<TAB>description"
_LINE = re.compile(r"^\s*([^:|\t]+?)\s*(?:\(\s*([A-Za-z ]+)\s*\))?\s*(?::|\||\t| - | – )\s*(.*)$")
_VALID_TYPES = {"INTEGER", "REAL", "TEXT", "NUMERIC", "DATE", "DATETIME", "BOOLEAN"}


@dataclass
class LoadedTable:
    engine: Engine
    db_path: Path
    table_name: str
    columns: list[str]
    descriptions: dict[str, str]
    row_count: int
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        """Metadata dict in the shape understood by ``schema.apply_metadata``."""
        return {"tables": {self.table_name: {"columns": {c: {"description": d}
                                                         for c, d in self.descriptions.items() if d}}}}


def sanitize_identifier(name: str, fallback: str = "col") -> str:
    """'Amount (INR)' -> 'amount_inr'; always a safe, unquoted SQL identifier."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip()).strip("_").lower()
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"{fallback}_{s}"
    return s


def parse_schema_text(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse the user's schema description.

    Accepts JSON (``{"col": "desc"}`` or ``{"columns": {...}}``) or lines of
    ``column: description`` / ``column (TYPE): description``.  Blank lines,
    ``#`` comments and a markdown/CSV header row are ignored.
    Returns ``(descriptions, declared_types)``.
    """
    text = (text or "").strip()
    if not text:
        return {}, {}
    if text[0] in "{[":
        data = json.loads(text)
        if isinstance(data, dict) and "columns" in data:
            data = data["columns"]
        if isinstance(data, list):  # [{"name":..., "description":...}]
            data = {d["name"]: d.get("description", "") for d in data}
        return {str(k): (v.get("description", "") if isinstance(v, dict) else str(v)) for k, v in data.items()}, {}

    descs: dict[str, str] = {}
    types: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().strip("|").strip()
        if not line or line.startswith("#") or set(line) <= set("-|: "):
            continue
        m = _LINE.match(line)
        if not m:
            # a bare column name with no description is still useful
            if re.fullmatch(r"[\w ]+", line):
                descs.setdefault(line.strip(), "")
            continue
        col, typ, desc = m.group(1).strip(), (m.group(2) or "").strip().upper(), m.group(3).strip()
        if col.lower() in ("column", "column name", "name") and desc.lower() in ("description", "meaning"):
            continue  # header row
        descs[col] = desc
        if typ in _VALID_TYPES:
            types[col] = typ
    return descs, types


def _match_columns(descs: dict[str, str], columns: list[str], raw_to_clean: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Map description keys (as the user typed them) onto the table's real columns."""
    out: dict[str, str] = {}
    warnings: list[str] = []
    by_norm = {sanitize_identifier(c): c for c in columns}
    for key, desc in descs.items():
        target = raw_to_clean.get(key) or by_norm.get(sanitize_identifier(key))
        if target is None:
            warnings.append(f"Description given for unknown column '{key}' - ignored.")
            continue
        out[target] = desc
    missing = [c for c in columns if not out.get(c)]
    if missing:
        warnings.append("No description for: " + ", ".join(missing) + " (the column name alone will be used).")
    return out, warnings


def load_table(
    csv_path: str | Path | None,
    schema: str | dict | None = None,
    table_name: str | None = None,
    workdir: str | Path | None = None,
) -> LoadedTable:
    """Create a read-only SQLite table from a CSV (and/or schema) supplied at runtime."""
    if isinstance(schema, dict):
        descs, types = {str(k): str(v) for k, v in schema.items()}, {}
    else:
        descs, types = parse_schema_text(schema or "")
    if csv_path is None and not descs:
        raise ValueError("Provide a CSV file, a schema description, or both.")

    default_name = Path(csv_path).stem if csv_path else "data"
    table = sanitize_identifier(table_name or default_name, "t")
    workdir = Path(workdir or tempfile.mkdtemp(prefix="nl2sql_"))
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / f"{table}.db"
    if db_path.exists():
        db_path.unlink()

    warnings: list[str] = []
    con = sqlite3.connect(db_path)
    try:
        if csv_path is not None:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            if df.columns.duplicated().any():
                raise ValueError("The CSV has duplicate column names.")
            raw_to_clean = {c: sanitize_identifier(c) for c in df.columns}
            renamed = [f"'{a}' -> {b}" for a, b in raw_to_clean.items() if a != b]
            if renamed:
                warnings.append("Renamed columns to safe SQL identifiers: " + ", ".join(renamed))
            df = df.rename(columns=raw_to_clean)
            # keep text columns as TEXT; pandas maps int -> INTEGER, float -> REAL
            df.to_sql(table, con, index=False, if_exists="replace")
            columns, rows = list(df.columns), len(df)
        else:
            raw_to_clean = {c: sanitize_identifier(c) for c in descs}
            cols_sql = ", ".join(f'"{raw_to_clean[c]}" {types.get(c, "TEXT")}' for c in descs)
            con.execute(f'CREATE TABLE "{table}" ({cols_sql})')
            columns, rows = list(raw_to_clean.values()), 0
            warnings.append("No data supplied: SQL will be generated and validated, but results will be empty.")
        con.commit()
    finally:
        con.close()

    matched, w = _match_columns(descs, columns, raw_to_clean) if descs else ({}, [
        "No column descriptions supplied: only column names and data profiling will ground the model."])
    warnings += w
    return LoadedTable(make_engine(f"sqlite:///{db_path}"), db_path, table, columns, matched, rows, warnings)


def schema_template(csv_path: str | Path) -> str:
    """A 'column: ' line per CSV column, to pre-fill a descriptions editor."""
    cols = pd.read_csv(csv_path, nrows=0).columns
    return "\n".join(f"{c}: " for c in cols)
