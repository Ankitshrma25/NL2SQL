"""Schema metadata model + schema introspection/enrichment.

This module is what makes the pipeline *schema-agnostic*: nothing
downstream knows any table or column name.  Everything is discovered
from the live database (via SQLAlchemy's inspector) and optionally
enriched with a user-supplied JSON file of descriptions and synonyms.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

# Logical types used by the validator and the semantic layer.
NUMERIC = "numeric"
TEXT = "text"
DATE = "date"
BOOLEAN = "boolean"
OTHER = "other"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def logical_type_from_sql(sql_type: str) -> str:
    t = (sql_type or "").upper()
    if any(k in t for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return NUMERIC
    if "BOOL" in t:
        return BOOLEAN
    if "DATE" in t or "TIME" in t:
        return DATE
    if any(k in t for k in ("CHAR", "CLOB", "TEXT", "STRING")):
        return TEXT
    return OTHER


def split_identifier(name: str) -> list[str]:
    """``productName`` / ``product_name`` -> ['product', 'name']."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [p for p in re.split(r"[\s_\-]+", name.lower()) if p]


@dataclass
class ColumnInfo:
    name: str
    table: str
    sql_type: str
    logical_type: str
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    primary_key: bool = False
    nullable: bool = True
    is_categorical: bool = False
    is_identifier: bool = False  
    sample_values: list = field(default_factory=list)
    distinct_count: int | None = None
    min_value: object = None
    max_value: object = None

    @property
    def phrases(self) -> list[str]:
        """All natural-language phrases that may refer to this column."""
        base = " ".join(split_identifier(self.name))
        out = {base, self.name.lower()}
        out.update(s.lower() for s in self.synonyms)
        return sorted(p for p in out if p)


@dataclass
class ForeignKey:
    table: str
    columns: list[str]
    ref_table: str
    ref_columns: list[str]


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo]
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    row_count: int | None = None
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def column(self, name: str) -> ColumnInfo | None:
        low = name.lower()
        for c in self.columns:
            if c.name.lower() == low:
                return c
        return None

    @property
    def phrases(self) -> list[str]:
        out = {" ".join(split_identifier(self.name)), self.name.lower()}
        out.update(s.lower() for s in self.synonyms)
        return sorted(out)


@dataclass
class SchemaMetadata:
    tables: list[TableInfo]
    dialect: str = "sqlite"

    # lookups 
    def table(self, name: str) -> TableInfo | None:
        low = name.lower()
        for t in self.tables:
            if t.name.lower() == low:
                return t
        return None

    def all_columns(self) -> Iterable[ColumnInfo]:
        for t in self.tables:
            yield from t.columns

    def find_columns(self, name: str) -> list[ColumnInfo]:
        low = name.lower()
        return [c for c in self.all_columns() if c.name.lower() == low]

    # rendering for prompts 
    def to_ddl(self, tables: Iterable[str] | None = None) -> str:
        """CREATE TABLE statements - the format most SQL models were trained on."""
        wanted = {t.lower() for t in tables} if tables else None
        chunks = []
        for t in self.tables:
            if wanted and t.name.lower() not in wanted:
                continue
            lines = []
            for c in t.columns:
                line = f"  {c.name} {c.sql_type or 'TEXT'}"
                if c.primary_key:
                    line += " PRIMARY KEY"
                lines.append(line)
            for fk in t.foreign_keys:
                lines.append(
                    f"  FOREIGN KEY ({', '.join(fk.columns)}) REFERENCES "
                    f"{fk.ref_table}({', '.join(fk.ref_columns)})"
                )
            chunks.append(f"CREATE TABLE {t.name} (\n" + ",\n".join(lines) + "\n);")
        return "\n\n".join(chunks)

    def describe(self, tables: Iterable[str] | None = None, max_values: int = 25) -> str:
        """Semantic annotations (descriptions, types, synonyms, sample values)."""
        wanted = {t.lower() for t in tables} if tables else None
        out = []
        for t in self.tables:
            if wanted and t.name.lower() not in wanted:
                continue
            head = f"Table {t.name}"
            if t.description:
                head += f": {t.description}"
            if t.row_count is not None:
                head += f" ({t.row_count} rows)"
            out.append(head)
            for c in t.columns:
                bits = [f"{c.logical_type}"]
                if c.is_identifier:
                    bits.append("identifier")
                if c.is_categorical:
                    bits.append("categorical")
                desc = f" - {c.description.rstrip('.')}" if c.description else ""
                line = f"  - {c.name} ({', '.join(bits)}){desc}"
                if c.synonyms:
                    line += f"; also called: {', '.join(c.synonyms)}"
                if c.is_categorical and c.sample_values:
                    vals = ", ".join(repr(v) for v in c.sample_values[:max_values])
                    line += f"; values: {vals}"
                elif c.logical_type in (NUMERIC, DATE) and c.min_value is not None:
                    line += f"; range: {c.min_value} .. {c.max_value}"
                out.append(line)
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "dialect": self.dialect,
            "tables": [
                {
                    "name": t.name,
                    "description": t.description,
                    "row_count": t.row_count,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.sql_type,
                            "logical_type": c.logical_type,
                            "description": c.description,
                            "synonyms": c.synonyms,
                            "categorical": c.is_categorical,
                            "identifier": c.is_identifier,
                            "sample_values": c.sample_values,
                        }
                        for c in t.columns
                    ],
                }
                for t in self.tables
            ],
        }



# Introspection

def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def introspect_schema(
    engine: Engine,
    metadata_path: str | Path | None = None,
    categorical_max_distinct: int = 25,
    metadata: dict | None = None,
) -> SchemaMetadata:
    """Build :class:`SchemaMetadata` from a live database.

    1. Structural facts (tables, columns, declared types, PK/FK) from the
       SQLAlchemy inspector - works for any SQLAlchemy dialect.
    2. Data profiling (row counts, distinct counts, sample values, ranges)
       so the model and validator can reason about values.
    3. Type inference for loosely typed columns (SQLite lets TEXT hold
       numbers / dates).
    4. Optional human metadata (descriptions, synonyms) merged on top -
       from a JSON file (``metadata_path``) and/or a dict supplied at
       runtime (``metadata``, e.g. descriptions typed in by the user).
    """
    insp = inspect(engine)
    tables: list[TableInfo] = []
    for tname in insp.get_table_names():
        pk_cols = set((insp.get_pk_constraint(tname) or {}).get("constrained_columns") or [])
        fks = [
            ForeignKey(tname, fk["constrained_columns"], fk["referred_table"], fk["referred_columns"])
            for fk in insp.get_foreign_keys(tname)
        ]
        fk_cols = {c for fk in fks for c in fk.columns}
        cols = []
        for col in insp.get_columns(tname):
            sql_type = str(col["type"]) if col.get("type") is not None else ""
            name = col["name"]
            parts = split_identifier(name)
            # *_id, *_key, *_number, *_no, *_code are identifiers (never summed, never a default measure)
            is_id = name in pk_cols or name in fk_cols or bool(parts and parts[-1] in ("id", "key", "number", "no", "code"))
            cols.append(
                ColumnInfo(
                    name=name,
                    table=tname,
                    sql_type=sql_type,
                    logical_type=logical_type_from_sql(sql_type),
                    primary_key=name in pk_cols,
                    nullable=bool(col.get("nullable", True)),
                    is_identifier=bool(is_id),
                )
            )
        tables.append(TableInfo(name=tname, columns=cols, foreign_keys=fks))

    schema = SchemaMetadata(tables=tables, dialect=engine.dialect.name)
    _profile(engine, schema, categorical_max_distinct)
    if metadata_path and Path(metadata_path).exists():
        apply_metadata_file(schema, metadata_path)
    if metadata:
        apply_metadata(schema, metadata)
    return schema


def _profile(engine: Engine, schema: SchemaMetadata, cat_max: int) -> None:
    with engine.connect() as conn:
        for t in schema.tables:
            t.row_count = conn.execute(text(f"SELECT COUNT(*) FROM {_q(t.name)}")).scalar()
            for c in t.columns:
                qc = _q(c.name)
                qt = _q(t.name)
                c.distinct_count = conn.execute(text(f"SELECT COUNT(DISTINCT {qc}) FROM {qt}")).scalar()
                sample = [
                    r[0]
                    for r in conn.execute(
                        text(f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL ORDER BY {qc} LIMIT {cat_max + 1}")
                    )
                ]
                # Infer a better logical type from actual values when the declared type is vague.
                if c.logical_type in (TEXT, OTHER) and sample:
                    if all(isinstance(v, str) and _DATE_RE.match(v) for v in sample):
                        c.logical_type = DATE
                    elif all(isinstance(v, (int, float)) for v in sample):
                        c.logical_type = NUMERIC
                if c.logical_type in (NUMERIC, DATE):
                    lo, hi = conn.execute(text(f"SELECT MIN({qc}), MAX({qc}) FROM {qt}")).one()
                    c.min_value, c.max_value = lo, hi
                # a code/key whose values repeat (e.g. a branch code) is still categorical;
                # a unique key (one value per row) is not
                repeats = t.row_count is not None and c.distinct_count is not None and c.distinct_count < t.row_count
                if (
                    c.logical_type in (TEXT, BOOLEAN, OTHER)
                    and (not c.is_identifier or repeats)
                    and c.distinct_count is not None
                    and 0 < c.distinct_count <= cat_max
                ):
                    c.is_categorical = True
                    c.sample_values = sample[:cat_max]
                else:
                    c.sample_values = sample[:5]


def apply_metadata_file(schema: SchemaMetadata, path: str | Path) -> None:
    """Merge a JSON metadata file of the form::

        {"tables": {"<table>": {"description": "...", "synonyms": [...],
                     "columns": {"<col>": {"description": "...", "synonyms": [...],
                                            "categorical": true, "identifier": false}}}}}
    Unknown tables/columns are ignored (the file can never introduce
    identifiers that do not exist in the database).
    """
    apply_metadata(schema, json.loads(Path(path).read_text(encoding="utf-8")))


def apply_metadata(schema: SchemaMetadata, data: dict) -> None:
    """Merge a metadata dict (same shape as the JSON file) into ``schema``."""
    for tname, tmeta in (data.get("tables") or {}).items():
        t = schema.table(tname)
        if not t:
            continue
        t.description = tmeta.get("description", t.description)
        t.synonyms = list(tmeta.get("synonyms", t.synonyms))
        for cname, cmeta in (tmeta.get("columns") or {}).items():
            c = t.column(cname)
            if not c:
                continue
            c.description = cmeta.get("description", c.description)
            c.synonyms = list(cmeta.get("synonyms", c.synonyms))
            if "categorical" in cmeta:
                c.is_categorical = bool(cmeta["categorical"])
            if "identifier" in cmeta:
                c.is_identifier = bool(cmeta["identifier"])
            if "logical_type" in cmeta:
                c.logical_type = cmeta["logical_type"]