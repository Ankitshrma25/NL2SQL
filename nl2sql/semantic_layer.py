"""Semantic layer.

Sits between the user's question and the SLM.  It owns the enriched
schema metadata and, for every question, produces a
:class:`SemanticContext` that bundles:

* the schema pruned to the relevant tables (DDL - the format SQL models
  are trained on) plus semantic annotations (descriptions, logical types,
  synonyms, categorical values, ranges);
* the NLP analysis (linked columns, detected intent, filters) rendered as
  short *hints*.

The same context is re-used for repair prompts so the model is always
grounded in the real schema.  All knowledge is derived from the database
and optional metadata file - no table/column names are hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from .nlp import QuestionAnalysis, SchemaLinker
from .schema import SchemaMetadata, introspect_schema, split_identifier

# Generic, domain-independent abbreviation expansions used to auto-generate
# synonyms for terse column names (qty -> quantity, amt -> amount ...).
_ABBREVIATIONS = {
    "qty": "quantity", "amt": "amount", "num": "number", "no": "number", "dt": "date",
    "desc": "description", "cust": "customer", "prod": "product", "cat": "category",
    "addr": "address", "emp": "employee", "dept": "department", "txn": "transaction",
    "tx": "transaction", "pct": "percent", "avg": "average", "cnt": "count", "ts": "timestamp",
    "yr": "year", "mo": "month", "val": "value", "px": "price", "sal": "salary",
}


@dataclass
class SemanticContext:
    question: str
    analysis: QuestionAnalysis
    tables: list[str]
    schema_ddl: str
    schema_notes: str
    hints: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Compact grounding block used by both generation and repair prompts."""
        parts = ["### Database schema (SQLite)", self.schema_ddl, "", "### Column notes", self.schema_notes]
        if self.hints:
            parts += ["", "### Hints from question analysis (may be incomplete)"]
            parts += [f"- {h}" for h in self.hints]
        return "\n".join(parts)


class SemanticLayer:
    def __init__(self, schema: SchemaMetadata, fuzzy_threshold: int = 84):
        self.schema = schema
        self._auto_enrich()
        self.linker = SchemaLinker(schema, fuzzy_threshold)

    # @classmethod
    # def from_engine(cls, engine: Engine, metadata_path: str | None = None,
    #                 categorical_max_distinct: int = 25, fuzzy_threshold: int = 84) -> "SemanticLayer":
    #     schema = introspect_schema(engine, metadata_path, categorical_max_distinct)
    #     return cls(schema, fuzzy_threshold)

    @classmethod
    def from_engine(cls, engine: Engine, metadata_path: str | None = None,
                    categorical_max_distinct: int = 25, fuzzy_threshold: int = 84,
                    metadata: dict | None = None) -> "SemanticLayer":
        schema = introspect_schema(engine, metadata_path, categorical_max_distinct, metadata)
        return cls(schema, fuzzy_threshold)

    # 
    def _auto_enrich(self) -> None:
        """Add synonyms derivable from identifiers alone (schema-agnostic)."""
        for t in self.schema.tables:
            for c in t.columns:
                parts = split_identifier(c.name)
                expanded = [_ABBREVIATIONS.get(p, p) for p in parts]
                syns = set(c.synonyms)
                if expanded != parts:
                    syns.add(" ".join(expanded))
                # "product_name" -> also "product" (the entity is usually referred to by its name)
                if len(parts) == 2 and parts[1] in ("name", "title", "label"):
                    syns.add(parts[0])
                # drop table-name prefix: products.product_price -> "price"
                tparts = split_identifier(t.name)
                if len(parts) > 1 and parts[0] in (tparts[0], tparts[0].rstrip("s")):
                    rest = " ".join(expanded[1:])
                    if rest not in ("id", "name"):
                        syns.add(rest)
                c.synonyms = sorted(s for s in syns if s and s != c.name.lower())

    
    def build_context(self, question: str) -> SemanticContext:
        analysis = self.linker.analyze(question)
        tables = [t.name for t in analysis.tables] or [t.name for t in self.schema.tables]
        # include tables reachable through a foreign key so the model can JOIN
        extra = []
        for t in self.schema.tables:
            for fk in t.foreign_keys:
                if fk.table in tables and fk.ref_table not in tables:
                    extra.append(fk.ref_table)
                if fk.ref_table in tables and fk.table not in tables:
                    extra.append(fk.table)
        tables = list(dict.fromkeys(tables + extra))
        return SemanticContext(
            question=question,
            analysis=analysis,
            tables=tables,
            schema_ddl=self.schema.to_ddl(tables),
            schema_notes=self.schema.describe(tables),
            hints=analysis.hints(),
        )
