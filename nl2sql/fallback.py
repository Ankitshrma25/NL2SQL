"""Deterministic, schema-aware template backend (last-resort fallback).

Used only after the SLM has exhausted its repair budget (or is not
available).  It composes a single-table SELECT from the semantic layer's
:class:`~nl2sql.nlp.QuestionAnalysis` using SQLGlot's expression builder,
so values are always emitted as properly quoted literals (no string
concatenation -> no injection) and only real schema identifiers are used.

Templates (chosen from detected intent):

=====================  ====================================================
intent                 SQL shape
=====================  ====================================================
group + aggregation    SELECT g, AGG(m) FROM t [WHERE] GROUP BY g [ORDER] [LIMIT]
aggregation            SELECT AGG(m) FROM t [WHERE]
superlative / TOP-N    SELECT * FROM t [WHERE] ORDER BY m DIR LIMIT n
distinct               SELECT DISTINCT c FROM t [WHERE]
plain lookup           SELECT * FROM t [WHERE] [ORDER BY sort column]
=====================  ====================================================

Its output still goes through the validator like any other candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from .nlp import Filter, QuestionAnalysis
from .schema import NUMERIC, ColumnInfo, TableInfo


@dataclass
class FallbackResult:
    sql: str | None
    template: str
    reason: str = ""


def _col(c: ColumnInfo) -> exp.Column:
    return exp.column(c.name, quoted=not c.name.isidentifier() or c.name.lower() in _RESERVED)


_RESERVED = {"order", "group", "select", "from", "where", "limit", "table", "index", "key", "date", "values"}


def _lit(v) -> exp.Expression:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return exp.Literal.number(v) if v >= 0 else exp.Neg(this=exp.Literal.number(-v))
    return exp.Literal.string(str(v))


def _filter_expr(f: Filter) -> exp.Expression:
    c = _col(f.column)
    if f.op == "BETWEEN":
        return exp.Between(this=c, low=_lit(f.value), high=_lit(f.value2))
    if f.op == "YEAR":
        return exp.EQ(this=exp.Anonymous(this="strftime", expressions=[exp.Literal.string("%Y"), c]),
                      expression=exp.Literal.string(str(f.value)))
    ops = {"=": exp.EQ, "!=": exp.NEQ, ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE}
    return ops[f.op](this=c, expression=_lit(f.value))


def _ordered(e: exp.Expression, direction: str | None) -> exp.Ordered:
    desc = (direction or "DESC") == "DESC"
    return exp.Ordered(this=e, desc=desc, nulls_first=not desc)  # SQLite's native NULL ordering


_AGG = {"SUM": exp.Sum, "AVG": exp.Avg, "MIN": exp.Min, "MAX": exp.Max, "COUNT": exp.Count}


class TemplateFallback:
    name = "template-fallback"

    def __init__(self, default_limit: int = 100):
        self.default_limit = default_limit

    def generate(self, a: QuestionAnalysis) -> FallbackResult:
        table = self._pick_table(a)
        if table is None:
            return FallbackResult(None, "none", "No table could be linked to the question.")
        own = lambda c: c is not None and c.table == table.name  # noqa: E731
        filters = [f for f in a.filters if own(f.column)]
        groups = [g for g in a.group_by if own(g)]
        measure = a.measure if own(a.measure) else None
        agg = a.aggregation

        # grouped superlative without explicit aggregation: "which city has the highest amount"
        if groups and not agg:
            agg = "SUM" if measure is not None else "COUNT"
        if agg in ("SUM", "AVG") and (measure is None or measure.logical_type != NUMERIC):
            measure = self._first_numeric(table)
            if measure is None:
                agg = "COUNT"

        q = exp.select().from_(exp.to_table(table.name))
        where = [_filter_expr(f) for f in filters]
        template = ""

        if agg:
            if agg == "COUNT" or measure is None:
                agg_expr = exp.Count(this=exp.Star())
                alias = "count"
                agg = "COUNT"
            else:
                agg_expr = _AGG[agg](this=_col(measure))
                alias = f"{agg.lower()}_{measure.name}"
            agg_alias = exp.alias_(agg_expr, alias)
            if groups:
                template = "group_aggregate"
                q = exp.select(*[_col(g) for g in groups], agg_alias).from_(exp.to_table(table.name))
                q = q.group_by(*[_col(g) for g in groups])
                if a.order_direction or a.top_n:
                    q = q.order_by(_ordered(exp.column(alias), a.order_direction))
                limit = a.top_n or (1 if a.superlative and not a.top_n and _is_single_answer(a) else None)
                if limit:
                    q = q.limit(limit)
                elif not a.order_direction:
                    q = q.order_by(*[_col(g) for g in groups])
            else:
                template = "aggregate"
                q = exp.select(agg_alias).from_(exp.to_table(table.name))
        elif a.order_direction or a.top_n:
            template = "top_n"
            sort_col = a.order_by if own(a.order_by) else (measure or self._first_numeric(table))
            q = q.select("*")
            if sort_col is not None:
                q = q.order_by(_ordered(_col(sort_col), a.order_direction))
            if a.top_n or a.superlative:
                q = q.limit(a.top_n or 1)
        elif a.wants_distinct and a.column_links:
            template = "distinct"
            target = a.distinct_column if own(a.distinct_column) else next(
                (l.column for l in a.column_links if own(l.column)), None)
            q = exp.select(_col(target)).distinct().from_(exp.to_table(table.name)) if target else q.select("*")
            if target is not None:
                q = q.order_by(_col(target))
        else:
            template = "lookup"
            q = q.select("*")  # all matching rows; the executor caps what is displayed

        for w in where:
            q = q.where(w)
        return FallbackResult(q.sql(dialect="sqlite"), template)

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_table(a: QuestionAnalysis) -> TableInfo | None:
        return a.primary_table

    @staticmethod
    def _first_numeric(t: TableInfo) -> ColumnInfo | None:
        for c in t.columns:
            if c.logical_type == NUMERIC and not c.is_identifier and not c.is_categorical:
                return c
        return None


def _is_single_answer(a: QuestionAnalysis) -> bool:
    q = a.normalized
    return q.startswith(("which", "what", "who")) or " the most" in q or " the highest" in q or " the lowest" in q