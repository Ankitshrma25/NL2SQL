"""Independent multi-factor SQL validator.

The SLM is never trusted.  Every candidate query goes through these
checks, in order, and the result carries *structured, specific* issues
that are fed back to the model in the repair prompt:

1. ``non_empty``      - something that looks like SQL was produced
2. ``syntax``         - parses with SQLGlot (SQLite dialect)
3. ``single_statement`` - exactly one statement (blocks ``SELECT 1; DROP ...``)
4. ``read_only``      - SELECT / set-operation only; no DML/DDL/PRAGMA/ATTACH
                         anywhere in the tree (including CTEs / subqueries)
5. ``tables``         - every referenced table exists in the schema
6. ``columns``        - every column exists, qualifiers point at real
                         aliases, unqualified names are not ambiguous,
                         double-quoted "strings" are caught
7. ``types``          - numeric vs text comparisons, SUM/AVG over
                         non-numeric columns, unknown categorical values
8. ``db_dry_run``     - ``EXPLAIN`` on the real database (optional), which
                         catches engine-specific errors without running
                         the query

Only a :class:`ValidatedQuery` (which only this module can create) is
accepted by the executor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import sqlglot
from rapidfuzz import process
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope

from .schema import DATE, NUMERIC, TEXT, ColumnInfo, SchemaMetadata

ERROR = "error"
WARNING = "warning"

_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge", "Command",
    "Pragma", "Attach", "Detach", "Transaction", "Commit", "Rollback", "Into", "Grant", "Revoke",
    "TruncateTable", "Use", "Set", "LoadData", "Copy", "Analyze",
]
_FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))
_FORBIDDEN_KEYWORDS = re.compile(
    r"^\s*(insert|update|delete|drop|create|alter|replace|truncate|attach|detach|pragma|vacuum|reindex|grant|revoke|begin|commit|rollback)\b",
    re.I,
)
_READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_SUM_LIKE = tuple(getattr(exp, n) for n in ("Sum", "Avg", "Stddev", "Variance") if hasattr(exp, n))
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


@dataclass
class ValidationIssue:
    code: str
    check: str
    message: str
    severity: str = ERROR
    identifier: str | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        s = f"[{self.code}] {self.message}"
        if self.suggestion:
            s += f" Suggestion: {self.suggestion}"
        return s

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ValidatedQuery:
    """Proof-of-validation token.  Only the validator can mint one."""

    __slots__ = ("sql", "_token")
    _MINT = object()

    def __init__(self, sql: str, _token: object):
        if _token is not ValidatedQuery._MINT:
            raise PermissionError("ValidatedQuery can only be created by SQLValidator")
        self.sql = sql
        self._token = _token

    def __repr__(self) -> str:
        return f"ValidatedQuery({self.sql!r})"


@dataclass
class ValidationResult:
    sql: str
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    normalized_sql: str | None = None
    query: ValidatedQuery | None = None

    @property
    def is_valid(self) -> bool:
        return self.query is not None and not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def error_codes(self) -> list[str]:
        return [i.code for i in self.errors]

    def feedback(self) -> str:
        """Precise, model-readable explanation of why the SQL failed."""
        if self.is_valid:
            return "SQL is valid."
        return "\n".join(f"- {e}" for e in self.errors)


class SQLValidator:
    def __init__(
        self,
        schema: SchemaMetadata,
        dry_run: Callable[[str], str | None] | None = None,
        dialect: str = "sqlite",
    ):
        """``dry_run(sql)`` returns ``None`` when the DB accepts the query, else an error string."""
        self.schema = schema
        self.dry_run = dry_run
        self.dialect = dialect


    def validate(self, sql: str | None) -> ValidationResult:
        sql = (sql or "").strip().rstrip(";").strip() if sql else ""
        res = ValidationResult(sql=sql)

        # 1. non-empty
        if not sql:
            res.issues.append(ValidationIssue("EMPTY_SQL", "non_empty", "No SQL query was produced."))
            return res
        res.checks_passed.append("non_empty")

        # Fast keyword guard (defence in depth; the AST check below is authoritative)
        if _FORBIDDEN_KEYWORDS.match(sql):
            kw = _FORBIDDEN_KEYWORDS.match(sql).group(1).upper()
            res.issues.append(ValidationIssue(
                "NOT_READ_ONLY", "read_only",
                f"{kw} statements are not allowed; only read-only SELECT queries may be generated.",
                identifier=kw, suggestion="Rewrite the answer as a single SELECT statement."))
            return res

        # 2. syntax
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except (ParseError, TokenError) as e:
            res.issues.append(ValidationIssue("SYNTAX_ERROR", "syntax", _parse_error_message(e),
                                              suggestion="Return one complete, syntactically valid SQLite SELECT statement."))
            return res
        if not statements:
            res.issues.append(ValidationIssue("SYNTAX_ERROR", "syntax", "Could not parse any SQL statement."))
            return res
        res.checks_passed.append("syntax")

        # 3. single statement
        if len(statements) > 1:
            kinds = ", ".join(type(s).__name__.upper() for s in statements)
            res.issues.append(ValidationIssue(
                "MULTIPLE_STATEMENTS", "single_statement",
                f"Expected exactly one statement but found {len(statements)} ({kinds}).",
                suggestion="Return a single SELECT statement with no semicolons."))
            # still report read-only violations for the extra statements
            for s in statements:
                bad = _forbidden(s)
                if bad:
                    res.issues.append(ValidationIssue(
                        "NOT_READ_ONLY", "read_only", f"Statement contains a forbidden {bad} operation.", identifier=bad))
            return res
        stmt = statements[0]
        res.checks_passed.append("single_statement")

        # 4. read-only
        bad = _forbidden(stmt)
        if bad or not isinstance(stmt, _READ_ROOTS):
            what = bad or type(stmt).__name__.upper()
            res.issues.append(ValidationIssue(
                "NOT_READ_ONLY", "read_only",
                f"Only read-only SELECT queries are allowed, but the query contains {what}.",
                identifier=what, suggestion="Rewrite the answer as a single SELECT statement."))
            return res
        res.checks_passed.append("read_only")

        # 5 + 6. tables / columns / identifiers
        try:
            scopes = traverse_scope(stmt)
        except Exception as e:  # pragma: no cover - sqlglot internal edge cases
            res.issues.append(ValidationIssue("SYNTAX_ERROR", "syntax", f"Could not analyse query structure: {e}"))
            return res
        resolved: dict[int, ColumnInfo] = {}
        n_before = len(res.errors)
        self._check_tables(stmt, scopes, res)
        if len(res.errors) == n_before:
            res.checks_passed.append("tables")
        n_before = len(res.errors)
        self._check_columns(scopes, res, resolved)
        if len(res.errors) == n_before:
            res.checks_passed.append("columns")
        if res.errors:
            return res

        # 7. types
        n_before = len(res.errors)
        self._check_types(stmt, res, resolved)
        self._check_grouping(stmt, res)
        if len(res.errors) == n_before:
            res.checks_passed.append("types")
        if res.errors:
            return res

        normalized = stmt.sql(dialect=self.dialect)
        res.normalized_sql = normalized

        # 8. database dry run
        if self.dry_run is not None:
            err = self.dry_run(sql)
            if err:
                res.issues.append(ValidationIssue("DB_ERROR", "db_dry_run",
                                                  f"The database rejected the query: {err}"))
                return res
            res.checks_passed.append("db_dry_run")

        res.query = ValidatedQuery(sql, ValidatedQuery._MINT)
        return res

  
    def _check_tables(self, stmt: exp.Expression, scopes: list[Scope], res: ValidationResult) -> None:
        cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
        known = [t.name for t in self.schema.tables]
        seen = set()
        for tbl in stmt.find_all(exp.Table):
            name = tbl.name
            if not name or name.lower() in cte_names or name.lower() in seen:
                continue
            seen.add(name.lower())
            if tbl.args.get("db") or tbl.args.get("catalog"):
                res.issues.append(ValidationIssue(
                    "UNKNOWN_TABLE", "tables", f"Schema-qualified table '{tbl.sql()}' is not allowed.",
                    identifier=tbl.sql(), suggestion=f"Use one of: {', '.join(known)}."))
                continue
            if not self.schema.table(name):
                res.issues.append(ValidationIssue(
                    "UNKNOWN_TABLE", "tables", f"Table '{name}' does not exist.",
                    identifier=name, suggestion=_did_you_mean(name, known, f"Available tables: {', '.join(known)}.")))

    def _source_columns(self, source) -> tuple[set[str] | None, str]:
        """Column names exposed by a scope source. None = unknown (e.g. SELECT *)."""
        if isinstance(source, exp.Table):
            t = self.schema.table(source.name)
            return ({c.name.lower() for c in t.columns} if t else None), (t.name if t else source.name)
        if isinstance(source, Scope):
            names = source.expression.named_selects if hasattr(source.expression, "named_selects") else []
            if not names or "*" in names:
                return None, "subquery"
            return {n.lower() for n in names}, "subquery"
        return None, "?"

    def _check_columns(self, scopes: list[Scope], res: ValidationResult, resolved: dict[int, ColumnInfo]) -> None:
        reported: set[tuple[str, str]] = set()

        def report(code, msg, ident, sugg=None):
            key = (code, ident)
            if key not in reported:
                reported.add(key)
                res.issues.append(ValidationIssue(code, "columns", msg, identifier=ident, suggestion=sugg))

        for scope in scopes:
            projections = scope.expression.expressions if isinstance(scope.expression, exp.Select) else []
            select_aliases = {p.alias.lower() for p in projections if isinstance(p, exp.Alias) and p.alias}
            for col in scope.columns:
                if isinstance(col.this, exp.Star):
                    continue
                name = col.name
                qual = col.table
                quoted = bool(col.this.args.get("quoted")) if isinstance(col.this, exp.Identifier) else False
                if qual:
                    src, owner = self._find_source(scope, qual)
                    if src is None:
                        visible = sorted(self._visible_aliases(scope))
                        report("UNKNOWN_TABLE_ALIAS", f"Column '{qual}.{name}' uses qualifier '{qual}', which is not a table or alias in the FROM clause.",
                               f"{qual}.{name}", f"Visible tables/aliases: {', '.join(visible) or 'none'}.")
                        continue
                    cols, tname = self._source_columns(src)
                    if cols is None:
                        continue
                    if name.lower() not in cols:
                        report("UNKNOWN_COLUMN", f"Column '{name}' does not exist in table '{tname}'.",
                               f"{qual}.{name}", _did_you_mean(name, sorted(cols), f"Columns of {tname}: {', '.join(sorted(cols))}."))
                        continue
                    if isinstance(src, exp.Table):
                        ci = self.schema.table(src.name).column(name)
                        if ci:
                            resolved[id(col)] = ci
                    continue

                # unqualified
                matches = []
                unknown_source = False
                s = scope
                while s is not None and not matches:
                    for alias, src in s.sources.items():
                        cols, tname = self._source_columns(src)
                        if cols is None:
                            unknown_source = True
                        elif name.lower() in cols:
                            matches.append((alias, src, tname))
                    s = s.parent
                if not matches:
                    if unknown_source or (name.lower() in select_aliases and not _in_projection(col, scope.expression)):
                        continue
                    if quoted:
                        report("UNKNOWN_COLUMN", f"\"{name}\" is treated as a column identifier because it is in double quotes, but no such column exists.",
                               name, f"Use single quotes for string literals: '{name}'.")
                        continue
                    visible = self._visible_columns(scope)
                    report("UNKNOWN_COLUMN", f"Column '{name}' does not exist in the referenced table(s).",
                           name, _did_you_mean(name, visible, f"Available columns: {', '.join(visible)}."))
                    continue
                own = [m for m in matches if m[0] in scope.sources]
                if len(own) > 1:
                    report("AMBIGUOUS_COLUMN", f"Column '{name}' is ambiguous; it exists in {', '.join(m[2] for m in own)}.",
                           name, f"Qualify it, e.g. {own[0][0]}.{name}.")
                    continue
                alias, src, _ = matches[0]
                if isinstance(src, exp.Table):
                    ci = self.schema.table(src.name).column(name) if self.schema.table(src.name) else None
                    if ci:
                        resolved[id(col)] = ci

    @staticmethod
    def _find_source(scope: Scope, qual: str):
        s = scope
        while s is not None:
            for alias, src in s.sources.items():
                if alias.lower() == qual.lower():
                    return src, alias
            s = s.parent
        return None, None

    @staticmethod
    def _visible_aliases(scope: Scope) -> set[str]:
        out = set()
        s = scope
        while s is not None:
            out.update(s.sources.keys())
            s = s.parent
        return out

    def _visible_columns(self, scope: Scope) -> list[str]:
        out: list[str] = []
        for src in scope.sources.values():
            cols, _ = self._source_columns(src)
            if cols:
                out.extend(sorted(cols))
        if not out:
            out = [c.name for c in self.schema.all_columns()]
        return list(dict.fromkeys(out))

    
    def _check_types(self, stmt: exp.Expression, res: ValidationResult, resolved: dict[int, ColumnInfo]) -> None:
        def colinfo(node) -> ColumnInfo | None:
            return resolved.get(id(node)) if isinstance(node, exp.Column) else None

        # comparisons
        for cmp in stmt.find_all(*_COMPARISONS):
            left, right = cmp.left, cmp.right
            for c_node, v_node in ((left, right), (right, left)):
                ci = colinfo(c_node)
                if ci is None:
                    continue
                self._compare(ci, cmp, v_node, res)
        for node in stmt.find_all(exp.In):
            ci = colinfo(node.this)
            if ci:
                for v in node.expressions:
                    self._compare(ci, node, v, res)
        for node in stmt.find_all(exp.Between):
            ci = colinfo(node.this)
            if ci:
                for v in (node.args.get("low"), node.args.get("high")):
                    self._compare(ci, node, v, res)
        for node in stmt.find_all(exp.Like):
            ci = colinfo(node.this)
            if ci and ci.logical_type == NUMERIC:
                res.issues.append(ValidationIssue(
                    "TYPE_MISMATCH", "types", f"LIKE is a text operation but '{ci.name}' is numeric.",
                    severity=WARNING, identifier=ci.name, suggestion="Use a numeric comparison instead."))

        # SUM / AVG on non-numeric columns
        if _SUM_LIKE:
            for agg in stmt.find_all(*_SUM_LIKE):
                ci = colinfo(agg.this)
                if ci is None:
                    continue
                fn = type(agg).__name__.upper()
                if ci.logical_type != NUMERIC or ci.is_categorical:
                    kind = "categorical" if ci.is_categorical else ci.logical_type
                    res.issues.append(ValidationIssue(
                        "INVALID_AGGREGATION", "types",
                        f"{fn}({ci.name}) is invalid: '{ci.name}' is a {kind} column, not a numeric measure.",
                        identifier=ci.name,
                        suggestion=f"Use COUNT({ci.name}) / COUNT(DISTINCT {ci.name}), GROUP BY {ci.name}, or {fn} a numeric column."))
                elif ci.is_identifier:
                    res.issues.append(ValidationIssue(
                        "INVALID_AGGREGATION", "types",
                        f"{fn}({ci.name}) aggregates an identifier column, which is almost never meaningful.",
                        severity=WARNING, identifier=ci.name, suggestion=f"Did you mean COUNT({ci.name})?"))

    def _compare(self, ci: ColumnInfo, op_node, v_node, res: ValidationResult) -> None:
        if not isinstance(v_node, exp.Literal):
            if isinstance(v_node, exp.Neg) and isinstance(v_node.this, exp.Literal):
                v_node = v_node.this
            else:
                return
        is_str = v_node.is_string
        val = v_node.this
        op = type(op_node).__name__.upper()
        if ci.logical_type == NUMERIC and is_str and not _looks_numeric(val):
            res.issues.append(ValidationIssue(
                "TYPE_MISMATCH", "types",
                f"Numeric column '{ci.name}' is compared ({op}) with the text value '{val}'.",
                identifier=ci.name,
                suggestion=f"Compare '{ci.name}' with a number (range {ci.min_value}..{ci.max_value}), or use the text column that holds '{val}'."))
        elif ci.logical_type == DATE and not is_str:
            res.issues.append(ValidationIssue(
                "TYPE_MISMATCH", "types",
                f"Date column '{ci.name}' is compared ({op}) with the number {val}; dates are stored as 'YYYY-MM-DD' text.",
                identifier=ci.name,
                suggestion=f"Use a quoted date like '{ci.min_value}' or strftime('%Y', {ci.name}) = '{val}'."))
        elif ci.logical_type == TEXT and not is_str and op in ("GT", "GTE", "LT", "LTE"):
            res.issues.append(ValidationIssue(
                "TYPE_MISMATCH", "types",
                f"Text column '{ci.name}' is compared ({op}) with the number {val}.",
                identifier=ci.name, suggestion="Compare text columns with quoted text values, or use a numeric column."))
        elif ci.is_categorical and is_str and ci.sample_values and op in ("EQ", "NEQ", "IN"):
            domain = [str(v) for v in ci.sample_values]
            complete = ci.distinct_count is not None and ci.distinct_count <= len(domain)
            if complete and val not in domain:
                close = [d for d in domain if d.lower() == val.lower()]
                sugg = _did_you_mean(val, domain, f"Valid values: {', '.join(repr(d) for d in domain)}.")
                res.issues.append(ValidationIssue(
                    "UNKNOWN_VALUE", "types",
                    f"'{val}' is not a value of categorical column '{ci.name}' (comparison is case-sensitive).",
                    severity=ERROR if close else WARNING, identifier=ci.name,
                    suggestion=f"Use '{close[0]}'." if close else sugg))

    def _check_grouping(self, stmt: exp.Expression, res: ValidationResult) -> None:
        for sel in stmt.find_all(exp.Select):
            group = sel.args.get("group")
            if not group:
                continue
            keys = {g.sql().lower() for g in group.expressions}
            key_names = {g.name.lower() for g in group.expressions if isinstance(g, exp.Column)}
            for proj in sel.expressions:
                inner = proj.this if isinstance(proj, exp.Alias) else proj
                if inner.find(exp.AggFunc) or isinstance(inner, exp.Literal):
                    continue
                if inner.sql().lower() in keys or proj.alias_or_name.lower() in keys:
                    continue
                if isinstance(inner, exp.Column) and inner.name.lower() in key_names:
                    continue
                res.issues.append(ValidationIssue(
                    "UNGROUPED_COLUMN", "types",
                    f"'{inner.sql()}' is selected but is neither aggregated nor in GROUP BY; its value would be arbitrary.",
                    severity=WARNING, identifier=inner.sql(),
                    suggestion=f"Add {inner.sql()} to GROUP BY or wrap it in an aggregate."))



def _forbidden(stmt: exp.Expression) -> str | None:
    if isinstance(stmt, _FORBIDDEN_NODES):
        return type(stmt).__name__.upper()
    for node in stmt.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, _FORBIDDEN_NODES):
            return type(n).__name__.upper()
    return None


def _in_projection(node: exp.Expression, select: exp.Expression) -> bool:
    """True when ``node`` sits inside the SELECT list (aliases are not visible there)."""
    projections = select.expressions if isinstance(select, exp.Select) else []
    cur = node
    while cur is not None and cur.parent is not select:
        cur = cur.parent
    return cur is not None and any(cur is p for p in projections)


def _looks_numeric(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _did_you_mean(name: str, candidates: list[str], otherwise: str) -> str:
    if not candidates:
        return otherwise
    best = process.extractOne(name.lower(), [c.lower() for c in candidates])
    if best and best[1] >= 60:
        return f"Did you mean '{candidates[best[2]]}'? {otherwise}"
    return otherwise


def _parse_error_message(e: Exception) -> str:
    errs = getattr(e, "errors", None) or []
    if errs:
        first = errs[0]
        desc = first.get("description") or "syntax error"
        line, col = first.get("line"), first.get("col")
        near = (first.get("highlight") or "").strip()
        msg = f"Syntax error: {desc}"
        if line is not None:
            msg += f" (line {line}, column {col}"
            msg += f", near '{near}')" if near else ")"
        return msg
    return "Syntax error: " + re.sub(r"\x1b\[[0-9;]*m", "", str(e)).splitlines()[0]
