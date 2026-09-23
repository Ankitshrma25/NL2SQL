"""NLP layer: schema linking + intent understanding.

Pure, deterministic, schema-agnostic analysis of a natural-language
question against :class:`~nl2sql.schema.SchemaMetadata`:

* **Schema linking** - which tables / columns the question mentions
  (exact, synonym and fuzzy matches via RapidFuzz over word n-grams).
* **Value linking** - which categorical values appear in the question
  (e.g. "Mumbai" -> ``city = 'Mumbai'``).
* **Intent** - aggregation (COUNT/SUM/AVG/MIN/MAX), GROUP BY dimensions,
  numeric / date filters, sort direction and TOP-N.

The output (:class:`QuestionAnalysis`) is used in two places: it becomes
*hints* in the SLM prompt, and it drives the deterministic fallback.
Nothing here references a concrete table or column name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .schema import DATE, NUMERIC, ColumnInfo, SchemaMetadata, TableInfo

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "by", "per", "and", "or", "is", "are", "was",
    "were", "what", "which", "who", "how", "many", "much", "show", "me", "list", "give", "get",
    "find", "all", "with", "each", "every", "from", "that", "have", "has", "had", "do", "does",
    "did", "than", "top", "bottom", "their", "its", "it", "be", "as", "at", "there", "total",
    "average", "number", "count", "sum", "highest", "lowest", "most", "least", "please", "tell",
    "whose", "where", "between", "greater", "less", "more", "over", "under", "above", "below",
}

AGG_PATTERNS: list[tuple[str, str]] = [
    ("COUNT", r"\b(how many|count of|count|number of|no\. of|num of)\b"),
    ("AVG", r"\b(average|avg|mean)\b"),
    ("SUM", r"\b(total|sum of|sum|overall|combined)\b"),
    ("MAX", r"\b(max|maximum)\b"),
    ("MIN", r"\b(min|minimum)\b"),
]
DESC_WORDS = r"(top|highest|largest|biggest|greatest|most|maximum|best|max)"
ASC_WORDS = r"(bottom|lowest|smallest|least|fewest|minimum|worst|cheapest|min)"
GROUP_TRIGGERS = r"\b(?:grouped by|group by|broken down by|split by|for each|for every|by|per|each|across)\s+"

NUM = r"(-?\d+(?:\.\d+)?)"
COMPARISONS: list[tuple[str, str]] = [
    (r"(?:greater than or equal to|at least|no less than|>=)", ">="),
    (r"(?:less than or equal to|at most|no more than|<=)", "<="),
    (r"(?:greater than|more than|over|above|exceeding|exceeds|higher than|>)", ">"),
    (r"(?:less than|fewer than|under|below|lower than|<)", "<"),
    (r"(?:equal to|equals|exactly|=)", "="),
]
NEGATION = r"\b(?:not|excluding|except|other than|without|apart from)\s+(?:in\s+|from\s+|equal to\s+|for\s+|paid (?:with|via|by)\s+)?$"


@dataclass
class ColumnLink:
    column: ColumnInfo
    score: float
    phrase: str
    start: int  # character offset in normalised question

    def __repr__(self) -> str:  # compact for logs
        return f"{self.column.table}.{self.column.name}~{self.score:.0f}"


@dataclass
class Filter:
    column: ColumnInfo
    op: str  # = != > >= < <= BETWEEN YEAR
    value: object
    value2: object = None
    source: str = ""

    def describe(self) -> str:
        if self.op == "BETWEEN":
            return f"{self.column.name} BETWEEN {self.value!r} AND {self.value2!r}"
        if self.op == "YEAR":
            return f"year({self.column.name}) = {self.value!r}"
        return f"{self.column.name} {self.op} {self.value!r}"


@dataclass
class QuestionAnalysis:
    question: str
    normalized: str
    tables: list[TableInfo] = field(default_factory=list)
    column_links: list[ColumnLink] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    aggregation: str | None = None
    measure: ColumnInfo | None = None
    group_by: list[ColumnInfo] = field(default_factory=list)
    order_direction: str | None = None  # "DESC" | "ASC"
    top_n: int | None = None
    superlative: bool = False
    wants_distinct: bool = False
    order_by: ColumnInfo | None = None  # explicit "sorted by <column>"
    distinct_column: ColumnInfo | None = None  # "list all payment modes" -> DISTINCT payment_mode

    @property
    def primary_table(self) -> TableInfo | None:
        return self.tables[0] if self.tables else None

    def hints(self) -> list[str]:
        """Human-readable hints injected into the SLM prompt."""
        h: list[str] = []
        if self.tables:
            h.append("Relevant table(s): " + ", ".join(t.name for t in self.tables))
        cols = []
        for l in self.column_links:
            name = f"{l.column.table}.{l.column.name}"
            if name not in cols:
                cols.append(name)
        if cols:
            h.append("Columns mentioned: " + ", ".join(cols))
        if self.aggregation:
            target = "*" if self.aggregation == "COUNT" and not self.measure else (self.measure.name if self.measure else "?")
            h.append(f"Aggregation: {self.aggregation}({target})")
        if self.group_by:
            h.append("Group by: " + ", ".join(c.name for c in self.group_by))
        for f in self.filters:
            h.append("Filter: " + f.describe())
        if self.order_by is not None:
            h.append(f"Sort by: {self.order_by.name} {self.order_direction or 'ASC'}")
        elif self.order_direction and self.measure is not None and not self.aggregation and not self.group_by:
            h.append(f"Sort by: {self.measure.name} {self.order_direction}")
        elif self.order_direction:
            h.append(f"Sort: {self.order_direction}")
        if self.top_n:
            h.append(f"Limit: {self.top_n}")
        if self.wants_distinct:
            target = f" of {self.distinct_column.name}" if self.distinct_column is not None else ""
            h.append(f"Return the DISTINCT values{target}, not whole rows")
        if not self.aggregation and not self.group_by and not self.wants_distinct:
            h.append("Return complete rows: SELECT * (the question asks for records, not specific columns)")
        return h


# ----------------------------------------------------------------------
def normalize(question: str) -> str:
    q = question.strip().lower()
    q = re.sub(r"[?!,;]", " ", q)
    q = re.sub(r"(?<=\d),(?=\d{3})", "", q)  # 1,000 -> 1000
    q = q.replace("$", " ").replace("₹", " ")
    return re.sub(r"\s+", " ", q).strip()


def _ngrams(text_: str, max_n: int = 4) -> list[tuple[str, int]]:
    """(ngram, char_start) for all word n-grams."""
    words = [(m.group(0).rstrip("."), m.start()) for m in re.finditer(r"[a-z0-9_.'\-]+", text_)]
    out = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            chunk = words[i : i + n]
            phrase = " ".join(w for w, _ in chunk)
            if n == 1 and phrase in STOPWORDS:
                continue
            if all(w in STOPWORDS for w, _ in chunk):
                continue
            out.append((phrase, chunk[0][1]))
    return out


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _sing_phrase(p: str) -> str:
    return " ".join(_singular(w) for w in p.split())


FILLER = {"the", "a", "an", "of"}


def _strip_filler(p: str) -> str:
    words = [w for w in p.split() if w not in FILLER]
    return " ".join(words) if words else p


def _phrase_score(ngram: str, phrase: str) -> float:
    # "price of the item" should match price_of_item
    # underscores count as word breaks, so "order_id" is two words and never fuzzy-matches "orders"
    a = _sing_phrase(_strip_filler(ngram.replace("_", " ")))
    b = _sing_phrase(_strip_filler(phrase.replace("_", " ")))
    if a == b:
        return 100.0
    if min(len(a), len(b)) < 4:  # avoid fuzzy noise on very short tokens
        return 0.0
    if len(a.split()) != len(b.split()):  # fuzzy only between phrases of equal word length
        return 0.0
    if any(w in STOPWORDS for w in a.split()):  # "transaction is" must not fuzzy-match "transaction id"
        return 0.0
    # multi-word: every word must be close, so "orders done" never matches "order id"
    if any(fuzz.ratio(x, y) < 75 for x, y in zip(a.split(), b.split())):
        return 0.0
    return float(fuzz.ratio(a, b))


def _stem(word: str) -> str:
    """Tiny suffix stripper: credited/crediting/credits -> credit, refunded -> refund."""
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


class SchemaLinker:
    """Links question text to schema elements and extracts intent."""

    def __init__(self, schema: SchemaMetadata, fuzzy_threshold: int = 84):
        self.schema = schema
        self.threshold = fuzzy_threshold

    # ---------------- linking ----------------
    def link_columns(self, q: str) -> list[ColumnLink]:
        grams = _ngrams(q)
        best: dict[tuple[str, str], ColumnLink] = {}
        for col in self.schema.all_columns():
            for phrase in col.phrases:
                for gram, pos in grams:
                    s = _phrase_score(gram, phrase)
                    if s < self.threshold:
                        continue
                    # prefer longer, more specific matches on ties
                    s_adj = s + min(len(phrase.split()), 3) * 0.5
                    key = (col.table, col.name)
                    cur = best.get(key)
                    if cur is None or s_adj > cur.score or (s_adj == cur.score and pos < cur.start):
                        best[key] = ColumnLink(col, s_adj, gram, pos)
        links = sorted(best.values(), key=lambda l: (l.start, -l.score))
        return self._drop_subsumed(links)

    @staticmethod
    def _drop_subsumed(links: list[ColumnLink]) -> list[ColumnLink]:
        """If "product name" matched product_name, drop a weaker 'name' match on the same span."""
        keep: list[ColumnLink] = []
        for l in links:
            span = (l.start, l.start + len(l.phrase))
            dominated = False
            for o in links:
                if o is l:
                    continue
                o_span = (o.start, o.start + len(o.phrase))
                inside = o_span[0] <= span[0] and span[1] <= o_span[1] and (o_span != span)
                if inside and o.score >= l.score - 5:
                    dominated = True
                    break
                if o_span == span and o.score > l.score:
                    dominated = True
                    break
            if not dominated:
                keep.append(l)
        return keep

    def link_tables(self, q: str, col_links: list[ColumnLink]) -> list[TableInfo]:
        scores: dict[str, float] = {}
        grams = _ngrams(q)
        for t in self.schema.tables:
            for phrase in t.phrases:
                for gram, _ in grams:
                    s = _phrase_score(gram, phrase)
                    if s >= self.threshold:
                        scores[t.name] = max(scores.get(t.name, 0), s + 50)
        for l in col_links:
            scores[l.column.table] = scores.get(l.column.table, 0) + l.score / 10
        if not scores and len(self.schema.tables) == 1:
            scores[self.schema.tables[0].name] = 1
        ordered = sorted(scores, key=lambda n: -scores[n])
        return [self.schema.table(n) for n in ordered]

    def link_values(self, q: str, tables: list[TableInfo]) -> list[Filter]:
        filters: list[Filter] = []
        cand_cols = [c for t in tables for c in t.columns if c.is_categorical] or [
            c for c in self.schema.all_columns() if c.is_categorical
        ]
        taken_spans: list[tuple[int, int]] = []
        # longest values first so "New Delhi" beats "Delhi"
        pairs = sorted(
            ((c, v) for c in cand_cols for v in c.sample_values if isinstance(v, str) and v.strip()),
            key=lambda cv: -len(cv[1]),
        )
        for col, val in pairs:
            v = val.lower()
            m = re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", q)
            if not m and " " not in v and len(v) >= 4:
                # inflected form of a single-word value: "credited" -> 'Credit', "refunds" -> 'Refunded'
                for w in re.finditer(r"[a-z0-9]+", q):
                    if _stem(w.group(0)) == _stem(v) and len(w.group(0)) >= 4:
                        m = w
                        break
            if not m and len(v) >= 5:
                # fuzzy: compare against n-grams of the same word length (handles typos)
                n = len(v.split())
                for gram, pos in _ngrams(q, max_n=n):
                    if len(gram.split()) == n and fuzz.ratio(gram, v) >= 90:
                        m = re.search(re.escape(gram), q[pos:])
                        start = pos
                        break
                else:
                    continue
                span = (start, start + len(gram))
            elif not m:
                continue
            else:
                span = m.span()
            if any(a < span[1] and span[0] < b for a, b in taken_spans):
                continue
            taken_spans.append(span)
            op = "!=" if re.search(NEGATION, q[: span[0]]) else "="
            filters.append(Filter(col, op, val, source=q[span[0] : span[1]]))
        return filters

    # ---------------- intent ----------------
    def analyze(self, question: str) -> QuestionAnalysis:
        q = normalize(question)
        links = self.link_columns(q)
        tables = self.link_tables(q, links)
        # keep only columns belonging to linked tables (when any)
        tnames = {t.name for t in tables}
        links = [l for l in links if not tnames or l.column.table in tnames]
        a = QuestionAnalysis(question=question, normalized=q, tables=tables, column_links=links)

        value_filters = self.link_values(q, tables)
        value_spans = [f.source for f in value_filters]
        # columns whose only "mention" is actually a categorical value are not column links
        a.column_links = [l for l in links if not any(l.phrase in vs for vs in value_spans)]
        a.filters.extend(value_filters)

        self._aggregation(a)
        self._numeric_filters(a)
        self._date_filters(a)
        self._group_by(a)
        self._ordering(a)
        self._measure(a)
        a.wants_distinct = bool(re.search(r"\b(distinct|unique|different)\b", q))
        self._distinct_values(a)
        return a

    def _aggregation(self, a: QuestionAnalysis) -> None:
        for agg, pat in AGG_PATTERNS:
            if re.search(pat, a.normalized):
                a.aggregation = agg
                return

    def _col_near(self, a: QuestionAnalysis, pos: int, numeric_only=True, before=True) -> ColumnInfo | None:
        cands = [
            l for l in a.column_links
            if (not numeric_only or l.column.logical_type == NUMERIC)
            and (l.start < pos if before else l.start >= pos)
        ]
        if not cands:
            return None
        cands.sort(key=lambda l: abs(pos - l.start))
        return cands[0].column

    def _numeric_filters(self, a: QuestionAnalysis) -> None:
        q = a.normalized
        for m in re.finditer(rf"\bbetween\s+{NUM}\s+and\s+{NUM}", q):
            col = self._col_near(a, m.start()) or self._default_measure(a)
            if col:
                lo, hi = sorted((float(m.group(1)), float(m.group(2))))
                a.filters.append(Filter(col, "BETWEEN", _num(lo), _num(hi), source=m.group(0)))
        for pat, op in COMPARISONS:
            for m in re.finditer(rf"(?<![a-z]){pat}\s+{NUM}(?!\s*(?:rows|results|records))", q):
                if any(f.source and m.start() >= q.find(f.source) and m.start() < q.find(f.source) + len(f.source) for f in a.filters if f.op == "BETWEEN"):
                    continue
                col = self._col_near(a, m.start()) or self._default_measure(a)
                if col is None:
                    continue
                val = _num(float(m.group(1)))
                if col.logical_type == DATE:
                    continue
                if not any(f.column is col and f.op == op and f.value == val for f in a.filters):
                    a.filters.append(Filter(col, op, val, source=m.group(0)))

    def _date_filters(self, a: QuestionAnalysis) -> None:
        date_cols = [l.column for l in a.column_links if l.column.logical_type == DATE]
        if not date_cols and a.primary_table:
            date_cols = [c for c in a.primary_table.columns if c.logical_type == DATE]
        if not date_cols:
            return
        for m in re.finditer(r"\b(?:in|during|for|of|year)\s+((?:19|20)\d{2})\b", a.normalized):
            a.filters.append(Filter(date_cols[0], "YEAR", m.group(1), source=m.group(0)))

    def _group_by(self, a: QuestionAnalysis) -> None:
        q = a.normalized
        filter_cols = {id(f.column) for f in a.filters}
        found: list[ColumnInfo] = []
        for m in re.finditer(GROUP_TRIGGERS, q):
            if re.search(r"\b(sort|sorted|order|ordered|rank|ranked)\s+$", q[: m.start()] + " "):
                continue
            tail_start = m.end()
            for l in a.column_links:
                in_window = tail_start <= l.start <= tail_start + 12
                groupable = (l.column.logical_type != NUMERIC or l.column.is_categorical) and not l.column.is_identifier
                if in_window and groupable:
                    if l.column not in found:
                        found.append(l.column)
                    break
        # "which city has the highest ..." / "what category ..." -> group by that dimension
        m = re.match(r"^(?:which|what)\s+", q)
        if m and (re.search(DESC_WORDS, q) or re.search(ASC_WORDS, q) or a.aggregation):
            for l in a.column_links:
                if l.start <= m.end() + 2 and l.column.logical_type != NUMERIC and not l.column.is_identifier:
                    if l.column not in found:
                        found.append(l.column)
                    break
        # "top 5 customers by total amount" -> dimension named right after TOP-N
        m = re.search(rf"\b{DESC_WORDS}\s+\d+\s+|\b{ASC_WORDS}\s+\d+\s+", q)
        if m and a.aggregation in ("SUM", "AVG", "COUNT"):
            for l in a.column_links:
                if m.end() <= l.start <= m.end() + 2 and l.column.logical_type != NUMERIC:
                    if l.column not in found:
                        found.insert(0, l.column)
                    break
        # a column used as an equality filter value is not also a grouping key
        a.group_by = [c for c in found if id(c) not in filter_cols or a.aggregation]

    def _ordering(self, a: QuestionAnalysis) -> None:
        q = a.normalized
        m = re.search(rf"\b(?:{DESC_WORDS[1:-1]}|{ASC_WORDS[1:-1]})\s+(\d+)\b", q) or re.search(
            rf"\b(\d+)\s+(?:{DESC_WORDS[1:-1]}|{ASC_WORDS[1:-1]})\b", q
        )
        if m:
            a.top_n = int(m.group(1))
        else:
            m2 = re.search(r"\b(?:first|limit)\s+(\d+)\b", q)
            if m2:
                a.top_n = int(m2.group(1))
        desc = re.search(rf"\b{DESC_WORDS}\b", q)
        asc = re.search(rf"\b{ASC_WORDS}\b", q)
        # MAX / MIN keywords that are pure aggregations are not orderings
        if a.aggregation in ("MAX", "MIN") and not a.group_by and not a.top_n:
            return
        if desc and (not asc or desc.start() < asc.start()):
            a.order_direction, a.superlative = "DESC", True
        elif asc:
            a.order_direction, a.superlative = "ASC", True
        m = re.search(r"\b(?:sort|sorted|order|ordered|arrange|arranged|rank|ranked)\s+by\s+", q)
        if m:
            for l in a.column_links:
                if m.end() <= l.start <= m.end() + 12:
                    a.order_by = l.column
                    break
            if a.order_by is not None and not a.superlative:
                a.order_direction = "ASC"
        if re.search(r"\b(ascending|alphabetical|oldest first|earliest)\b", q):
            a.order_direction = "ASC"
        elif re.search(r"\b(descending|newest first|latest|most recent)\b", q):
            a.order_direction = "DESC"

    def _distinct_values(self, a: QuestionAnalysis) -> None:
        """'all the order statuses' / 'list the payment modes' asks for a column's values, not whole rows.

        Applies when a categorical column is named, nothing is aggregated, grouped, ranked or sorted, and the
        column is not already pinned by an equality filter ('all UPI orders' stays a row lookup).
        """
        filtered = {id(f.column) for f in a.filters}
        cands = [l.column for l in a.column_links if l.column.is_categorical and id(l.column) not in filtered]
        if a.wants_distinct:
            a.distinct_column = cands[0] if cands else (a.column_links[0].column if a.column_links else None)
            return
        if (cands and not a.aggregation and not a.group_by and not a.top_n and not a.superlative
                and a.order_by is None):
            a.wants_distinct, a.distinct_column = True, cands[0]

    def _default_measure(self, a: QuestionAnalysis) -> ColumnInfo | None:
        t = a.primary_table
        if not t:
            return None
        nums = [c for c in t.columns if c.logical_type == NUMERIC and not c.is_identifier]
        return nums[0] if nums else None

    def _measure(self, a: QuestionAnalysis) -> None:
        filter_only = {id(f.column) for f in a.filters}
        numeric_links = [
            l for l in a.column_links if l.column.logical_type == NUMERIC and not l.column.is_identifier
        ]
        # prefer a numeric column not used purely as a filter
        preferred = [l for l in numeric_links if id(l.column) not in filter_only] or numeric_links
        if a.aggregation == "COUNT":
            a.measure = None  # COUNT(*)
            return
        if a.aggregation in ("MAX", "MIN"):
            any_link = [l for l in a.column_links if not l.column.is_identifier and id(l.column) not in filter_only]
            if preferred:
                a.measure = preferred[0].column
            elif any_link:
                a.measure = any_link[0].column
            else:
                a.measure = self._default_measure(a)
            return
        if preferred:
            a.measure = preferred[0].column
        elif a.aggregation in ("SUM", "AVG") or a.superlative or a.group_by:
            a.measure = self._default_measure(a)


def _num(x: float):
    return int(x) if float(x).is_integer() else x