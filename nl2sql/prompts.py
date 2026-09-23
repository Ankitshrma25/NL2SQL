"""Prompt construction for generation and self-repair.

Prompts are plain data (:class:`Prompt`) so any backend - chat model,
seq2seq text-to-SQL model or a test double - can render them the way it
needs.  This keeps model-specific formatting out of the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .semantic_layer import SemanticContext

SYSTEM_INSTRUCTIONS = (
    "You are an expert SQLite analyst. Translate the user's question into ONE read-only "
    "SQLite SELECT statement.\n"
    "Rules:\n"
    "- Use ONLY the tables and columns listed in the schema; never invent names.\n"
    "- Use single quotes for string literals and match categorical values exactly as listed.\n"
    "- Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA or multiple statements.\n"
    "- Only SUM/AVG numeric columns. Use COUNT for categorical columns.\n"
    "- When the question asks to show/list/find records, return all columns with SELECT * "
    "unless it names specific columns.\n"
    "- Only add filters the question asks for.\n"
    "- Dates are 'YYYY-MM-DD' text; filter years with strftime('%Y', col) = 'YYYY'.\n"
    "- Return only the SQL, no explanation."
)


@dataclass
class Prompt:
    kind: str  # "generate" | "repair"
    system: str
    user: str
    attempt: int = 0
    meta: dict = field(default_factory=dict)

    def as_text(self) -> str:
        return f"{self.system}\n\n{self.user}"

    def as_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]


def build_generation_prompt(ctx: SemanticContext) -> Prompt:
    user = f"{ctx.render()}\n\n### Question\n{ctx.question}\n\n### SQL\n"
    return Prompt("generate", SYSTEM_INSTRUCTIONS, user, attempt=0, meta={"tables": ctx.tables})


def build_repair_prompt(
    ctx: SemanticContext,
    failed_sql: str,
    error_feedback: str,
    attempt: int,
    history: list[tuple[str, str]] | None = None,
) -> Prompt:
    """Repair prompt = question + schema/semantic grounding + previous SQL + exact validator errors."""
    parts = [ctx.render(), "", "### Question", ctx.question, ""]
    older = (history or [])[:-1]
    if older:
        parts.append("### Earlier failed attempts (do not repeat these mistakes)")
        for i, (sql, err) in enumerate(older, 1):
            parts += [f"Attempt {i}: {sql or '<empty>'}", f"Errors: {err}"]
        parts.append("")
    parts += [
        "### Previous SQL (INVALID)",
        failed_sql or "<empty output>",
        "",
        "### Why it failed (from the SQL validator)",
        error_feedback,
        "",
        "### Task",
        "Write a corrected SQLite SELECT statement that answers the question and fixes every error above. "
        "Return only the SQL.",
        "",
        "### Corrected SQL",
        "",
    ]
    return Prompt("repair", SYSTEM_INSTRUCTIONS, "\n".join(parts), attempt=attempt,
                  meta={"failed_sql": failed_sql, "errors": error_feedback})


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)
_START = re.compile(r"\b(SELECT|WITH)\b", re.I)


def extract_sql(raw: str | None) -> str:
    """Pull the SQL statement out of free-form model output."""
    if not raw:
        return ""
    text_ = raw.strip()
    m = _FENCE.search(text_)
    if m:
        text_ = m.group(1).strip()
    else:
        text_ = re.sub(r"^```(?:sql)?", "", text_).strip()
    # drop leading chatter like "Here is the query:" but keep non-SELECT statements
    # visible so the validator can reject them explicitly.
    m = _START.search(text_)
    lead = text_[: m.start()] if m else ""
    if m and not re.search(r"\b(insert|update|delete|drop|create|alter|pragma|attach)\b", lead, re.I):
        text_ = text_[m.start():]
    text_ = re.sub(r"^(SQL|Answer|Query)\s*:\s*", "", text_, flags=re.I)
    # stop at a blank line followed by prose / a second "Question:" block
    text_ = re.split(r"\n\s*\n(?=[A-Za-z#])(?!\s*(?:SELECT|FROM|WHERE|GROUP|ORDER|LIMIT|HAVING|JOIN|UNION)\b)", text_, flags=re.I)[0]
    text_ = re.split(r"\n\s*(?:###|Question:|Explanation:)", text_)[0]
    return text_.strip().rstrip(";").strip()