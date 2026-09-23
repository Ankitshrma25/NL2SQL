"""Run the assessment's 10 validation questions and compare RESULT SETS with the expected SQL.

    python scripts/run_validation.py                # with the offline SLM
    python scripts/run_validation.py --no-model     # semantic layer + template fallback only
    python scripts/run_validation.py --spec other.json   # any other table / question set

A question passes when the bot's SQL returns the same rows as the expected
SQL (column names and SQL text may differ). Row order is compared only
when the expected query has an ORDER BY that matters ("ordered": true).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nl2sql.config import PipelineConfig  # noqa: E402
from nl2sql.orchestrator import NL2SQLChatbot  # noqa: E402


def _norm(rows, ordered: bool):
    out = [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows]
    return out if ordered else sorted(out, key=repr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(ROOT / "data" / "task2_validation.json"))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every attempt")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    base = Path(args.spec).parent
    schema_text = (base / spec["schema"]).read_text(encoding="utf-8") if spec.get("schema") else None
    bot = NL2SQLChatbot.from_csv(base / spec["csv"], schema_text, spec.get("table"), PipelineConfig(),
                                 use_hf_model=not args.no_model)
    print(f"Table '{bot.loaded.table_name}': {bot.loaded.row_count} rows | backend: "
          f"{'template only' if args.no_model else bot.slm.describe()}")
    for w in bot.loaded.warnings:
        print("  note:", w)

    ref = sqlite3.connect(bot.loaded.db_path)  # ground truth runs directly on the same data
    passed, t_all = 0, time.perf_counter()
    for q in spec["questions"]:
        resp = bot.ask(q["question"])
        expected = ref.execute(q["expected_sql"]).fetchall()
        got = resp.result.rows if resp.ok else None
        ok = got is not None and _norm(got, q.get("ordered", False)) == _norm(expected, q.get("ordered", False))
        passed += ok
        print(f"\n[{q['id']:>2}] {'PASS' if ok else 'FAIL'}  {q['question']}")
        print(f"     source={resp.source}  slm_calls={resp.slm_calls}  {resp.elapsed_ms:.0f} ms")
        print(f"     SQL: {resp.sql}")
        if not ok:
            print(f"     expected: {q['expected_sql']}")
            print(f"     got {len(got) if got is not None else 'no'} rows, expected {len(expected)}"
                  + (f" | error: {resp.error}" if resp.error else ""))
        if args.verbose or not ok:
            for a in resp.history():
                errs = "; ".join(a["errors"]) or "valid"
                print(f"       attempt {a['attempt']} ({a['stage']}): {a['sql']}  ->  {errs}")
    print(f"\n{passed}/{len(spec['questions'])} passed in {time.perf_counter() - t_all:.1f} s")
    return 0 if passed == len(spec["questions"]) else 1


if __name__ == "__main__":
    sys.exit(main())
