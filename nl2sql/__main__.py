"""Command-line interface.

    python -m nl2sql "total amount by city"
    python -m nl2sql --no-model "top 3 customers by total amount"
    python -m nl2sql                      # interactive REPL
"""
from __future__ import annotations

import argparse
import json
import logging

from .config import PipelineConfig
from .orchestrator import NL2SQLChatbot
from pathlib import Path


def _print(resp, verbose: bool) -> None:
    print(f"\nSource: {resp.source}   SLM calls: {resp.slm_calls}   {resp.elapsed_ms:.0f} ms")
    if verbose:
        print(json.dumps(resp.history(), indent=2, default=str))
    print(f"SQL: {resp.sql}")
    if not resp.ok:
        print("ERROR:", resp.error)
        return
    cols = resp.result.columns
    print(" | ".join(cols))
    print("-+-".join("-" * len(c) for c in cols))
    for row in resp.result.rows[:50]:
        print(" | ".join("" if v is None else str(v) for v in row))
    if len(resp.result.rows) > 50:
        print(f"... ({len(resp.result.rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline NL->SQL chatbot")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--db", help="SQLAlchemy URL, e.g. sqlite:///path/to.db")

    ap.add_argument("--csv", help="CSV file to load as the table (runtime schema)")
    ap.add_argument("--schema", help="text/JSON file with 'column: description' lines")
    ap.add_argument("--table", help="table name for --csv (default: derived from the file name)")
    
    ap.add_argument("--metadata", help="optional schema metadata JSON")
    ap.add_argument("--no-model", action="store_true", help="skip the SLM; use the template backend only")
    ap.add_argument("--retries", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true", help="print attempt history")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cfg = PipelineConfig()
    if args.db:
        cfg.db_url = args.db
        cfg.metadata_path = args.metadata
    elif args.metadata:
        cfg.metadata_path = args.metadata
    if args.retries is not None:
        cfg.max_repair_retries = args.retries
    # bot = NL2SQLChatbot.from_config(cfg, use_hf_model=not args.no_model)
    
    if args.csv or args.schema:
        schema_text = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        bot = NL2SQLChatbot.from_csv(args.csv, schema_text, args.table, cfg, use_hf_model=not args.no_model)
        ld = bot.loaded
        print(f"Loaded table '{ld.table_name}': {ld.row_count} rows, {len(ld.columns)} columns, "
              f"{sum(1 for d in ld.descriptions.values() if d)} described")
        for w in ld.warnings:
            print("  note:", w)
    else:
        bot = NL2SQLChatbot.from_config(cfg, use_hf_model=not args.no_model)

    if args.question:
        _print(bot.ask(" ".join(args.question)), args.verbose)
        return
    print("Ask a question (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        _print(bot.ask(q), args.verbose)


if __name__ == "__main__":
    main()
