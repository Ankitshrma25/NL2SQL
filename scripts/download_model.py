"""One-time, ONLINE step: download the SLM weights into a plain local folder.

    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out models/qwen2.5-coder-0.5b

The default (Qwen/Qwen2.5-Coder-1.5B-Instruct -> models/qwen2.5-coder-1.5b) is
where nl2sql/config.py looks, so nothing else needs configuring. The script
finishes by loading the model with networking disabled and generating one
query, proving the folder is complete. After that everything runs offline.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Hugging Face model id")
    ap.add_argument("--out", default=str(ROOT / "models" / "qwen2.5-coder-1.5b"), help="target folder")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model} -> {args.out}  (about 3 GB for the 1.5B model; re-run to resume)")
    snapshot_download(repo_id=args.model, local_dir=args.out,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt", "vocab.json"])

    # verify: load with networking disabled
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from nl2sql.config import ModelConfig
    from nl2sql.prompts import Prompt
    from nl2sql.slm.hf_backend import HuggingFaceSQLGenerator

    gen = HuggingFaceSQLGenerator(ModelConfig(model_name_or_path=args.out, local_files_only=True))
    out = gen.generate(Prompt("generate", "Return only SQL.",
                              "CREATE TABLE t (city TEXT, amount REAL);\n### Question\ntotal amount by city\n### SQL\n"))
    print("Offline smoke test output:\n", out)
    if Path(args.out).resolve() != (ROOT / "models" / "qwen2.5-coder-1.5b").resolve():
        print(f"\nNon-default folder: set NL2SQL_MODEL={args.out}")


if __name__ == "__main__":
    main()