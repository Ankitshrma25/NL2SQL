"""Central configuration.

Every tunable of the pipeline lives here so that components never read
environment variables or hard-code paths themselves.  Values can be
overridden with environment variables prefixed ``NL2SQL_``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"NL2SQL_{name}", default)


@dataclass
class ModelConfig:
    """Settings for the offline small language model (SLM)."""

    # Local folder with the downloaded weights (scripts/download_model.py puts them here).
    # Any causal-LM (chat) or seq2seq text-to-SQL checkpoint works; the backend
    # auto-detects the architecture.  A Hugging Face id also works if it is cached.
    model_name_or_path: str = _env("MODEL", str(PROJECT_ROOT / "models" / "qwen2.5-coder-1.5b"))
    # Local directory holding cached weights (see scripts/download_model.py).
    cache_dir: str = _env("MODEL_CACHE", str(PROJECT_ROOT / "models"))
    # Never touch the network at inference time.
    local_files_only: bool = _env("LOCAL_FILES_ONLY", "1") == "1"
    device: str = _env("DEVICE", "auto")  # "auto" | "cpu" | "cuda" | "mps"
    max_new_tokens: int = int(_env("MAX_NEW_TOKENS", "128"))  # a SQL answer is ~30-60 tokens
    temperature: float = float(_env("TEMPERATURE", "0.0"))  # 0 => greedy / deterministic


@dataclass
class PipelineConfig:
    """Settings for the orchestrator, validator and executor."""

    db_url: str = _env("DB_URL", f"sqlite:///{DATA_DIR / 'sample.db'}")
    # Optional JSON file with human-written column descriptions / synonyms.
    metadata_path: str | None = _env("METADATA", str(DATA_DIR / "schema_metadata.json")) or None
    # Retries AFTER the initial generation.  Total SLM calls <= 1 + max_repair_retries.
    max_repair_retries: int = int(_env("MAX_RETRIES", "2"))
    # Hard upper bound regardless of what the caller asks for (loop-safety guarantee).
    retry_hard_cap: int = 5
    enable_fallback: bool = _env("ENABLE_FALLBACK", "1") == "1"
    enable_db_dry_run: bool = _env("DB_DRY_RUN", "1") == "1"
    max_result_rows: int = int(_env("MAX_ROWS", "500"))
    default_limit: int = int(_env("DEFAULT_LIMIT", "100"))
    # Columns with <= this many distinct values are treated as categorical.
    categorical_max_distinct: int = int(_env("CATEGORICAL_MAX_DISTINCT", "25"))
    fuzzy_threshold: int = int(_env("FUZZY_THRESHOLD", "84"))
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def effective_retries(self) -> int:
        return max(0, min(self.max_repair_retries, self.retry_hard_cap))