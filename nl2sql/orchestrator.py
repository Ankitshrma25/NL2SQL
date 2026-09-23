"""Chatbot orchestrator: the self-correcting NL -> SQL pipeline.

    Question
      -> SemanticLayer.build_context()          (schema linking, intent, hints)
      -> SLM.generate(generation prompt)        attempt 1
      -> SQLValidator.validate()
           valid   -> DatabaseExecutor.execute() -> answer
           invalid -> build_repair_prompt(question, schema, bad SQL, exact errors)
                      -> SLM.generate() -> validate()     (<= max_repair_retries times)
      -> after the retry budget: TemplateFallback -> validate() -> execute

The loop is a ``for`` over a fixed ``range`` - it cannot run more than
``1 + effective_retries`` SLM calls, whatever the model returns.
Every step is recorded in ``ChatResponse.attempts`` so the UI / logs can
show how the SQL evolved.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import PipelineConfig
from .executor import DatabaseExecutor, QueryResult, make_engine
from .fallback import TemplateFallback
from .prompts import build_generation_prompt, build_repair_prompt, extract_sql
from .semantic_layer import SemanticContext, SemanticLayer
from .slm.base import SQLGenerator
from .validator import SQLValidator, ValidationResult

log = logging.getLogger(__name__)

SOURCE_SLM = "slm"
SOURCE_SLM_REPAIRED = "slm_repaired"
SOURCE_FALLBACK = "template_fallback"
SOURCE_NONE = "none"


@dataclass
class Attempt:
    number: int  # 1-based
    stage: str  # mention the stage of the progress
    backend: str
    prompt: str | None
    raw_output: str | None
    sql: str
    validation: ValidationResult | None
    error: str | None = None  # backend exception, if any
    elapsed_ms: float = 0.0

    @property
    def valid(self) -> bool:
        return self.validation is not None and self.validation.is_valid

    def summary(self) -> dict:
        v = self.validation
        return {
            "attempt": self.number,
            "stage": self.stage,
            "backend": self.backend,
            "sql": self.sql,
            "valid": self.valid,
            "errors": [str(e) for e in v.errors] if v else ([self.error] if self.error else []),
            "warnings": [str(w) for w in v.warnings] if v else [],
            "checks_passed": v.checks_passed if v else [],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class ChatResponse:
    question: str
    sql: str | None
    source: str
    result: QueryResult | None
    attempts: list[Attempt] = field(default_factory=list)
    context: SemanticContext | None = None
    error: str | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def slm_calls(self) -> int:
        return sum(1 for a in self.attempts if a.stage in ("generate", "repair"))

    def history(self) -> list[dict]:
        return [a.summary() for a in self.attempts]


class NL2SQLChatbot:
    def __init__(
        self,
        semantic_layer: SemanticLayer,
        validator: SQLValidator,
        executor: DatabaseExecutor,
        slm: SQLGenerator | None,
        fallback: TemplateFallback | None = None,
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        self.semantic = semantic_layer
        self.validator = validator
        self.executor = executor
        self.slm = slm
        self.fallback = fallback if fallback is not None else TemplateFallback(self.config.default_limit)
        self.loaded = None  

    # @classmethod
    # def from_config(cls, config: PipelineConfig | None = None, slm: SQLGenerator | None = None,
    #                 use_hf_model: bool = True) -> "NL2SQLChatbot":
    #     """Wire all components from configuration (the only place they are assembled)."""
    #     config = config or PipelineConfig()
    #     engine = make_engine(config.db_url)
    #     semantic = SemanticLayer.from_engine(engine, config.metadata_path, config.categorical_max_distinct,
    #                                          config.fuzzy_threshold)
    #     executor = DatabaseExecutor(engine, config.max_result_rows)
    #     validator = SQLValidator(semantic.schema, executor.dry_run if config.enable_db_dry_run else None)
    #     if slm is None and use_hf_model:
    #         from .slm.hf_backend import HuggingFaceSQLGenerator
    #         slm = HuggingFaceSQLGenerator(config.model)
    #     return cls(semantic, validator, executor, slm, TemplateFallback(config.default_limit), config)

    @classmethod
    def from_config(cls, config: PipelineConfig | None = None, slm: SQLGenerator | None = None,
                    use_hf_model: bool = True) -> "NL2SQLChatbot":
        config = config or PipelineConfig()
        return cls._assemble(make_engine(config.db_url), config, slm, use_hf_model,
                             metadata_path=config.metadata_path)

    @classmethod
    def from_loaded(cls, loaded, config: PipelineConfig | None = None, slm: SQLGenerator | None = None,
                    use_hf_model: bool = True) -> "NL2SQLChatbot":
        """Build a chatbot over a table supplied at runtime (see nl2sql.ingest.load_table)."""
        config = config or PipelineConfig()
        config.db_url = f"sqlite:///{loaded.db_path}"
        config.metadata_path = None
        return cls._assemble(loaded.engine, config, slm, use_hf_model, metadata=loaded.metadata())

    @classmethod
    def from_csv(cls, csv_path=None, schema=None, table_name: str | None = None,
                 config: PipelineConfig | None = None, slm: SQLGenerator | None = None,
                 use_hf_model: bool = True) -> "NL2SQLChatbot":
        """CSV + column descriptions -> ready chatbot. bot.loaded.warnings lists any issues."""
        from .ingest import load_table
        loaded = load_table(csv_path, schema, table_name)
        bot = cls.from_loaded(loaded, config, slm, use_hf_model)
        bot.loaded = loaded
        return bot

    @classmethod
    def _assemble(cls, engine, config: PipelineConfig, slm, use_hf_model: bool,
                  metadata_path: str | None = None, metadata: dict | None = None) -> "NL2SQLChatbot":
        semantic = SemanticLayer.from_engine(engine, metadata_path, config.categorical_max_distinct,
                                             config.fuzzy_threshold, metadata=metadata)
        executor = DatabaseExecutor(engine, config.max_result_rows)
        validator = SQLValidator(semantic.schema, executor.dry_run if config.enable_db_dry_run else None)
        if slm is None and use_hf_model:
            from .slm.hf_backend import HuggingFaceSQLGenerator
            slm = HuggingFaceSQLGenerator(config.model)
        return cls(semantic, validator, executor, slm, TemplateFallback(config.default_limit), config)

    # ------------------------------------------------------------------
    def ask(self, question: str) -> ChatResponse:
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return ChatResponse(question, None, SOURCE_NONE, None, error="Please ask a question.")

        ctx = self.semantic.build_context(question)
        attempts: list[Attempt] = []
        failures: list[tuple[str, str]] = []  # (sql, error feedback) per failed SLM attempt
        max_calls = 1 + self.config.effective_retries

        if self.slm is not None:
            prompt = build_generation_prompt(ctx)
            for i in range(max_calls):  # hard bound - no while loop
                stage = "generate" if i == 0 else "repair"
                ts = time.perf_counter()
                raw, err = None, None
                try:
                    raw = self.slm.generate(prompt)
                except Exception as e:  # model crash / weights missing -> counts as a failed attempt
                    err = f"{type(e).__name__}: {e}"
                    log.warning("SLM attempt %d failed: %s", i + 1, err)
                sql = extract_sql(raw) if raw is not None else ""
                validation = self.validator.validate(sql) if err is None else None
                att = Attempt(i + 1, stage, getattr(self.slm, "name", "slm"), prompt.as_text(), raw, sql,
                              validation, err, (time.perf_counter() - ts) * 1000)
                attempts.append(att)

                if att.valid:
                    return self._finish(question, ctx, attempts, validation,
                                        SOURCE_SLM if i == 0 else SOURCE_SLM_REPAIRED, t0)
                if err is not None and _is_unavailable(err):
                    break  # weights missing: retrying cannot help, go straight to fallback

                feedback = validation.feedback() if validation else f"- [MODEL_ERROR] {err}"
                failures.append((sql, feedback))
                if i + 1 < max_calls:
                    prompt = build_repair_prompt(ctx, sql, feedback, attempt=i + 1, history=failures)

        # ---- deterministic fallback (reliability mechanism) ----
        if self.config.enable_fallback and self.fallback is not None:
            ts = time.perf_counter()
            fb = self.fallback.generate(ctx.analysis)
            validation = self.validator.validate(fb.sql) if fb.sql else None
            attempts.append(Attempt(len(attempts) + 1, "fallback", f"{self.fallback.name}:{fb.template}", None,
                                    None, fb.sql or "", validation, fb.reason or None,
                                    (time.perf_counter() - ts) * 1000))
            if validation is not None and validation.is_valid:
                return self._finish(question, ctx, attempts, validation, SOURCE_FALLBACK, t0)

        last = attempts[-1] if attempts else None
        why = (last.validation.feedback() if last and last.validation else (last.error if last else "no backend"))
        return ChatResponse(question, None, SOURCE_NONE, None, attempts, ctx,
                            error=f"Could not produce a valid SQL query. {why}",
                            elapsed_ms=(time.perf_counter() - t0) * 1000)

    def _finish(self, question, ctx, attempts, validation: ValidationResult, source, t0) -> ChatResponse:
        try:
            result = self.executor.execute(validation.query)  # only ValidatedQuery reaches the DB
        except Exception as e:
            return ChatResponse(question, validation.sql, source, None, attempts, ctx,
                                error=f"Execution failed: {e}", elapsed_ms=(time.perf_counter() - t0) * 1000)
        return ChatResponse(question, validation.sql, source, result, attempts, ctx,
                            elapsed_ms=(time.perf_counter() - t0) * 1000)


def _is_unavailable(err: str) -> bool:
    return "SLMError" in err and ("Could not load" in err or "not installed" in err or "not found" in err)
