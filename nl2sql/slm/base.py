"""SLM backend interface.

The orchestrator depends only on this protocol, so the model can be
swapped (another HF checkpoint, llama.cpp, ONNX, a test double) without
touching the rest of the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..prompts import Prompt


class SLMError(RuntimeError):
    """Raised when a backend cannot produce output (e.g. weights missing)."""


class SQLGenerator(ABC):
    name: str = "slm"

    @abstractmethod
    def generate(self, prompt: Prompt) -> str:
        """Return raw model text for a generation or repair prompt."""

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return self.name
