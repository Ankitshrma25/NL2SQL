"""Test doubles for the SLM (used by the test suite and the UI's demo mode)."""
from __future__ import annotations

from typing import Callable, Iterable

from ..prompts import Prompt
from .base import SLMError, SQLGenerator


class ScriptedSLM(SQLGenerator):
    """Returns a fixed sequence of outputs, one per call, and records every prompt.

    Items may be strings or Exceptions (raised to simulate a model crash).
    If more calls are made than scripted, the last item is repeated.
    """

    name = "scripted"

    def __init__(self, outputs: Iterable[str | Exception]):
        self.outputs = list(outputs)
        if not self.outputs:
            raise ValueError("ScriptedSLM needs at least one output")
        self.prompts: list[Prompt] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def generate(self, prompt: Prompt) -> str:
        idx = min(len(self.prompts), len(self.outputs) - 1)
        self.prompts.append(prompt)
        out = self.outputs[idx]
        if isinstance(out, Exception):
            raise out
        return out


class RuleSLM(SQLGenerator):
    """Maps the question to SQL with a user-supplied function (oracle for e2e tests)."""

    name = "rule"

    def __init__(self, fn: Callable[[str, Prompt], str]):
        self.fn = fn
        self.prompts: list[Prompt] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def generate(self, prompt: Prompt) -> str:
        self.prompts.append(prompt)
        question = prompt.user.split("### Question\n", 1)[-1].split("\n", 1)[0]
        return self.fn(question, prompt)


class UnavailableSLM(SQLGenerator):
    """Simulates missing weights."""

    name = "unavailable"

    def __init__(self, reason: str = "model weights not found"):
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def generate(self, prompt: Prompt) -> str:
        raise SLMError(self.reason)
