"""The decision engine — the only module in the codebase that knows about Laya.

Everything upstream (router, skills, API) talks to :class:`DecisionEngine`, so
swapping Laya for another small model, or for an HTTP client in front of
``laya-serve``, is a one-file change. ``import laya`` appearing anywhere else is
a bug, not a shortcut.

Laya answers *typed questions*, it does not generate text. A question is one of:

* ``choice`` — pick one labelled option (this is how the router picks a skill);
* ``score``  — place the input on an ordered scale;
* ``noul``   — a calibrated yes/no probability.

All of them are evaluated in a single forward pass, which is why :meth:`ask` —
the batch method — is the one the router actually calls: ten questions in one
``predict`` cost far less than ten single-question calls.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_MODEL = "multilingual"
DEFAULT_DEVICE = "cpu"


@dataclass(frozen=True)
class ChoiceResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    confidence: float
    distribution: dict[str, float]


@dataclass(frozen=True)
class NoulResult:
    probability: float
    confidence: float


Answer = ChoiceResult | ScoreResult | NoulResult


class DecisionEngine(Protocol):
    """A small model that answers typed questions about a piece of text."""

    def choice(
        self, state: str, name: str, instructions: str, criteria: dict[str, str]
    ) -> ChoiceResult: ...

    def score(
        self, state: str, name: str, instructions: str, criteria: list[str]
    ) -> ScoreResult: ...

    def noul(self, state: str, name: str, instructions: str) -> NoulResult: ...

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Answer]: ...


class SingleQuestionMixin:
    """The three single-question helpers, expressed as one-question batches.

    An implementation only has to provide :meth:`ask`; test doubles included.
    """

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Answer]:
        raise NotImplementedError

    def choice(
        self, state: str, name: str, instructions: str, criteria: dict[str, str]
    ) -> ChoiceResult:
        question = {"type": "choice", "instructions": instructions, "criteria": criteria}
        return self.ask(state, {name: question})[name]

    def score(self, state: str, name: str, instructions: str, criteria: list[str]) -> ScoreResult:
        question = {"type": "score", "instructions": instructions, "criteria": criteria}
        return self.ask(state, {name: question})[name]

    def noul(self, state: str, name: str, instructions: str) -> NoulResult:
        return self.ask(state, {name: {"type": "noul", "instructions": instructions}})[name]


def _to_answer(raw: dict[str, Any]) -> Answer:
    """Map one raw Laya answer onto our own result types.

    Laya names the score spread ``probabilities``; we expose it as
    ``distribution`` so ``probabilities`` means "over labels" everywhere.
    """
    kind = raw["type"]
    if kind == "choice":
        return ChoiceResult(raw["choice"], raw["confidence"], raw["probabilities"])
    if kind == "score":
        return ScoreResult(raw["score"], raw["confidence"], raw["probabilities"])
    if kind == "noul":
        return NoulResult(raw["noul"], raw["confidence"])
    raise ValueError(f"unknown answer type: {kind}")


class LayaEngine(SingleQuestionMixin):
    """Laya, loaded once and called under a lock.

    The checkpoint is ``convaiinnovations/laya`` + ``subfolder="multilingual"``
    (mmBERT-base, 322M, 100+ languages). The English root checkpoint is not an
    option here: Pixel's UI is Russian, and the English weights answer Cyrillic
    input confidently and wrongly.
    """

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        # Imported here, not at module scope: this is deliberately the only
        # `import laya` in the tree, and it also keeps torch off the import path
        # of anything that never builds a LayaEngine (the whole test suite).
        import laya

        model = model or os.environ.get("LAYA_MODEL", DEFAULT_MODEL)
        device = device or os.environ.get("LAYA_DEVICE", DEFAULT_DEVICE)
        repo, subfolder = laya.DEFAULT_MODELS[model]
        self._agent = laya.load(repo, subfolder=subfolder, device=device)
        # One shared model, one forward pass at a time. Never taken together
        # with `db.lock` — see the ordering note in `backend/api.py`.
        self._lock = threading.Lock()

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Answer]:
        if not questions:
            return {}
        with self._lock:
            raw = self._agent.predict(state, questions)
        return {name: _to_answer(answer) for name, answer in raw["answers"].items()}


_engine: DecisionEngine | None = None


def set_engine(engine: DecisionEngine | None) -> None:
    global _engine
    _engine = engine


def get_engine() -> DecisionEngine:
    if _engine is None:
        raise RuntimeError("decision engine is not loaded")
    return _engine
