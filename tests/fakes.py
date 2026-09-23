"""Scripted stand-ins for both models, so CI needs neither weights nor network.

``FakeEngine`` replaces Laya. ``FakeGeminiClient`` replaces the ``google.genai``
client inside :class:`backend.teacher.client.GeminiTeacher`, which takes its
client as an argument precisely so the retry and fallback paths are testable.
"""

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

from backend.brain.engine import Answer, ChoiceResult, SingleQuestionMixin
from backend.brain.router import ROUTER_QUESTION

EMBED_DIM = 64


def hash_vector(text: str) -> list[float]:
    """A stable vector for a text no test scripted.

    Deterministic, and effectively orthogonal to every other one: in 64
    dimensions two of these sit far below any clustering threshold. That is the
    point — an unscripted command must not drift into somebody else's cluster.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * (EMBED_DIM // len(digest) + 1))[:EMBED_DIM]
    return [value / 128.0 - 1.0 for value in struct.unpack(f"{EMBED_DIM}B", raw)]


class FakeEngine(SingleQuestionMixin):
    """Laya, scripted.

    ``pick`` is the skill the router gets back for anything; ``routes`` overrides
    it per command. The override is what a whole *library* needs: one fake that
    answers "greet" to every phrase cannot show that a mined skill left the other
    skills alone, which is the regression check stage 4 turns on.

    ``embeddings`` is the same idea for :meth:`embed` — scripted vectors where a
    test means something by them, :func:`hash_vector` everywhere else. A fake
    does not get to imitate a sentence encoder; it gets to be predictable.
    """

    def __init__(
        self,
        pick: str = "unknown",
        confidence: float = 0.9,
        answers: dict[str, Answer] | None = None,
        routes: dict[str, str] | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self.pick = pick
        self.confidence = confidence
        self.answers = answers or {}
        self.routes = routes or {}
        self.embeddings = embeddings or {}
        self.calls: list[dict[str, Any]] = []
        #: The verbalized prompt of every counted call, so a test can assert what
        #: text actually reached the model, not just that it was asked something.
        self.prompts: list[str] = []
        self.embedded: list[list[str]] = []

    def _pick_for(self, state: str) -> str:
        # `state` is the verbalised prompt, so the command is a substring of it.
        for command, skill_id in self.routes.items():
            if command in state:
                return skill_id
        return self.pick

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Answer]:
        if not questions:
            # Laya short-circuits an empty question set without a forward pass,
            # so a call that never reaches the model is not counted as one.
            return {}
        self.calls.append(questions)
        self.prompts.append(state)
        if ROUTER_QUESTION in questions:
            options = questions[ROUTER_QUESTION]["criteria"]
            picked = self._pick_for(state)
            probabilities = {key: float(key == picked) for key in options}
            return {ROUTER_QUESTION: ChoiceResult(picked, self.confidence, probabilities)}
        return {name: self.answers[name] for name in questions if name in self.answers}

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.append(list(texts))
        return [self.embeddings.get(text) or hash_vector(text) for text in texts]


@dataclass
class FakeInteraction:
    output_text: str


class FakeInteractions:
    """``client.interactions`` — one scripted answer per call.

    A scripted entry is either the raw text the model would return, or an
    exception instance to raise instead (a timeout, a 500). The last entry is
    reused if the teacher asks more times than the script has answers.
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeInteraction:
        self.calls.append(kwargs)
        answer = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        return FakeInteraction(output_text=answer)


class FakeGeminiClient:
    def __init__(self, *script: Any) -> None:
        self.interactions = FakeInteractions(list(script))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.interactions.calls
