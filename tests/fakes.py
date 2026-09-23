"""Scripted stand-ins for both models, so CI needs neither weights nor network.

``FakeEngine`` replaces Laya. ``FakeGeminiClient`` replaces the ``google.genai``
client inside :class:`backend.teacher.client.GeminiTeacher`, which takes its
client as an argument precisely so the retry and fallback paths are testable.
"""

from dataclasses import dataclass
from typing import Any

from backend.brain.engine import Answer, ChoiceResult, SingleQuestionMixin
from backend.brain.router import ROUTER_QUESTION


class FakeEngine(SingleQuestionMixin):
    def __init__(
        self,
        pick: str = "unknown",
        confidence: float = 0.9,
        answers: dict[str, Answer] | None = None,
    ) -> None:
        self.pick = pick
        self.confidence = confidence
        self.answers = answers or {}
        self.calls: list[dict[str, Any]] = []

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Answer]:
        if not questions:
            # Laya short-circuits an empty question set without a forward pass,
            # so a call that never reaches the model is not counted as one.
            return {}
        self.calls.append(questions)
        if ROUTER_QUESTION in questions:
            options = questions[ROUTER_QUESTION]["criteria"]
            probabilities = {key: float(key == self.pick) for key in options}
            return {ROUTER_QUESTION: ChoiceResult(self.pick, self.confidence, probabilities)}
        return {name: self.answers[name] for name in questions if name in self.answers}


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
