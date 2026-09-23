"""A decision engine with scripted answers and no model behind it.

CI never downloads weights, so every test that exercises the router, the skills
or /api/chat runs against this instead of Laya.
"""

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
