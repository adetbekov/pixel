"""The action library — the only things Pixel can physically do.

The library is fixed: neither a mined skill nor Gemini may add a primitive. Every
plan produced anywhere in the system passes through :func:`validate_plan`, which
drops whatever it does not recognise. That is the project's safety guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_PLAN_LEN = 8
MAX_SAY_LEN = 200

FACES = ("happy", "sad", "sleepy", "angry", "curious")


class InvalidPlan(ValueError):
    """Raised when a plan is structurally unusable (too long, not a list)."""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Action:
    action: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "args": dict(self.args)}


ACTIONS: dict[str, ActionSpec] = {
    "jump": ActionSpec("jump", "Подпрыгнуть на месте"),
    "dance": ActionSpec("dance", "Станцевать"),
    "sleep": ActionSpec("sleep", "Уснуть и восстановить энергию"),
    "eat": ActionSpec("eat", "Поесть"),
    "spin": ActionSpec("spin", "Покрутиться вокруг своей оси"),
    "wave": ActionSpec("wave", "Помахать рукой"),
    "say": ActionSpec("say", "Сказать фразу", ("text",)),
    "set_face": ActionSpec("set_face", "Сменить выражение лица", ("face",)),
}


def _clean(step: Any) -> Action | None:
    """Normalise one raw step, or return None if it is not a library action."""
    if not isinstance(step, dict):
        return None
    name = step.get("action")
    if name not in ACTIONS:
        return None
    raw_args = step.get("args") or {}
    if not isinstance(raw_args, dict):
        return None

    if name == "say":
        text = raw_args.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return Action("say", {"text": text[:MAX_SAY_LEN]})

    if name == "set_face":
        face = raw_args.get("face")
        if face not in FACES:
            return None
        return Action("set_face", {"face": face})

    return Action(name, {})


def validate_plan(plan: list[dict]) -> list[Action]:
    """Turn a raw plan into executable actions.

    Unknown actions and malformed arguments are silently dropped — an LLM that
    invents ``launch_rocket`` gets it removed, not executed. ``InvalidPlan`` is
    raised only when the plan itself is unusable: not a list, or longer than
    :data:`MAX_PLAN_LEN` steps.
    """
    if not isinstance(plan, list):
        raise InvalidPlan("plan must be a list")
    if len(plan) > MAX_PLAN_LEN:
        raise InvalidPlan(f"plan too long: {len(plan)} > {MAX_PLAN_LEN}")

    return [action for action in (_clean(step) for step in plan) if action is not None]
