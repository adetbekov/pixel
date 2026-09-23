"""The fast path: two forward passes turn a sentence into a plan.

1. **Which skill?** One ``choice`` question over every active skill plus a
   mandatory ``unknown`` option. Without ``unknown`` the model is forced to pick
   *something*, and "расскажи про квантовую физику" becomes a dance.
2. **How should it behave?** Every question of the chosen skill, batched into a
   single ``ask``. A skill with no questions skips this pass entirely.

Below the skill's threshold, or on ``unknown``, the router reports a
:class:`RouterMiss` and executes nothing. Stage 3 hands that miss to Gemini;
until then the caller answers with a polite stub.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..state import RobotState
from .engine import ChoiceResult, DecisionEngine
from .skill import Skill
from .verbalize import command_only, command_with_state

DEFAULT_THRESHOLD = 0.6

ROUTER_QUESTION = "skill"
ROUTER_INSTRUCTIONS = "Какой навык робота подходит к команде пользователя"

UNKNOWN = "unknown"
UNKNOWN_DESCRIPTION = "никакой"

#: `head_max_len` on the multilingual checkpoint is 256 tokens for the whole
#: option head, so a verbose skill description would push the others out.
MAX_DESCRIPTION_LEN = 120

# TODO(stage 4): past ~50 active skills the single choice head stops being the
# right shape. Switch to:
#   laya.predict_shortlist(agent, state, questions,
#                          embed_fn=laya.embed_fn_from_agent(agent), k=20)
SHORTLIST_ABOVE = 50


@dataclass(frozen=True)
class RouterHit:
    skill_id: str
    confidence: float
    raw_plan: list[dict[str, Any]]


@dataclass(frozen=True)
class RouterMiss:
    confidence: float


RouterOutcome = RouterHit | RouterMiss


def default_threshold() -> float:
    return float(os.environ.get("ROUTER_THRESHOLD", DEFAULT_THRESHOLD))


def build_criteria(skills: list[Skill]) -> dict[str, str]:
    criteria = {skill.id: skill.description[:MAX_DESCRIPTION_LEN] for skill in skills}
    criteria[UNKNOWN] = UNKNOWN_DESCRIPTION
    return criteria


def route(
    engine: DecisionEngine, skills: list[Skill], text: str, state: RobotState
) -> RouterOutcome:
    if not skills:
        return RouterMiss(0.0)

    # Pass 1 deliberately sees the command alone: the robot's mood does not
    # decide *which* skill was asked for, and feeding it in measurably drags the
    # choice towards `sleep` (see backend/brain/verbalize.py).
    picked = engine.choice(
        command_only(text), ROUTER_QUESTION, ROUTER_INSTRUCTIONS, build_criteria(skills)
    )
    if not isinstance(picked, ChoiceResult):
        return RouterMiss(0.0)

    by_id = {skill.id: skill for skill in skills}
    # `unknown` is not a skill, and neither is a label the engine invented.
    skill = by_id.get(picked.label)
    if skill is None:
        return RouterMiss(picked.confidence)

    threshold = skill.threshold if skill.threshold is not None else default_threshold()
    if picked.confidence < threshold:
        return RouterMiss(picked.confidence)

    # Pass 2 is where the state matters, so the skill's own questions get it.
    answers = engine.ask(command_with_state(text, state), skill.to_questions())
    rule = skill.select_rule(answers, state)
    return RouterHit(skill.id, picked.confidence, rule.actions)
