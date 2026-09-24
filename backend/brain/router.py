"""The fast path: a lookup, then at most two forward passes.

0. **Have we been told this exact phrase?** Every skill lists the commands it
   was written (or mined) for in ``examples``, and until now nothing read them —
   the ``choice`` head only ever saw ``description``. So a phrase the skill
   itself claims could sit under its own threshold: measured on the live
   checkpoint, ``сделай фокус`` reached ``show_trick`` at 0.55 and ``хай``
   reached ``greet`` at 0.22 (see ``scripts/calibrate_router.py``). A normalised
   exact match now routes straight to that skill at confidence 1.0, costing no
   forward pass at all.
1. **Which skill?** One ``choice`` question over every active skill plus a
   mandatory ``unknown`` option. Without ``unknown`` the model is forced to pick
   *something*, and "расскажи про квантовую физику" becomes a dance.
2. **How should it behave?** Every question of the chosen skill, batched into a
   single ``ask``. A skill with no questions skips this pass entirely.

Below the skill's threshold, or on ``unknown``, the router reports a
:class:`RouterMiss` and executes nothing. The miss goes to Gemini, and the
teacher's answer is what the miner later turns into a skill.

``use_examples=False`` turns step 0 off. The miner needs that: its backtest is
the one place that has to measure what the *head* does, and a candidate's
``examples`` are exactly the cluster it is being tested on — the lookup would
score every candidate 1.0 and prove nothing (see ``backend/miner/backtest.py``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ..state import RobotState
from .engine import ChoiceResult, DecisionEngine
from .skill import Skill
from .verbalize import command_only, command_with_state

#: Calibrated against the live checkpoint, not guessed (JEB-1548). Over both the
#: 4-skill starter library and the 5-skill one the miner produces, 0.66 keeps the
#: hit rate of 0.60 exactly (71% and 67%) and takes wrong fires from 5 to 3 —
#: it is the knee, and every step past it only loses hits. Re-measure with
#: ``scripts/calibrate_router.py`` after changing the checkpoint or the library:
#: the number is a property of the option set, not of any one skill, and the
#: same probe moves by up to 0.38 between those two registries.
DEFAULT_THRESHOLD = 0.66

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

#: An exact-match lookup that trips over a comma is not worth having. Case, `ё`,
#: punctuation and repeated spaces are dropped; word order and wording are not,
#: because that is the head's job.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


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


def normalize(text: str) -> str:
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", text.lower().replace("ё", "е"))).strip()


def example_index(skills: list[Skill]) -> dict[str, Skill]:
    """``normalised example -> skill``, first skill in registry order winning.

    Registry order is the same order the ``choice`` head sees, and a mined skill
    is appended — so a candidate that copies a phrase off an older skill cannot
    take it over through this path.
    """
    index: dict[str, Skill] = {}
    for skill in skills:
        for example in skill.examples:
            index.setdefault(normalize(example), skill)
    return index


def build_criteria(skills: list[Skill]) -> dict[str, str]:
    criteria = {skill.id: skill.description[:MAX_DESCRIPTION_LEN] for skill in skills}
    criteria[UNKNOWN] = UNKNOWN_DESCRIPTION
    return criteria


def pick_skill(
    engine: DecisionEngine, skills: list[Skill], text: str
) -> tuple[Skill | None, float]:
    """Pass 1 alone: which skill wins and how sure, with no plan built.

    Separate from :func:`route` because a caller that only wants to know *which*
    skill a phrase reaches should not pay pass 2 for it. The miner's two control
    checks are exactly that caller, and each of theirs is one forward pass here
    against two in a full ``route`` (JEB-1548 review). It is also why they cannot
    be answered by the ``examples`` lookup: this entry point does not have one.
    """
    if not skills:
        return None, 0.0

    # Pass 1 deliberately sees the command alone: the robot's mood does not
    # decide *which* skill was asked for, and feeding it in measurably drags the
    # choice towards `sleep` (see backend/brain/verbalize.py).
    picked = engine.choice(
        command_only(text), ROUTER_QUESTION, ROUTER_INSTRUCTIONS, build_criteria(skills)
    )
    if not isinstance(picked, ChoiceResult):
        return None, 0.0

    # `unknown` is not a skill, and neither is a label the engine invented.
    skill = {one.id: one for one in skills}.get(picked.label)
    if skill is None:
        return None, picked.confidence

    threshold = skill.threshold if skill.threshold is not None else default_threshold()
    if picked.confidence < threshold:
        return None, picked.confidence
    return skill, picked.confidence


def route(
    engine: DecisionEngine,
    skills: list[Skill],
    text: str,
    state: RobotState,
    *,
    use_examples: bool = True,
) -> RouterOutcome:
    if not skills:
        return RouterMiss(0.0)

    skill = example_index(skills).get(normalize(text)) if use_examples else None
    confidence = 1.0

    if skill is None:
        skill, confidence = pick_skill(engine, skills, text)
        if skill is None:
            return RouterMiss(confidence)

    # Pass 2 is where the state matters, so the skill's own questions get it.
    answers = engine.ask(command_with_state(text, state), skill.to_questions())
    rule = skill.select_rule(answers, state)
    return RouterHit(skill.id, confidence, rule.actions)
