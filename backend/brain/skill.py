"""The skill format: a JSON template, not code.

A skill never adds a capability — it only *combines* the fixed action library.
That is the whole safety story: a mined skill, like a hand-written one, is a
declaration the loader can check, and a skill naming an action outside
``backend/actions.py`` is refused at load time instead of half-executing.

A skill is:

* ``questions`` — what the engine should be asked once this skill is picked;
* ``rules``     — ``when -> actions``, first match wins, last rule is the
  unconditional fallback;
* ``threshold`` — the router confidence this skill needs to fire.

``when`` keys are either a question name or one of the state scales
``mood | energy | fullness``; state values are ``low | mid | high``
(see :mod:`backend.brain.verbalize`).
"""

from __future__ import annotations

import json
import logging
import operator
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from ..actions import InvalidPlan, validate_plan
from ..state import RobotState, iso, utcnow
from .engine import Answer, ChoiceResult, NoulResult, ScoreResult
from .verbalize import band

log = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent / "seed_skills"

STATE_KEYS = ("mood", "energy", "fullness")
STATE_VALUES = ("low", "mid", "high")

#: A ``noul`` answer counts as "yes" from here up.
NOUL_TRUE = 0.5

#: Longest comparison operator first, so ``>=`` is not read as ``>``.
SCORE_OPS = (
    (">=", operator.ge),
    ("<=", operator.le),
    ("==", operator.eq),
    (">", operator.gt),
    ("<", operator.lt),
)


class Question(BaseModel):
    type: Literal["choice", "score", "noul"]
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _criteria_match_type(self) -> Question:
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("a choice question needs criteria as an object")
        if self.type == "score" and not (
            isinstance(self.criteria, list) and len(self.criteria) > 1
        ):
            raise ValueError("a score question needs criteria as a list of two or more levels")
        return self

    def to_engine(self) -> dict[str, Any]:
        question: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            question["criteria"] = self.criteria
        return question


class Rule(BaseModel):
    when: dict[str, str] = Field(default_factory=dict)
    actions: list[dict[str, Any]]

    @model_validator(mode="after")
    def _actions_are_library_actions(self) -> Rule:
        try:
            cleaned = validate_plan(self.actions)
        except InvalidPlan as exc:
            raise ValueError(str(exc)) from exc
        if len(cleaned) != len(self.actions):
            raise ValueError("rule uses an action outside the library")
        return self


class Skill(BaseModel):
    id: str
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    questions: dict[str, Question] = Field(default_factory=dict)
    rules: list[Rule]
    threshold: float | None = None
    status: Literal["active", "disabled"] = "active"
    origin: str = "seed"

    @model_validator(mode="after")
    def _rules_are_usable(self) -> Skill:
        if not self.rules:
            raise ValueError("a skill needs at least one rule")
        if self.rules[-1].when:
            raise ValueError("the last rule must be the unconditional fallback (empty `when`)")
        for rule in self.rules:
            for key, expected in rule.when.items():
                self._check_condition(key, expected)
        return self

    def _check_condition(self, key: str, expected: str) -> None:
        if key in STATE_KEYS:
            if expected not in STATE_VALUES:
                raise ValueError(f"`{key}` must be one of {STATE_VALUES}, got {expected!r}")
            return
        question = self.questions.get(key)
        if question is None:
            raise ValueError(f"`when` names `{key}`, which is neither a question nor a state scale")
        if question.type == "noul" and expected not in ("yes", "no"):
            raise ValueError(f"`{key}` is a noul question, so it takes yes/no, got {expected!r}")
        if question.type == "score":
            _parse_score_condition(expected)

    def to_questions(self) -> dict[str, dict[str, Any]]:
        """The whole question set, ready for one batched :meth:`DecisionEngine.ask`."""
        return {name: question.to_engine() for name, question in self.questions.items()}

    def select_rule(self, answers: dict[str, Answer], state: RobotState) -> Rule:
        """First rule whose every condition holds. The fallback always does."""
        for rule in self.rules:
            if all(
                _condition_holds(key, expected, answers, state)
                for key, expected in rule.when.items()
            ):
                return rule
        return self.rules[-1]


def _parse_score_condition(expected: str) -> tuple[Any, float]:
    for token, compare in SCORE_OPS:
        if expected.startswith(token):
            return compare, float(expected[len(token) :])
    # A bare number compares against the rounded level, so `"1"` means level 1.
    return operator.eq, float(expected)


def _condition_holds(
    key: str, expected: str, answers: dict[str, Answer], state: RobotState
) -> bool:
    if key in STATE_KEYS:
        return band(getattr(state, key)) == expected

    answer = answers.get(key)
    if answer is None:
        # The engine skipped the question — treat the condition as unmet rather
        # than firing a rule on an answer nobody gave.
        return False
    if isinstance(answer, NoulResult):
        return (answer.probability >= NOUL_TRUE) == (expected == "yes")
    if isinstance(answer, ScoreResult):
        compare, level = _parse_score_condition(expected)
        value = round(answer.score) if compare is operator.eq else answer.score
        return bool(compare(value, level))
    if isinstance(answer, ChoiceResult):
        return answer.label == expected
    return False


def parse_skill(payload: dict[str, Any]) -> Skill | None:
    """Build a skill, or log a WARNING and return ``None`` if it is unusable.

    Refusing one bad skill must never take the app down with it — a mined skill
    that names ``fly`` is dropped, and Pixel keeps serving the other skills.
    """
    try:
        return Skill.model_validate(payload)
    except ValidationError as exc:
        name = payload.get("id", "<no id>") if isinstance(payload, dict) else "<not an object>"
        log.warning("skill %s rejected: %s", name, exc)
        return None


def load_seed_skills() -> list[Skill]:
    """The starter skills, in filename order — and the order is load-bearing.

    The option order of the router's ``choice`` question changes its answer:
    sweeping all 24 orderings of the four starter skills on a 15-command probe
    moved the hit rate between 7/10 and 9/10, and the numeric filename prefixes
    pin the best one (greet, feed, play, sleep). Keep them when adding a skill.
    """
    skills = []
    for path in sorted(SEED_DIR.glob("*.json")):
        skill = parse_skill(json.loads(path.read_text(encoding="utf-8")))
        if skill is not None:
            skills.append(skill)
    return skills


def seed_db(conn: sqlite3.Connection) -> int:
    """Fill an empty ``skills`` table with the hand-written starter skills."""
    if conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]:
        return 0
    skills = load_seed_skills()
    now = iso(utcnow())
    conn.executemany(
        "INSERT INTO skills (id, json, status, origin, created_at) VALUES (?, ?, ?, ?, ?)",
        [(s.id, s.model_dump_json(), s.status, s.origin, now) for s in skills],
    )
    conn.commit()
    return len(skills)


def load_skills(conn: sqlite3.Connection, *, only_active: bool = False) -> list[Skill]:
    """Read the library back out, dropping (and logging) anything unusable.

    ``status`` is taken from the row, not from the stored JSON: stage 5 disables
    a skill by updating the column, and the row is the authority.

    Ordered by ``rowid`` — insertion order — because the router's option order
    matters (see :func:`load_seed_skills`) and every seeded row shares one
    ``created_at``. Stage 4's mined skills therefore append at the end.
    """
    where = " WHERE status = 'active'" if only_active else ""
    rows = conn.execute(f"SELECT json, status FROM skills{where} ORDER BY rowid").fetchall()
    skills = ((parse_skill(json.loads(row["json"])), row["status"]) for row in rows)
    return [skill.model_copy(update={"status": status}) for skill, status in skills if skill]
