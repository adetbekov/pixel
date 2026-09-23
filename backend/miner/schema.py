"""The shape Gemini must answer a skill in — flat, so structured output holds.

:class:`backend.brain.skill.Skill` is the real format, but it is not the format
to *ask* for. It carries open maps (``questions``, ``Rule.when``, action
``args``), and a schema full of free-form objects is where structured output
degrades first. So the model fills in a flat draft and :meth:`SkillDraft.to_skill`
assembles the real thing — the same split, and for the same reason, as
:mod:`backend.teacher.schema`.

Two deliberate restrictions on what a *mined* skill may be, both narrowing, not
widening — a hand-written skill in ``seed_skills/`` is unaffected:

* **No ``questions``.** A generated question means generated instructions for
  Laya, a second forward pass shaped by them, and a failure mode nobody can see
  from the proposal card. A mined skill combines the library and branches on the
  robot's own state; that already covers "tired robot refuses to dance".
* **``when`` over the state scales only.** Which is all that is left once the
  questions are gone.

Validation is not spread out: ``action`` stays a plain string here so that
:func:`backend.actions.validate_plan` — reached through ``Rule`` inside
``Skill`` — remains the single check that decides what Pixel can do.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..actions import MAX_PLAN_LEN, MAX_SAY_LEN
from ..brain.skill import STATE_KEYS, STATE_VALUES, Skill

#: Stage 2 measured this: the router's `choice` head sees `description` as the
#: option text, and long descriptions dropped routing from 6/6 to 2/6. The
#: starter skills sit at 11-20 characters. `MAX_DESCRIPTION_LEN` in the router
#: truncates at 120, but truncation only prevents overflow — it cannot rescue a
#: sentence-shaped description. A candidate over this is rejected, not trimmed.
MAX_DESCRIPTION_LEN = 60

#: Enough for "state exception, state exception, default" and no more.
MAX_RULES = 4

MAX_EXAMPLES = 12

#: Latin snake_case, because the id is a router option label and a primary key.
ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,31}")


class DraftAction(BaseModel):
    """One step. Optional arguments come back as ``""``, never ``null`` —
    structured output handles a plain string far better than a nullable union."""

    action: str
    text: str = ""
    face: str = ""

    def to_raw(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.text:
            args["text"] = self.text[:MAX_SAY_LEN]
        if self.face:
            args["face"] = self.face
        return {"action": self.action, "args": args}


class DraftRule(BaseModel):
    """``when_state``/``when_band`` empty means the unconditional fallback."""

    when_state: str = ""
    when_band: str = ""
    actions: list[DraftAction] = Field(default_factory=list, max_length=MAX_PLAN_LEN)

    @property
    def is_fallback(self) -> bool:
        return not (self.when_state and self.when_band)

    def to_raw(self) -> dict[str, Any]:
        when = {} if self.is_fallback else {self.when_state: self.when_band}
        return {"when": when, "actions": [action.to_raw() for action in self.actions]}


class SkillDraft(BaseModel):
    id: str
    name: str
    description: str = Field(max_length=MAX_DESCRIPTION_LEN)
    examples: list[str] = Field(default_factory=list, max_length=MAX_EXAMPLES)
    rules: list[DraftRule] = Field(default_factory=list, max_length=MAX_RULES)

    @field_validator("id")
    @classmethod
    def _id_is_a_label(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(f"id must be latin snake_case, got {value!r}")
        return value

    @field_validator("description")
    @classmethod
    def _description_is_a_noun_phrase(cls, value: str) -> str:
        # Not truncated on purpose: a description that needs trimming is a
        # description that will route badly, and a bad option label poisons the
        # choice for every *other* skill too.
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("description must not be empty")
        return cleaned

    @field_validator("rules")
    @classmethod
    def _rules_name_real_scales(cls, value: list[DraftRule]) -> list[DraftRule]:
        for rule in value:
            if rule.is_fallback:
                continue
            if rule.when_state not in STATE_KEYS:
                raise ValueError(f"when_state must be one of {STATE_KEYS}, got {rule.when_state!r}")
            if rule.when_band not in STATE_VALUES:
                raise ValueError(f"when_band must be one of {STATE_VALUES}, got {rule.when_band!r}")
        return value

    def to_skill(self) -> Skill:
        """The real skill. Raises ``ValidationError`` if the draft is unusable.

        ``Skill`` is where the library check happens: its ``Rule`` validator runs
        every action through ``validate_plan`` and refuses a rule that names
        anything outside :data:`backend.actions.ACTIONS`. It also enforces the
        empty-``when`` last rule, so this method does not repeat either check.
        """
        return Skill(
            id=self.id,
            name=self.name,
            description=self.description,
            examples=self.examples,
            questions={},
            rules=[rule.to_raw() for rule in self.rules],
            status="active",
            origin="mined",
        )


class Grouping(BaseModel):
    """The fallback clustering answer: groups of indices into the command list."""

    groups: list[list[int]] = Field(default_factory=list)
