"""The shape Gemini must answer in.

``action`` is deliberately a plain string, not a ``Literal`` over the library.
The schema's job is the *form* of the answer; whether ``"hack_nasa"`` is a real
primitive is decided by :func:`backend.actions.validate_plan`, which is the one
check that cannot be talked out of. Making the field an enum here would hide
that second layer behind a schema the model is merely asked to respect.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..actions import MAX_PLAN_LEN, MAX_SAY_LEN


class TeacherAction(BaseModel):
    """One step. Absent optional arguments come back as ``""``, not ``null``:
    structured output handles a plain string far more reliably than a nullable
    union, and an empty string is simply not passed on as an argument."""

    action: str
    text: str = ""
    face: str = ""

    def to_raw(self) -> dict[str, Any]:
        """The ``{"action": ..., "args": {...}}`` shape the rest of the app uses."""
        args: dict[str, Any] = {}
        if self.text:
            args["text"] = self.text
        if self.face:
            args["face"] = self.face
        return {"action": self.action, "args": args}


class TeacherPlan(BaseModel):
    reply: str = Field(max_length=MAX_SAY_LEN)
    actions: list[TeacherAction] = Field(default_factory=list, max_length=MAX_PLAN_LEN)

    @field_validator("reply", mode="before")
    @classmethod
    def _truncate(cls, value: Any) -> Any:
        """Trim an over-long reply instead of failing the whole plan.

        ``max_length`` stays on the field so the model is *told* the limit in the
        JSON schema; a reply that overshoots it by a few characters is a
        cosmetic miss and must not cost the user a second 8-second round trip.
        """
        return value[:MAX_SAY_LEN] if isinstance(value, str) else value

    @field_validator("actions", mode="before")
    @classmethod
    def _cap(cls, value: Any) -> Any:
        return value[:MAX_PLAN_LEN] if isinstance(value, list) else value

    def to_raw_plan(self) -> list[dict[str, Any]]:
        return [action.to_raw() for action in self.actions]
