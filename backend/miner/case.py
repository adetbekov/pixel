"""One row of ``teacher_log``, read back as something the miner can use.

``teacher_log`` stores the command and the robot's state inside ``state_json``
(see :func:`backend.teacher.log.log_case`), so reading the pool is a parse, not
a plain SELECT. Anything that does not parse is skipped and logged rather than
taking the whole run down — the pool is untrusted input by construction, since
half of it was written by a language model.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..state import START_ENERGY, START_FACE, START_FULLNESS, START_MOOD, RobotState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    """One command the router missed, plus the plan the teacher answered with."""

    id: int
    user_text: str
    state: RobotState
    actions: list[dict[str, Any]]

    @property
    def action_names(self) -> set[str]:
        """What the teacher did, ignoring order and arguments.

        Two plans that say different sentences are still the same behaviour, and
        the teacher never phrases a reply the same way twice — so the backtest
        compares these sets and not the plans.
        """
        return {step["action"] for step in self.actions if isinstance(step, dict)}


def _state(payload: dict[str, Any]) -> RobotState:
    raw = payload.get("state")
    raw = raw if isinstance(raw, dict) else {}
    return RobotState(
        mood=float(raw.get("mood", START_MOOD)),
        energy=float(raw.get("energy", START_ENERGY)),
        fullness=float(raw.get("fullness", START_FULLNESS)),
        face=str(raw.get("face", START_FACE)),
    )


def _parse(row: sqlite3.Row) -> Case | None:
    try:
        payload = json.loads(row["state_json"] or "{}")
        actions = json.loads(row["actions_json"] or "[]")
    except (TypeError, ValueError):
        log.warning("teacher_log row %s is not readable JSON", row["id"])
        return None
    if not isinstance(payload, dict) or not isinstance(actions, list):
        return None

    # A failed teacher call is logged too, and it is a useful signal — but its
    # plan is FALLBACK_PLAN, i.e. "curious face, I did not understand". Mining a
    # skill out of those would teach Pixel to not understand on purpose.
    if payload.get("error"):
        return None

    text = str(payload.get("user_text") or "").strip()
    if not text or not actions:
        return None
    return Case(id=int(row["id"]), user_text=text, state=_state(payload), actions=actions)


def load_pool(conn: sqlite3.Connection) -> list[Case]:
    """Every case still waiting to be mined, oldest first."""
    rows = conn.execute(
        "SELECT id, state_json, actions_json FROM teacher_log WHERE mined = 0 ORDER BY id"
    ).fetchall()
    return [case for case in (_parse(row) for row in rows) if case is not None]


def pool_size(conn: sqlite3.Connection) -> int:
    """How many rows are unmined — the counter the automatic trigger watches."""
    return int(
        conn.execute("SELECT COUNT(*) AS n FROM teacher_log WHERE mined = 0").fetchone()["n"]
    )
