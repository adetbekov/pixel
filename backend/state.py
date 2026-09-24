"""Robot state: three 0..100 scales plus a face, decayed lazily on read.

There is no background worker. Every read computes how long it has been since
``last_tick_at`` and applies the decay for that span, so a database that sat
untouched for a week still produces a sane (clamped) state on the next request.

Reading never writes. ``last_tick_at`` moves only when an action changes the
scales, which is what makes the decay a function of elapsed time alone: N reads
inside an interval land on exactly the same state as one read at its end, no
matter how often a client polls or how many tabs it has open.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from .actions import Action

START_MOOD = 70.0
START_ENERGY = 80.0
START_FULLNESS = 60.0
START_FACE = "curious"

# Units lost per minute.
DECAY_PER_MIN = {"mood": 0.5, "energy": 0.7, "fullness": 1.0}

LOW_ENERGY = 15.0
LOW_FULLNESS = 15.0

# A hungry robot is a grumpy robot: mood loses this much extra per minute spent
# below LOW_FULLNESS. A rate, not a per-read event — -10 mood over ten hungry
# minutes keeps the old feel without making it depend on the polling interval.
EXTRA_HUNGRY_DECAY_PER_MIN = 1.0

# How each primitive moves the scales.
EFFECTS: dict[str, dict[str, float]] = {
    "jump": {"mood": 5, "energy": -8, "fullness": -3},
    "dance": {"mood": 10, "energy": -15, "fullness": -5},
    "sleep": {"mood": 5, "energy": 40, "fullness": -5},
    "eat": {"mood": 5, "energy": -2, "fullness": 30},
    "spin": {"mood": 4, "energy": -6, "fullness": -2},
    "wave": {"mood": 2, "energy": -2, "fullness": -1},
    "say": {"mood": 0, "energy": -1, "fullness": 0},
    "set_face": {"mood": 0, "energy": 0, "fullness": 0},
}


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


@dataclass
class RobotState:
    mood: float
    energy: float
    fullness: float
    face: str

    def to_dict(self) -> dict:
        return {
            "mood": round(self.mood, 1),
            "energy": round(self.energy, 1),
            "fullness": round(self.fullness, 1),
            "face": self.face,
        }


def _hungry_minutes(fullness: float, dt_min: float) -> float:
    """How much of the next ``dt_min`` minutes is spent below LOW_FULLNESS.

    Fullness falls linearly, so the crossing point is exact arithmetic rather
    than a sampled guess — that exactness is what makes the hunger penalty
    additive over any split of the same span.
    """
    if fullness < LOW_FULLNESS:
        return dt_min
    rate = DECAY_PER_MIN["fullness"]
    if rate <= 0:
        return 0.0
    return max(0.0, dt_min - (fullness - LOW_FULLNESS) / rate)


def decay(state: RobotState, dt_min: float) -> RobotState:
    """Apply ``dt_min`` minutes of decay. Never moves the scales upwards.

    Additive by construction: ``decay(s, a)`` then ``decay(s, b)`` leaves the
    same state as ``decay(s, a + b)``.
    """
    dt_min = max(0.0, dt_min)
    hungry_min = _hungry_minutes(state.fullness, dt_min)
    mood_loss = DECAY_PER_MIN["mood"] * dt_min + EXTRA_HUNGRY_DECAY_PER_MIN * hungry_min
    state.mood = clamp(state.mood - mood_loss)
    state.energy = clamp(state.energy - DECAY_PER_MIN["energy"] * dt_min)
    state.fullness = clamp(state.fullness - DECAY_PER_MIN["fullness"] * dt_min)

    if state.energy < LOW_ENERGY:
        state.face = "sleepy"
    return state


def _row_to_state(row: sqlite3.Row) -> RobotState:
    return RobotState(row["mood"], row["energy"], row["fullness"], row["face"])


def read_state(conn: sqlite3.Connection, now: datetime | None = None) -> RobotState:
    """Read the state, applying the decay owed since ``last_tick_at``.

    Pure: a function of the stored row and ``now``, with no write of its own.
    ``GET /api/state`` must not be observable, or the decay speed would follow
    the client's polling interval instead of the clock.
    """
    now = now or utcnow()
    row = conn.execute("SELECT * FROM robot_state WHERE id = 1").fetchone()
    state = _row_to_state(row)
    dt_min = (now - parse_iso(row["last_tick_at"])).total_seconds() / 60.0
    return decay(state, dt_min)


def write_state(conn: sqlite3.Connection, state: RobotState, now: datetime) -> None:
    conn.execute(
        "UPDATE robot_state SET mood = ?, energy = ?, fullness = ?, face = ?, last_tick_at = ?"
        " WHERE id = 1",
        (state.mood, state.energy, state.fullness, state.face, iso(now)),
    )
    conn.commit()


def apply_actions(
    conn: sqlite3.Connection, actions: list[Action], now: datetime | None = None
) -> RobotState:
    """Decay first, then apply every action's effect, then persist."""
    now = now or utcnow()
    state = read_state(conn, now)
    for action in actions:
        effect = EFFECTS[action.action]
        state.mood = clamp(state.mood + effect["mood"])
        state.energy = clamp(state.energy + effect["energy"])
        state.fullness = clamp(state.fullness + effect["fullness"])
        if action.action == "set_face":
            state.face = action.args["face"]
    write_state(conn, state, now)
    return state
