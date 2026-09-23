"""``teacher_log`` — the miner's only raw material.

Stage 4 groups these rows into clusters and asks Gemini for a skill that covers
each one. A row missing its state or its plan is a sample the miner cannot use,
so every teacher call writes a complete row — the fallback ones included, since
"the teacher could not answer this either" is itself a signal.

The table has no timestamp of its own; ``interaction_id`` joins to
``interactions.ts`` when one is needed (see ``/api/metrics``).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..state import RobotState


def log_case(
    conn: sqlite3.Connection,
    *,
    interaction_id: int,
    user_text: str,
    state: RobotState,
    confidence: float | None,
    raw_response: str,
    actions: list[dict[str, Any]],
    error: str | None = None,
) -> int:
    """Record one teacher call. ``actions`` is the plan *after* validation."""
    payload = {
        "user_text": user_text,
        "state": state.to_dict(),
        # The router's confidence in the miss: how close Laya came to handling
        # this alone, which is what tells the miner a skill is worth mining.
        "router_confidence": confidence,
        "error": error,
    }
    cursor = conn.execute(
        "INSERT INTO teacher_log"
        " (interaction_id, state_json, raw_response, actions_json, mined, cluster_id)"
        " VALUES (?, ?, ?, ?, 0, NULL)",
        (
            interaction_id,
            json.dumps(payload, ensure_ascii=False),
            raw_response,
            json.dumps(actions, ensure_ascii=False),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)
