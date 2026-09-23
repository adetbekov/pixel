"""The worked example stage 4 is specified against: five ways to ask for a trick.

Shared by the cluster, backtest and proposal-API tests so all three describe the
same run of the pipeline instead of three unrelated toy setups.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.state import RobotState
from backend.teacher.log import log_case

from .fakes import EMBED_DIM

TRICK_COMMANDS = [
    "покажи фокус",
    "сделай фокус",
    "фокус покажи",
    "удиви фокусом",
    "а фокус умеешь?",
]

#: A sixth case for the same cluster, arriving later than the first five.
LATER_TRICK = "ну покажи фокус ещё"

#: What the teacher answered every one of them with. Same three primitives, and
#: a different sentence each time — which is why the backtest compares names.
TEACHER_PLANS = [
    [
        {"action": "spin", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": text}},
    ]
    for text in ("Тада!", "Вот так фокус!", "Смотри!", "Оп-ля!", "Умею, смотри!")
]

#: Nothing extreme, so the candidate's fallback rule is the one that fires.
MID_STATE = RobotState(mood=60.0, energy=60.0, fullness=60.0, face="curious")

#: One direction plus a nudge per command: every pair sits at cos >= 0.97, well
#: clear of MINER_SIM (0.88), and an unscripted command stays far away — below
#: 0.19 through `hash_vector`. Same width as `hash_vector`, so the two can
#: appear in one batch.
TRICK_EMBEDDINGS = {
    text: [1.0, 0.05 * index, *([0.0] * (EMBED_DIM - 2))]
    for index, text in enumerate([*TRICK_COMMANDS, LATER_TRICK])
}

#: Control phrases: `examples[0]` of each starter skill, which is exactly the set
#: the regression check builds.
SEED_CONTROLS = {"привет": "greet", "покорми": "feed", "поиграй": "play", "пора спать": "sleep"}

TRICK_ROUTES = {
    **dict.fromkeys([*TRICK_COMMANDS, LATER_TRICK], "show_trick"),
    **SEED_CONTROLS,
}

TRICK_DRAFT: dict[str, Any] = {
    "id": "show_trick",
    "name": "Фокус",
    "description": "показать фокус",
    "examples": TRICK_COMMANDS,
    "rules": [
        {
            "when_state": "energy",
            "when_band": "low",
            "actions": [
                {"action": "set_face", "face": "sleepy"},
                {"action": "say", "text": "Я слишком устал для фокусов"},
            ],
        },
        {
            "when_state": "",
            "when_band": "",
            "actions": [
                {"action": "spin"},
                {"action": "set_face", "face": "happy"},
                {"action": "say", "text": "Тада! Вот мой фокус."},
            ],
        },
    ],
}


def draft_json(**overrides: Any) -> str:
    return json.dumps({**TRICK_DRAFT, **overrides}, ensure_ascii=False)


def fill_pool(
    conn: sqlite3.Connection,
    commands: list[str] | None = None,
    state: RobotState = MID_STATE,
    error: str | None = None,
    handled: bool = True,
) -> list[int]:
    """Put teacher cases in the pool through the real writer, not a hand-built row."""
    commands = TRICK_COMMANDS if commands is None else commands
    return [
        log_case(
            conn,
            interaction_id=index + 1,
            user_text=command,
            state=state,
            confidence=0.4,
            raw_response="{}",
            actions=TEACHER_PLANS[index % len(TEACHER_PLANS)],
            error=error,
            handled=handled,
        )
        for index, command in enumerate(commands)
    ]
