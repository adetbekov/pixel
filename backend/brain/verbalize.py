"""Turning the robot's numbers into the words a small model can reason about.

Two separate strings, because they are read by two different passes and the
split is measured, not stylistic:

* :func:`command_only` — what pass 1 (which skill?) sees. Appending the robot's
  state to it costs accuracy: on a 6-command probe against the multilingual
  checkpoint the bare command scored 6/6 and the command-plus-state string 4/6,
  because the state words drag the choice towards ``sleep`` and ``feed``.
* :func:`command_with_state` — what pass 2 (how should the skill behave?) sees.
  There the state is the point.

Hard rules ``energy=low -> refuse`` are *not* left to the model: they are skill
rules matched against :func:`band` on the real numbers.
"""

from __future__ import annotations

from ..state import RobotState

LOW_MAX = 30.0
MID_MAX = 70.0

BAND_WORDS = {"low": "низкая", "mid": "средняя", "high": "высокая"}


def band(value: float) -> str:
    """``0-30 -> low``, ``31-70 -> mid``, ``71-100 -> high``."""
    if value <= LOW_MAX:
        return "low"
    if value <= MID_MAX:
        return "mid"
    return "high"


def word(value: float) -> str:
    return BAND_WORDS[band(value)]


def command_only(text: str) -> str:
    return f'Команда пользователя: "{text}"'


def command_with_state(text: str, state: RobotState) -> str:
    return (
        f"{command_only(text)}\n"
        f"Состояние робота: настроение {word(state.mood)}, "
        f"энергия {word(state.energy)}, сытость {word(state.fullness)}."
    )
