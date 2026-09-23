"""The teacher's prompt.

The action list is generated from :data:`backend.actions.ACTIONS` rather than
spelled out, so a primitive added to the library can never silently go missing
from the prompt — the two would drift apart on the very next stage otherwise.
"""

from __future__ import annotations

from ..actions import ACTIONS, FACES, MAX_PLAN_LEN, MAX_SAY_LEN
from ..brain.skill import Skill
from ..brain.verbalize import word
from ..state import RobotState

#: Mirrors ``backend.api.MAX_CHAT_TEXT``, declared here rather than imported
#: because ``backend.api`` imports this package. `/api/chat` already rejects
#: anything longer, so this only matters for a future caller that does not.
#: ``tests/test_teacher.py`` pins the two together.
MAX_TEXT_LEN = 500

#: Keeps the skill list from crowding out the rest of the prompt.
MAX_SKILLS_IN_PROMPT = 40


def _signature(name: str) -> str:
    spec = ACTIONS[name]
    if name == "set_face":
        return f"- set_face(face) — {spec.description}. face: {', '.join(FACES)}"
    if name == "say":
        return f"- say(text) — {spec.description}. text: не длиннее {MAX_SAY_LEN} символов"
    return f"- {name}() — {spec.description}"


ACTION_LIST = "\n".join(_signature(name) for name in ACTIONS)

SYSTEM_PROMPT = f"""Ты — Pixel, виртуальный робот-питомец на веб-странице. У тебя есть характер: \
ты любопытный, дружелюбный и немного упрямый. Ты не ассистент и не чат-бот — ты питомец.

Пользователь дал команду, которую твой быстрый распознаватель не понял. Твоя задача — \
превратить её в план из действий и ответить пользователю одной репликой.

Тебе доступны РОВНО эти {len(ACTIONS)} действия и никакие другие:
{ACTION_LIST}

Жёсткие правила:
- используй ТОЛЬКО действия из списка выше, ничего не выдумывай и не изобретай новых;
- максимум {MAX_PLAN_LEN} действий в плане;
- ты не пишешь код и ничего не исполняешь вне этого списка;
- учитывай своё состояние: уставший — отказывайся прыгать и танцевать, голодный — проси еды, \
грустный — отвечай вяло;
- реплика `reply` — на языке пользователя, от первого лица, не длиннее {MAX_SAY_LEN} символов;
- если команда непонятна или невыполнима — честно скажи об этом и предложи научить тебя иначе.

Ответ верни строго в заданной JSON-схеме."""


def _state_words(state: RobotState) -> str:
    return (
        f"Твоё состояние: настроение {word(state.mood)}, "
        f"энергия {word(state.energy)}, сытость {word(state.fullness)}, "
        f"лицо сейчас: {state.face}."
    )


def _skills_words(skills: list[Skill]) -> str:
    """The skills Pixel already has, as ``id: description``.

    Not the examples — the router already owns those, and what the teacher needs
    is only the *boundary* of what is covered, so stage 4's miner can tell a
    genuinely new behaviour from a near-duplicate of an existing skill.
    """
    if not skills:
        return "Пока ты не умеешь ничего заранее заготовленного."
    lines = "\n".join(f"- {s.id}: {s.description}" for s in skills[:MAX_SKILLS_IN_PROMPT])
    return (
        "Навыки, которые у тебя уже есть:\n"
        f"{lines}\n"
        "Если команда подходит под один из них — всё равно выполни её, но ответь кратко."
    )


def build_input(text: str, state: RobotState, skills: list[Skill]) -> str:
    return (
        f"{_state_words(state)}\n\n"
        f"{_skills_words(skills)}\n\n"
        f'Команда пользователя: "{text[:MAX_TEXT_LEN]}"'
    )


def retry_hint(reason: str) -> str:
    return (
        "\n\nПредыдущий ответ содержал недопустимое действие или был непригоден "
        f"({reason}). Используй ТОЛЬКО действия из списка и верни валидный JSON."
    )
