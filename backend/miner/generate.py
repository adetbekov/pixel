"""Asking Gemini for the skill a cluster is asking for.

This is the offline half of the teacher/miner split. The teacher answers a user
who is waiting, so it runs on the cheapest, fastest tier. Nobody waits on this
call: it runs after the fact, once per cluster, and what it produces is a
*schema* that will route thousands of later commands. Quality beats latency, so
it uses a bigger model — ``gemini-3.8-flash`` ($0.75 / $3.75 per 1M, GA since
2026-09-02), overridable with ``GEMINI_MINER_MODEL``.

Like the teacher, this never raises at the caller: a bad draft costs one retry
and then the cluster is left in the pool for the next run.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

from pydantic import ValidationError

from ..actions import ACTIONS, MAX_PLAN_LEN
from ..brain.skill import STATE_KEYS, STATE_VALUES, Skill

# The library description the teacher is handed, verbatim. Two prompts listing
# the same eight primitives would drift apart on the very next stage, and this
# one is the more dangerous place to drift: what the teacher gets wrong costs one
# reply, what the miner gets wrong is baked into a skill.
from ..teacher.prompt import ACTION_LIST
from .case import Case
from .schema import MAX_DESCRIPTION_LEN, MAX_RULES, Grouping, SkillDraft

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"

TIMEOUT_S = 30.0

#: One first try plus one retry, then the cluster waits for the next run.
MAX_ATTEMPTS = 2

EXAMPLE_SKILL = json.dumps(
    {
        "id": "show_trick",
        "name": "Фокус",
        "description": "показать фокус",
        "examples": ["покажи фокус", "сделай фокус", "удиви фокусом"],
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
    },
    ensure_ascii=False,
    indent=2,
)

SYSTEM_PROMPT = f"""Ты — конструктор навыков для робота-питомца Pixel.

Тебе дают группу похожих команд, которые быстрый распознаватель робота не понял, \
и планы действий, которыми на них ответил учитель. Твоя задача — придумать ОДИН навык, \
который покроет всю группу, чтобы робот дальше отвечал на такие команды сам.

Навык — это не код. Это шаблон, который комбинирует РОВНО эти {len(ACTIONS)} действий:
{ACTION_LIST}

Правила навыка:
- `id` — латиницей в snake_case (например `show_trick`), это ключ и метка варианта;
- `name` — короткое название по-русски;
- `description` — КОРОТКАЯ именная группа по-русски, не длиннее {MAX_DESCRIPTION_LEN} символов \
(«показать фокус», «покрутиться»). Это измеренное требование: описание — это текст варианта \
для распознавателя, и длинное описание роняет точность маршрутизации ВСЕХ навыков. \
Не предложение, не инструкция, без примеров фраз;
- `examples` — все команды группы, слово в слово;
- `rules` — не больше {MAX_RULES} правил, первое подходящее выигрывает;
- `when_state` — одно из {", ".join(STATE_KEYS)} (или "" ), `when_band` — одно из \
{", ".join(STATE_VALUES)} (или ""). Никаких других условий не бывает;
- ПОСЛЕДНЕЕ правило обязано быть безусловным: `when_state` и `when_band` пустые;
- максимум {MAX_PLAN_LEN} действий в правиле, и хотя бы одно `say`, чтобы робот ответил;
- реплики — от первого лица, как питомец, а не как ассистент.

Пример готового навыка:
{EXAMPLE_SKILL}

Ответ верни строго в заданной JSON-схеме."""

GROUP_SYSTEM_PROMPT = """Ты группируешь команды пользователя по смыслу.

Тебе дают пронумерованный список команд. Верни группы номеров: в одной группе — команды, \
которые просят у робота одно и то же. Каждый номер ровно в одной группе. \
Команду, похожую на которую в списке нет, положи в группу из одного номера."""


class SkillGenerator(Protocol):
    def propose(self, cases: list[Case], skills: list[Skill]) -> Skill | None: ...

    def group(self, texts: list[str]) -> list[list[int]]: ...


def _plan_words(actions: list[dict[str, Any]]) -> str:
    parts = []
    for step in actions:
        args = step.get("args") or {}
        detail = args.get("face") or args.get("text")
        parts.append(f"{step.get('action')}({detail})" if detail else f"{step.get('action')}()")
    return ", ".join(parts)


def build_input(cases: list[Case], skills: list[Skill]) -> str:
    """The cluster, as commands plus what the teacher actually did about them."""
    lines = "\n".join(
        f'{index}. "{case.user_text}" -> {_plan_words(case.actions)}'
        for index, case in enumerate(cases, start=1)
    )
    known = ", ".join(f"{skill.id} ({skill.description})" for skill in skills)
    covered = (
        f"У робота уже есть навыки: {known}.\n"
        "Новый навык обязан отличаться от них — иначе он перетянет на себя чужие команды.\n\n"
        if skills
        else ""
    )
    return f"{covered}Группа команд и ответы учителя:\n{lines}"


def retry_hint(reason: str) -> str:
    return (
        "\n\nПредыдущий ответ не прошёл проверку "
        f"({reason}). Исправь это и верни валидный JSON по схеме."
    )


class GeminiSkillGenerator:
    """The real generator. ``client`` is injectable so CI needs no key."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or os.environ.get("GEMINI_MINER_MODEL", DEFAULT_MODEL)

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Lazy, exactly as in the teacher: no key must mean no SDK import.
            from google import genai

            self._client = genai.Client()
        return self._client

    def _call(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        interaction = self._ensure_client().interactions.create(
            model=self._model,
            input=prompt,
            system_instruction=system,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
            timeout=TIMEOUT_S,
        )
        return interaction.output_text or ""

    def propose(self, cases: list[Case], skills: list[Skill]) -> Skill | None:
        prompt = build_input(cases, skills)
        reason = "the generator produced nothing"

        for attempt in range(MAX_ATTEMPTS):
            hint = "" if attempt == 0 else retry_hint(reason)
            try:
                raw = self._call(SYSTEM_PROMPT, prompt + hint, SkillDraft.model_json_schema())
            except Exception as exc:  # noqa: BLE001 — a 500 only costs us a retry
                reason = f"{type(exc).__name__}: {exc}"
                log.warning("miner: skill call failed (attempt %d): %s", attempt + 1, reason)
                continue

            skill, reason = _parse(raw)
            if skill is not None:
                return skill
            log.warning("miner: draft rejected (attempt %d): %s", attempt + 1, reason)

        log.info("miner: cluster left in the pool — %s", reason)
        return None

    def group(self, texts: list[str]) -> list[list[int]]:
        listing = "\n".join(f"{index}. {text}" for index, text in enumerate(texts))
        try:
            raw = self._call(GROUP_SYSTEM_PROMPT, listing, Grouping.model_json_schema())
            return Grouping.model_validate_json(raw).groups
        except Exception as exc:  # noqa: BLE001 — no grouping simply means no mining
            log.warning("miner: fallback grouping failed: %s", exc)
            return []


def _parse(raw: str) -> tuple[Skill | None, str]:
    """Draft -> skill, with the reason to retry on when it does not survive."""
    try:
        draft = SkillDraft.model_validate_json(raw)
    except ValidationError as exc:
        return None, f"draft did not match the schema: {exc}"
    try:
        return draft.to_skill(), ""
    except ValidationError as exc:
        # This is where an invented primitive dies: `Skill` -> `Rule` ->
        # `validate_plan`. It is the same gate a hand-written skill passes.
        return None, f"skill did not validate: {exc}"


_generator: SkillGenerator | None = None


def build_generator() -> SkillGenerator | None:
    """The generator, or ``None`` when there is no API key.

    No key means no teacher either, so ``teacher_log`` never fills and there is
    nothing to mine — ``/api/mine`` answers honestly with zero proposals rather
    than failing.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    return GeminiSkillGenerator()


def set_generator(generator: SkillGenerator | None) -> None:
    global _generator
    _generator = generator


def get_generator() -> SkillGenerator | None:
    return _generator
