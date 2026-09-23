"""HTTP contract — frozen at stage 1.

Every later stage (router, teacher, miner, metrics) writes against these shapes.
Fields are only ever added, never renamed. Endpoints that belong to a later
stage already exist here as stubs so the stage-1 frontend is complete.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db
from .actions import Action, validate_plan
from .state import apply_actions, iso, read_state, utcnow

router = APIRouter(prefix="/api")

LOW_ENERGY_FOR_PLAY = 20.0

BUTTON_PLANS: dict[str, list[dict[str, Any]]] = {
    "feed": [
        {"action": "eat", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": "Ням!"}},
    ],
    "pet": [
        {"action": "wave", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": "Приятно!"}},
    ],
    "play": [
        {"action": "jump", "args": {}},
        {"action": "spin", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": "Ура!"}},
    ],
}

# A tired robot refuses to play. Stage 4's mined "play" skill reuses this rule.
TIRED_PLAN: list[dict[str, Any]] = [
    {"action": "set_face", "args": {"face": "sleepy"}},
    {"action": "say", "args": {"text": "Я устал, давай позже"}},
]

CHAT_STUB_REPLY = "Пока я умею только кнопки — скоро научусь понимать слова!"


class StateOut(BaseModel):
    mood: float
    energy: float
    fullness: float
    face: str


class ActionOut(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class Reply(BaseModel):
    interaction_id: int
    reply: str
    actions: list[ActionOut]
    engine: Literal["laya", "gemini", "button"]
    skill_id: str | None
    confidence: float | None
    latency_ms: int
    state: StateOut


class ChatIn(BaseModel):
    text: str


class ActionIn(BaseModel):
    name: Literal["feed", "pet", "play"]


class FeedbackIn(BaseModel):
    interaction_id: int
    value: Literal[1, -1]


class Ok(BaseModel):
    ok: bool = True


class MineOut(BaseModel):
    started: bool = True
    proposals: int = 0


class SkillCard(BaseModel):
    id: str
    name: str
    description: str
    status: str
    origin: str
    uses: int = 0
    likes: int = 0
    dislikes: int = 0


class Proposal(BaseModel):
    id: str
    skill: dict[str, Any]
    match_rate: float
    sample_ids: list[int]
    status: str
    created_at: str


class Metrics(BaseModel):
    laya_share: float
    avg_latency_laya_ms: float
    avg_latency_gemini_ms: float
    skills_active: int
    total_commands: int


def _log_interaction(
    conn,
    *,
    user_text: str,
    engine: str,
    skill_id: str | None,
    confidence: float | None,
    latency_ms: int,
    actions: list[Action],
    reply_text: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO interactions"
        " (ts, user_text, engine, skill_id, confidence, latency_ms, actions_json, reply_text,"
        "  feedback)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (
            iso(utcnow()),
            user_text,
            engine,
            skill_id,
            confidence,
            latency_ms,
            json.dumps([a.to_dict() for a in actions], ensure_ascii=False),
            reply_text,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _reply_text(actions: list[Action], fallback: str) -> str:
    for action in actions:
        if action.action == "say":
            return action.args["text"]
    return fallback


def execute_plan(
    *,
    user_text: str,
    raw_plan: list[dict[str, Any]],
    engine: str,
    started: float,
    skill_id: str | None = None,
    confidence: float | None = None,
    fallback_reply: str = "Готово!",
) -> dict:
    """The single execution path: validate, apply, log, and shape a ``Reply``.

    Stages 2-4 route the router's and the teacher's plans through here too, so
    the frozen response shape is built in exactly one place.
    """
    conn = db.get_conn()
    with db.lock:
        actions = validate_plan(raw_plan)
        state = apply_actions(conn, actions)
        latency_ms = int((time.perf_counter() - started) * 1000)
        reply = _reply_text(actions, fallback_reply)
        interaction_id = _log_interaction(
            conn,
            user_text=user_text,
            engine=engine,
            skill_id=skill_id,
            confidence=confidence,
            latency_ms=latency_ms,
            actions=actions,
            reply_text=reply,
        )
    return {
        "interaction_id": interaction_id,
        "reply": reply,
        "actions": [a.to_dict() for a in actions],
        "engine": engine,
        "skill_id": skill_id,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "state": state.to_dict(),
    }


@router.get("/state", response_model=StateOut)
def get_state() -> dict:
    conn = db.get_conn()
    with db.lock:
        return read_state(conn).to_dict()


@router.post("/action", response_model=Reply)
def post_action(payload: ActionIn) -> dict:
    started = time.perf_counter()
    conn = db.get_conn()
    with db.lock:
        state = read_state(conn)

    raw_plan = BUTTON_PLANS[payload.name]
    if payload.name == "play" and state.energy < LOW_ENERGY_FOR_PLAY:
        raw_plan = TIRED_PLAN

    return execute_plan(
        user_text=payload.name,
        raw_plan=raw_plan,
        engine="button",
        started=started,
    )


@router.post("/chat", response_model=Reply)
def post_chat(payload: ChatIn) -> dict:
    """Stage-1 stub: no router and no teacher yet, so every message gets the same
    polite canned answer. Stages 2 and 3 replace the body, not the shape."""
    started = time.perf_counter()
    return execute_plan(
        user_text=payload.text,
        raw_plan=[
            {"action": "set_face", "args": {"face": "curious"}},
            {"action": "say", "args": {"text": CHAT_STUB_REPLY}},
        ],
        engine="button",
        started=started,
    )


@router.post("/feedback", response_model=Ok)
def post_feedback(payload: FeedbackIn) -> dict:
    conn = db.get_conn()
    with db.lock:
        cursor = conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (payload.value, payload.interaction_id),
        )
        conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="unknown interaction_id")
    return {"ok": True}


@router.get("/skills", response_model=list[SkillCard])
def get_skills() -> list[dict]:
    """Empty until stage 4 mines the first skill."""
    return []


@router.get("/proposals", response_model=list[Proposal])
def get_proposals() -> list[dict]:
    """Empty until stage 4 mines the first proposal."""
    return []


@router.post("/proposals/{proposal_id}/accept", response_model=Ok)
def accept_proposal(proposal_id: str) -> dict:
    return {"ok": True}


@router.post("/proposals/{proposal_id}/reject", response_model=Ok)
def reject_proposal(proposal_id: str) -> dict:
    return {"ok": True}


@router.post("/mine", response_model=MineOut)
def post_mine() -> dict:
    return {"started": True, "proposals": 0}


@router.get("/metrics", response_model=Metrics)
def get_metrics() -> dict:
    conn = db.get_conn()
    with db.lock:
        rows = conn.execute(
            "SELECT engine, COUNT(*) AS n, AVG(latency_ms) AS avg_ms"
            " FROM interactions GROUP BY engine"
        ).fetchall()
        skills_active = conn.execute(
            "SELECT COUNT(*) AS n FROM skills WHERE status = 'active'"
        ).fetchone()["n"]

    by_engine = {row["engine"]: row for row in rows}
    laya = by_engine.get("laya")
    gemini = by_engine.get("gemini")
    laya_n = laya["n"] if laya else 0
    gemini_n = gemini["n"] if gemini else 0
    total = sum(row["n"] for row in rows)
    routed = laya_n + gemini_n

    return {
        # The headline learning metric: share of understood commands Laya handled alone.
        "laya_share": round(laya_n / routed, 3) if routed else 0.0,
        "avg_latency_laya_ms": round(laya["avg_ms"], 1) if laya else 0.0,
        "avg_latency_gemini_ms": round(gemini["avg_ms"], 1) if gemini else 0.0,
        "skills_active": int(skills_active),
        "total_commands": int(total),
    }
