#!/usr/bin/env python3
"""
bench_router_under_mining.py — what a mining run costs a *concurrent* chat.

Why this exists (JEB-1599). JEB-1543 took the mining run off the request path:
the miss that triggers it no longer waits for it. What that did not change is
`LayaEngine._lock`, so a chat that arrives while a run is in flight still queues
behind whatever the run is holding it for. The question this script answers is
the only one that matters for the user: **what does `POST /api/chat` cost while
the miner is working, against what it costs when the miner is idle** — and,
because the answer is not one number, which part of the run it was waiting on.

Needs the weights and therefore a network on first run — it is a tool, not a
test. CI's miner tests run on `FakeEngine`, where a forward pass is free, so the
number is not measurable there by construction.

    python scripts/bench_router_under_mining.py [--samples 30] [--pool 40]

Three arms, and the third is the one that took two attempts to get right:

1. **miner idle** — the control.
2. **mining, teacher grouping** — the run reaches the backtest, which takes the
   lock once per forward pass.
3. **mining, fallback grouping** — the teacher grouper answers `[]` (a quota
   error, a dead key, three malformed answers; `cluster.py` plans for it), so
   `group_texts` embeds the whole window locally — and `LayaEngine.embed` takes
   the lock **once for every row at once**. The first version of this script had
   no such arm and slept 2 s to "get past the grouping", so it measured the one
   path where the per-pass claim holds and reported it as universal.

What is real and what is not:

* Real: the checkpoint (`LayaEngine`, `multilingual`, cpu), the router, the
  backtest, the fallback grouper, the pool, `db.lock`, and the threading — the
  sampler and the miner are separate threads, exactly as
  `backend/miner/worker.py` runs them.
* Stubbed: Gemini. The drafter and the teacher grouper are local stand-ins —
  both are network calls made with every lock released, so what the stub removes
  is wall-clock the chat does *not* spend waiting. The stub grouper's `[]` is
  arm 3, and that path is entirely real code.
* Not measured: HTTP framing. The sampler calls `backend.api.post_chat`, the
  request body itself; uvicorn and the socket add the same constant to every
  arm.

**What it read** (laya 0.3.10, `multilingual`, cpu, 6 cores, window 40, n=40 per
arm, one run; the box is shared, so treat the chat percentiles as a band and the
lock holds — which do not depend on the sampler — as the firm numbers):

    arm                chat p50   chat p95   chat max   lock wait p50/max
    miner idle           456 ms     832 ms    1162 ms       0 /    0 ms
    mining, teacher      926 ms    1206 ms    1345 ms     254 /  477 ms
    mining, fallback     936 ms    1104 ms    1837 ms     270 / 1045 ms

    inside `predict`: 272 ms p50 idle, 223…224 ms under mining — so the add is
    lock wait, not CPU contention. A chat is 1.5 forward passes here, so the
    per-request add is the wait times that, not one wait.

    what the miner held the lock for, per acquisition:
      backtest forward pass   260 ms p50 / 484 ms max  (n=60)
      whole-pool `embed`     1092 ms p50 / 1339 ms max (n=3, window 40)
                              572 ms p50 /  749 ms max (n=2, window 20)

So there are two different worst cases, not one. On the teacher-grouped path the
chat waits one backtest pass and `MINER_POOL_WINDOW` changes the run's length,
not the wait. On the fallback path the chat can wait the whole-pool embed, and
that hold is linear in the window — 40 rows cost roughly twice what 20 do, which
is the same shape `case.py` quotes from JEB-1509 (430 ms at 20, 2473 ms at 100).

Why the run was still left alone: 1.1 s of lock hold happens on a path that
needs Gemini to have failed, it is already bounded by the window, and removing
it means a second resident copy of a 322M-parameter checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import db
from backend.api import ChatIn, post_chat

# `_to_answer` is private, and this is the one caller outside the module: the
# instrumented `ask` below is a copy of `LayaEngine.ask` with the lock unrolled.
from backend.brain.engine import LayaEngine, _to_answer, set_engine
from backend.brain.skill import Skill, load_skills, seed_db
from backend.miner import mine_once, set_generator
from backend.miner.case import Case
from backend.state import RobotState

#: Commands the sampler sends. None of them is any skill's `examples[0]`, so
#: each one costs the full router — pass 1 (`choice` over the library) and
#: pass 2 (the skill's own questions) — which is what a real chat that lands on
#: a skill costs. An exact-match command would cost step 0 and nothing else and
#: would measure the lookup, not the engine.
CHAT_PHRASES = [
    "покорми робота пожалуйста",
    "давай поиграем немного",
    "привет как дела",
    "иди поспи уже",
]

#: The pool. Five intents worth drafting (three phrasings each) plus one-off
#: noise, which is what the backtest's over-broad check pays a forward pass per
#: row for — the term that scales with `MINER_POOL_WINDOW`.
INTENTS = {
    "trick": ["покажи фокус", "сделай фокус", "а фокус умеешь?"],
    "salto": ["сделай сальто", "покажи сальто", "а сальто можешь?"],
    "song": ["спой песню", "спой что-нибудь", "песню спой"],
    "story": ["расскажи историю", "расскажи сказку", "историю расскажи"],
    "count": ["посчитай до десяти", "посчитай вслух", "сосчитай до пяти"],
}

NOISE = [
    "расскажи про квантовую физику",
    "который час",
    "закажи пиццу",
    "какая погода завтра",
    "переведи на английский",
    "сколько будет два плюс два",
    "включи музыку",
    "напомни позвонить маме",
    "что такое гравитация",
    "найди рецепт борща",
    "позвони бабушке",
    "открой окно",
    "сколько стоит биткоин",
    "как дела на бирже",
    "покажи новости",
    "поставь будильник",
    "сколько километров до луны",
    "кто написал войну и мир",
    "выключи свет",
    "какой сегодня день недели",
    "сколько лететь до марса",
    "прочитай почту",
    "где мои ключи",
    "запусти таймер",
    "как приготовить омлет",
]

TEACHER_PLAN = [
    {"action": "jump", "args": {}},
    {"action": "set_face", "args": {"face": "happy"}},
    {"action": "say", "args": {"text": "Готово!"}},
]


class InstrumentedEngine(LayaEngine):
    """`LayaEngine`, with the lock split out of the forward pass.

    Same body as `LayaEngine.ask`, with `with self._lock` unrolled so the time
    spent *waiting* for the lock and the time spent *inside* `predict` are two
    numbers instead of one. Only the sampling thread's passes are recorded —
    the miner's own are the thing being waited on, not the measurement.

    `embed` is deliberately left alone: it takes the same lock once for the
    whole batch, and that is the hold this benchmark's fallback arm exists to
    measure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float] = []
        self.inside: list[float] = []
        #: How long the *miner* thread holds the lock, per acquisition, split by
        #: what it was doing. The box is shared, so chat percentiles carry the
        #: neighbours' load; these two do not depend on the sampler at all and
        #: are the load-robust half of the answer.
        self.pass_holds: list[float] = []
        self.embed_holds: list[float] = []

    def ask(self, state, questions):
        if not questions:
            return {}
        queued = time.perf_counter()
        self._lock.acquire()
        entered = time.perf_counter()
        try:
            raw = self._agent.predict(state, questions)
        finally:
            self._lock.release()
        left = time.perf_counter()
        if threading.current_thread() is threading.main_thread():
            self.waits.append((entered - queued) * 1000.0)
            self.inside.append((left - entered) * 1000.0)
        else:
            self.pass_holds.append((left - entered) * 1000.0)
        return {name: _to_answer(answer) for name, answer in raw["answers"].items()}

    def embed(self, texts):
        # One acquisition for the whole batch — that is the point being
        # measured, so the call is timed end to end rather than unrolled.
        started = time.perf_counter()
        vectors = super().embed(texts)
        self.embed_holds.append((time.perf_counter() - started) * 1000.0)
        return vectors


class StubGenerator:
    """Gemini, replaced by something local and deterministic.

    Neither method holds `LayaEngine._lock` in production — `group` is one
    `models.generate_content` and `propose` is one more, both made with every
    lock released — so replacing them changes what the *run* takes and not what
    a concurrent chat waits for.

    `grouping="fallback"` is the exception, and it is a *path* the production
    code plans for, not a broken stub: `SkillGenerator.group` swallows its own
    failures and answers `[]`, and `backend.miner.cluster.group_texts` then
    embeds the whole window locally — one `engine.embed` call holding the lock
    for every row at once. That is the arm where the window is a latency knob.
    """

    def __init__(self, labels: list[str], grouping: str = "teacher") -> None:
        self._labels = labels
        self._grouping = grouping

    def group(self, texts: list[str]) -> list[list[int]]:
        if self._grouping == "fallback":
            # Exactly what a quota error, a dead key or three malformed answers
            # produce. `group_texts` reads it as "the teacher cannot" and goes
            # to the local vectors.
            return []
        groups: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            groups.setdefault(
                self._labels[index] if index < len(self._labels) else text, []
            ).append(index)
        return [members for members in groups.values() if len(members) > 1]

    def propose(self, cases: list[Case], skills: list[Skill]) -> Skill | None:
        # Shaped like a real draft: the cluster copied into `examples` word for
        # word (that is what the real prompt asks for) and no questions, so the
        # backtest pays one pass per case and not two.
        phrases = [case.user_text for case in cases]
        return Skill(
            id=f"bench_{abs(hash(tuple(sorted(phrases)))) % 10**8}",
            name="bench",
            description="выполнить трюк по просьбе пользователя",
            examples=phrases,
            rules=[{"when": {}, "actions": TEACHER_PLAN}],
            origin="mined",
        )


def seed_pool(conn, size: int) -> list[str]:
    """Fill `teacher_log` with `size` mineable cases. Returns their labels."""
    state = RobotState(mood=70.0, energy=70.0, fullness=70.0, face="neutral").to_dict()
    texts: list[tuple[str, str]] = []
    for label, phrases in INTENTS.items():
        texts.extend((label, phrase) for phrase in phrases)
    texts.extend((f"noise:{phrase}", phrase) for phrase in NOISE)
    texts = texts[:size]

    for label, text in texts:
        payload = {"user_text": text, "state": state, "handled": True, "error": None}
        conn.execute(
            "INSERT INTO teacher_log (interaction_id, state_json, raw_response, actions_json,"
            " mined, cluster_id) VALUES (NULL, ?, '', ?, 0, NULL)",
            (json.dumps(payload, ensure_ascii=False), json.dumps(TEACHER_PLAN)),
        )
    conn.commit()
    return [label for label, _ in texts]


def chat_once(index: int) -> float:
    text = CHAT_PHRASES[index % len(CHAT_PHRASES)]
    started = time.perf_counter()
    post_chat(ChatIn(text=text))
    return (time.perf_counter() - started) * 1000.0


def percentiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return p50, p95, max(ordered)


@dataclass
class Arm:
    """One sampling window: what the chat paid, and how it split."""

    name: str
    chat: list[float]
    waits: list[float]
    inside: list[float]
    pass_holds: list[float]
    embed_holds: list[float]
    runs: int

    @property
    def passes_per_chat(self) -> float:
        return len(self.waits) / len(self.chat) if self.chat else 0.0


def sample(engine: InstrumentedEngine, count: int, tag: str, runs: dict) -> Arm:
    for series in (engine.waits, engine.inside, engine.pass_holds, engine.embed_holds):
        series.clear()
    latencies = []
    for index in range(count):
        latencies.append(chat_once(index))
        print(f"  {tag} {index + 1}/{count}: {latencies[-1]:.0f} ms", flush=True)
    return Arm(
        tag,
        latencies,
        list(engine.waits),
        list(engine.inside),
        list(engine.pass_holds),
        list(engine.embed_holds),
        runs.get("count", 0),
    )


def mining_window(conn, grouping: str):
    """Start a miner thread that keeps a run in flight, and hand back a stopper.

    No settling sleep before sampling: an earlier version slept two seconds to
    "get past the grouping", which is exactly the part the fallback arm is
    about. Sampling starts with the run, so the window covers both the grouping
    and the backtest in the proportion the run itself spends on them.
    """
    stop = threading.Event()
    runs = {"count": 0}
    set_generator(StubGenerator(_LABELS, grouping=grouping))

    def mine_forever() -> None:
        while not stop.is_set():
            mine_once()
            runs["count"] += 1
            # Every run after the first re-reads a pool whose drafted clusters
            # are now proposals, so reset them: the point is a *run in flight*
            # for the whole sampling window, not a realistic mining cadence.
            with db.lock:
                conn.execute("UPDATE teacher_log SET mined = 0, cluster_id = NULL")
                conn.execute("DELETE FROM skill_proposals")
                conn.execute("DELETE FROM mining_attempts")
                conn.execute("DELETE FROM skills WHERE origin = 'mined'")
                conn.commit()

    thread = threading.Thread(target=mine_forever, name="bench-miner", daemon=True)
    thread.start()

    def finish() -> dict:
        stop.set()
        thread.join(timeout=180)
        return runs

    return runs, finish


_LABELS: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30, help="chat calls per arm")
    parser.add_argument("--pool", type=int, default=40, help="mineable cases in the pool")
    parser.add_argument("--db", default="./bench_pixel.db")
    args = parser.parse_args()

    Path(args.db).unlink(missing_ok=True)
    conn = db.init(args.db)
    seed_db(conn)
    global _LABELS
    _LABELS = seed_pool(conn, args.pool)

    print(f"loading the checkpoint (pool={args.pool}, samples={args.samples})…", flush=True)
    started = time.perf_counter()
    engine = InstrumentedEngine()
    set_engine(engine)
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)

    skills = load_skills(conn, only_active=True)
    print(f"active skills: {[skill.id for skill in skills]}", flush=True)

    # Baseline: one forward pass, nothing else running. JEB-1509 read 110 ms.
    warm = [_time_predict(engine) for _ in range(5)]
    print(f"single predict, idle: {statistics.median(warm):.0f} ms", flush=True)

    print("\n[idle] chat with the miner idle", flush=True)
    set_generator(None)
    arms = [sample(engine, args.samples, "idle", {})]

    for grouping, tag in (("teacher", "mining/teacher"), ("fallback", "mining/fallback")):
        print(f"\n[{tag}] chat with a mining run in flight", flush=True)
        runs, finish = mining_window(conn, grouping)
        arms.append(sample(engine, args.samples, tag, runs))
        finish()
        arms[-1].runs = runs["count"]

    print()
    for arm in arms:
        report(arm)

    idle_p50, _, _ = percentiles(arms[0].chat)
    for arm in arms[1:]:
        p50, p95, _ = percentiles(arm.chat)
        print(
            f"{arm.name}: chat pays {p50 / idle_p50:.1f}x at p50 and"
            f" {p95 / idle_p50:.1f}x at p95 against the idle arm"
        )

    set_engine(None)
    set_generator(None)
    db.close()


def _time_predict(engine: LayaEngine) -> float:
    started = time.perf_counter()
    engine.choice("привет", "q", "какой навык", {"a": "первый", "b": "второй"})
    return (time.perf_counter() - started) * 1000.0


def report(arm: Arm) -> None:
    chat = percentiles(arm.chat)
    wait = percentiles(arm.waits) if arm.waits else (0.0, 0.0, 0.0)
    inside = percentiles(arm.inside) if arm.inside else (0.0, 0.0, 0.0)
    print(
        f"{arm.name:>16}: chat p50 {chat[0]:6.0f}  p95 {chat[1]:6.0f}  max {chat[2]:6.0f} ms"
        f" | lock wait p50 {wait[0]:6.0f}  p95 {wait[1]:6.0f}  max {wait[2]:6.0f} ms"
        f" | in predict p50 {inside[0]:5.0f}  p95 {inside[1]:5.0f} ms"
        f" | {arm.passes_per_chat:.1f} passes/chat, runs={arm.runs}, n={len(arm.chat)}"
    )
    # What the miner held the lock for, which is what the wait above is made of.
    if arm.pass_holds:
        holds = percentiles(arm.pass_holds)
        print(
            f"{'':>16}  miner backtest pass holds the lock: p50 {holds[0]:5.0f}"
            f"  p95 {holds[1]:5.0f}  max {holds[2]:5.0f} ms  (n={len(arm.pass_holds)})"
        )
    if arm.embed_holds:
        holds = percentiles(arm.embed_holds)
        print(
            f"{'':>16}  miner whole-pool embed holds the lock: p50 {holds[0]:5.0f}"
            f"  max {holds[2]:5.0f} ms  (n={len(arm.embed_holds)})"
        )


if __name__ == "__main__":
    main()
