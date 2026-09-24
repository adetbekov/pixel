#!/usr/bin/env python3
"""
bench_router_under_mining.py — what a mining run costs a *concurrent* chat.

Why this exists (JEB-1599). JEB-1543 took the mining run off the request path:
the miss that triggers it no longer waits for it. What that did not change is
`LayaEngine._lock` — one lock, one forward pass at a time — so a chat that
arrives while a run is in flight still queues behind the backtest's passes. The
question this script answers is the only one that matters for the user: **what
does `POST /api/chat` cost while the miner is working, against what it costs
when the miner is idle.**

Needs the weights and therefore a network on first run — it is a tool, not a
test. CI's miner tests run on `FakeEngine`, where a forward pass is free, so the
number is not measurable there by construction.

    python scripts/bench_router_under_mining.py [--samples 30] [--pool 40]

What is real and what is not:

* Real: the checkpoint (`LayaEngine`, `multilingual`, cpu), the router, the
  backtest, the pool, `db.lock`, and the threading — the sampler and the miner
  are separate threads, exactly as `backend/miner/worker.py` runs them.
* Stubbed: Gemini. The grouper and the drafter are local stand-ins, because
  neither holds `_lock` (both are network calls made with every lock released)
  and a paid call per run would make the benchmark unrepeatable. What that
  removes from the run is wall-clock the chat does *not* spend waiting.
* Not measured: HTTP framing. The sampler calls `backend.api.post_chat`, the
  request body itself; uvicorn and the socket add the same constant to both
  arms.

Output is p50/p95 of the chat latency in two arms — miner idle (control) and
miner running — plus the single-`predict` baseline JEB-1509 reported as 110 ms.

**What it read** (laya 0.3.10, `multilingual`, cpu, 6 cores, pool 40, n=40 per
arm, three runs — the spread is the box, which is shared):

    arm            p50            p95             max
    miner idle     396…470 ms     719…889 ms       853…1367 ms
    miner running  650…827 ms    1020…1183 ms     1227…1416 ms

and, instrumented per forward pass in the same shape, the split between waiting
and computing:

    arm             lock wait p50/p95    inside predict p50/p95
    miner idle          0 /    0 ms          220 / 361 ms
    miner running     245 /  286 ms          233 / 280 ms

So the whole penalty is lock wait, the wait is **one** miner forward pass — the
lock is taken per pass, never for the run — and running the model on a busy box
costs the chat nothing measurable on top (220 ms vs 233 ms inside `predict`).
That is why the run was left as it is: `MINER_POOL_WINDOW` changes how long a
run lasts and not what one chat inside it pays (pool 20 read p50 752 / p95
1079 ms, the same band as pool 40), and the only thing that would remove the
245 ms is a second resident copy of a 322M-parameter checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import db
from backend.api import ChatIn, post_chat
from backend.brain.engine import LayaEngine, set_engine
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


class StubGenerator:
    """Gemini, replaced by something local and deterministic.

    Neither method holds `LayaEngine._lock` in production — `group` is one
    `models.generate_content` and `propose` is one more, both made with every
    lock released — so replacing them changes what the *run* takes and not what
    a concurrent chat waits for.
    """

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def group(self, texts: list[str]) -> list[list[int]]:
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


def sample(count: int, tag: str) -> list[float]:
    latencies = []
    for index in range(count):
        latencies.append(chat_once(index))
        print(f"  {tag} {index + 1}/{count}: {latencies[-1]:.0f} ms", flush=True)
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30, help="chat calls per arm")
    parser.add_argument("--pool", type=int, default=40, help="mineable cases in the pool")
    parser.add_argument("--db", default="./bench_pixel.db")
    args = parser.parse_args()

    Path(args.db).unlink(missing_ok=True)
    conn = db.init(args.db)
    seed_db(conn)
    labels = seed_pool(conn, args.pool)
    set_generator(StubGenerator(labels))

    print(f"loading the checkpoint (pool={args.pool}, samples={args.samples})…", flush=True)
    started = time.perf_counter()
    engine = LayaEngine()
    set_engine(engine)
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)

    skills = load_skills(conn, only_active=True)
    print(f"active skills: {[skill.id for skill in skills]}", flush=True)

    # Baseline: one forward pass, nothing else running. JEB-1509 read 110 ms.
    warm = [_time_predict(engine) for _ in range(5)]
    print(f"single predict, idle: {statistics.median(warm):.0f} ms", flush=True)

    print("\n[control] chat with the miner idle", flush=True)
    control = sample(args.samples, "control")

    print("\n[loaded] chat with a mining run in flight", flush=True)
    stop = threading.Event()
    runs = {"count": 0}

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

    miner = threading.Thread(target=mine_forever, name="bench-miner", daemon=True)
    miner.start()
    time.sleep(2.0)  # let the run get past its grouping and into the backtest
    loaded = sample(args.samples, "loaded")
    stop.set()
    miner.join(timeout=120)

    print(f"\nmining runs completed during the window: {runs['count']}")
    report("miner idle", control)
    report("miner running", loaded)

    control_p50, _, _ = percentiles(control)
    loaded_p50, loaded_p95, _ = percentiles(loaded)
    print(
        f"\nchat pays {loaded_p50 / control_p50:.1f}x at p50 and"
        f" {loaded_p95 / control_p50:.1f}x at p95 while the miner runs"
    )

    set_engine(None)
    set_generator(None)
    db.close()


def _time_predict(engine: LayaEngine) -> float:
    started = time.perf_counter()
    engine.choice("привет", "q", "какой навык", {"a": "первый", "b": "второй"})
    return (time.perf_counter() - started) * 1000.0


def report(name: str, values: list[float]) -> None:
    p50, p95, worst = percentiles(values)
    print(
        f"{name:>14}: p50 {p50:7.0f} ms   p95 {p95:7.0f} ms   max {worst:7.0f} ms   n={len(values)}"
    )


if __name__ == "__main__":
    main()
