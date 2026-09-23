#!/usr/bin/env python3
"""
calibrate_miner_sim.py — measure MINER_SIM against the real Laya vectors, and
time what `embed` costs the request it runs inside.

Why this exists (JEB-1509). `MINER_SIM=0.75` was picked before anyone had run
the checkpoint, and the docstring that carried it warned against going *above*
0.85. Both were wrong in the same direction: mean-pooled encoder states are
anisotropic, so every cosine comes out high and the band that separates "same
request" from "different request" sits near the top of the range, not in the
middle. At 0.75 single-link joins almost the whole pool into one component, so
the miner spends a Gemini call per run drafting a skill out of unrelated
commands. This script is what produced the replacement default, and it is here
so the next person changing that number measures instead of guessing.

Needs the weights and therefore a network on first run — it is a tool, not a
test. CI covers the clustering itself with scripted vectors (`tests/fakes.py`).

    python scripts/calibrate_miner_sim.py

Four sections, in the order the argument is made:

  1. **Cosines.** Pairwise similarity on a labelled probe of Russian commands,
     split into within-intent and across-intent. The two distributions overlap;
     everything after this is about where they overlap least.
  2. **Sweep.** For each candidate threshold, `components()` run over simulated
     pools drawn from repeated intents plus one-off noise, scored on what mining
     actually costs: `purity` (share of mineable clusters holding one intent —
     the rest are paid Gemini calls on unrelated commands), `recall` (share of
     repeated intents that came out as a cluster), `calls/run`.
  3. **Centering.** The usual anisotropy fix, and why it is not used: it wins on
     a large mixed probe and shatters a small pool that is genuinely one intent,
     which is the case the miner exists for.
  4. **Latency.** `mine_once()` runs inline in `POST /api/chat`, and the pool it
     embeds only shrinks for cases that became a proposal — so this grows.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.miner.cluster import DEFAULT_MIN_CLUSTER, components, cosine_matrix

BATCH = 5
THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(26)]

#: Commands that recur — the thing the miner exists to catch. Several phrasings
#: each, because one phrasing repeated verbatim is not the interesting case.
INTENTS: dict[str, list[str]] = {
    "фокус": [
        "покажи фокус",
        "сделай фокус",
        "удиви фокусом",
        "а фокус показать можешь?",
        "фокус покажи пожалуйста",
        "хочу увидеть фокус",
    ],
    "песня": ["спой песню", "спой что-нибудь", "давай ты споёшь", "спой мне", "исполни песенку"],
    "шутка": [
        "расскажи анекдот",
        "пошути",
        "расскажи шутку",
        "пошути что-нибудь",
        "знаешь анекдоты?",
    ],
    "погода": ["какая сегодня погода", "погода на улице", "что там с погодой", "на улице тепло?"],
    "счёт": [
        "посчитай до десяти",
        "сосчитай от одного до десяти",
        "считай вслух до 10",
        "досчитай до десяти",
    ],
}

#: One-off commands nobody repeats. Most of a real pool is this, which is why a
#: threshold that looks fine on a probe of pure clusters can still be useless.
NOISE: list[str] = [
    "привет",
    "расскажи про квантовую физику",
    "как тебя зовут",
    "выключи свет",
    "сколько тебе лет",
    "открой окно",
    "переведи слово кот на английский",
    "что ты умеешь",
    "спокойной ночи",
    "который час",
    "ты молодец",
    "ты меня достал",
    "прыгни повыше",
    "сколько будет два плюс два",
    "кто тебя сделал",
    "хочу спать",
    "нарисуй кота",
    "где мои ключи",
    "включи музыку погромче",
    "закажи пиццу",
]


def corpus() -> tuple[list[str], list[str]]:
    """Texts and their gold labels; every noise phrase is its own label."""
    texts, labels = [], []
    for name, phrases in INTENTS.items():
        texts.extend(phrases)
        labels.extend([name] * len(phrases))
    texts.extend(NOISE)
    labels.extend(f"noise:{phrase}" for phrase in NOISE)
    return texts, labels


def report_cosines(similarity: np.ndarray, labels: list[str]) -> None:
    within, across = [], []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            bucket = within if labels[left] == labels[right] else across
            bucket.append(float(similarity[left, right]))
    within_arr, across_arr = np.array(within), np.array(across)

    print("\n=== 1. pairwise cosines ===")
    print(
        f"  within-intent  n={len(within):<5} min={within_arr.min():.3f} "
        f"median={np.median(within_arr):.3f} max={within_arr.max():.3f}"
    )
    print(
        f"  across-intent  n={len(across):<5} median={np.median(across_arr):.3f} "
        f"p90={np.percentile(across_arr, 90):.3f} max={across_arr.max():.3f}"
    )
    print(
        "  the distributions overlap: an unrelated pair can beat a related one, "
        "so no threshold is clean"
    )


def report_sweep(vectors: np.ndarray, labels: list[str], trials: int = 200) -> None:
    """Score every threshold over simulated pools at each mining trigger."""
    rng = np.random.default_rng(7)
    stats = {t: Counter() for t in THRESHOLDS}

    for _ in range(trials):
        order = rng.permutation(len(labels))
        for size in range(BATCH, len(labels) + 1, BATCH):
            index = order[:size]
            pool = [labels[i] for i in index]
            counts = Counter(pool)
            present = {
                name
                for name, count in counts.items()
                if not name.startswith("noise:") and count >= DEFAULT_MIN_CLUSTER
            }
            similarity = cosine_matrix(vectors[index].tolist())
            for threshold in THRESHOLDS:
                tally = stats[threshold]
                tally["runs"] += 1
                tally["present"] += len(present)
                groups = [
                    group
                    for group in components(similarity, threshold)
                    if len(group) >= DEFAULT_MIN_CLUSTER
                ]
                for group in groups:
                    names = {pool[i] for i in group}
                    tally["groups"] += 1
                    if len(names) == 1:
                        tally["pure"] += 1
                        tally["found"] += int(names.pop() in present)

    print(f"\n=== 2. threshold sweep ({trials} simulated pool histories) ===")
    print("   thr   purity  recall  calls/run")
    for threshold in THRESHOLDS:
        tally = stats[threshold]
        # No mineable cluster at all has no purity to report — "1.0" would read
        # as a perfect setting when it means the miner never runs.
        purity = f"{tally['pure'] / tally['groups']:.3f}" if tally["groups"] else "  -  "
        recall = tally["found"] / tally["present"] if tally["present"] else 0.0
        print(
            f"  {threshold:.2f}   {purity}   {recall:.3f}   "
            f"{tally['groups'] / tally['runs']:.2f}"
        )
    print("  pick the lowest threshold whose purity is 1.0: below it the miner pays for mistakes")


def report_centering(vectors: np.ndarray, labels: list[str]) -> None:
    """Centered cosines on the full probe, then on a pool that is one intent."""
    print("\n=== 3. centering the pool before the cosine ===")

    def centered_similarity(rows: np.ndarray) -> np.ndarray:
        shifted = rows - rows.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(shifted, axis=1, keepdims=True)
        unit = shifted / np.where(norms == 0.0, 1.0, norms)
        return unit @ unit.T

    similarity = centered_similarity(vectors)
    within, across = [], []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            bucket = within if labels[left] == labels[right] else across
            bucket.append(float(similarity[left, right]))
    print(
        f"  full probe: within median={np.median(within):+.3f}, "
        f"across median={np.median(across):+.3f} — separated far better than raw"
    )

    homogeneous = INTENTS["фокус"][:5]
    index = [i for i, label in enumerate(labels) if label == "фокус"][:5]
    pairs = centered_similarity(vectors[index])[np.triu_indices(len(index), k=1)]
    print(
        f"  pool of {len(homogeneous)} phrasings of ONE intent: centered cosines run "
        f"{pairs.min():+.3f}..{pairs.max():+.3f}"
    )
    print(
        "  every pair is negative — centering subtracts the pool mean, and in a "
        "single-intent pool the mean IS the intent. Not used."
    )


def report_latency(engine) -> None:
    print("\n=== 4. embed latency (inline in POST /api/chat) ===")
    phrases = [phrase for group in INTENTS.values() for phrase in group] + NOISE
    engine.embed(phrases[:5])  # warm: the first call builds the embed fn

    print("     n   median    ms/phrase")
    for size in (1, 5, 10, 20, 40, 100):
        sample = (phrases * 5)[:size]
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            engine.embed(sample)
            timings.append(time.perf_counter() - start)
        median = statistics.median(timings)
        print(f"   {size:>3}   {median * 1000:>6.0f} ms   {median / size * 1000:>6.1f}")

    question = {
        "skill": {
            "type": "choice",
            "instructions": "Какой навык подходит команде",
            "criteria": {"greet": "приветствие", "none": "ничего не подходит"},
        }
    }
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        engine.ask("покажи фокус", question)
        timings.append(time.perf_counter() - start)
    print(f"   one router predict, for scale: {statistics.median(timings) * 1000:.0f} ms")
    print("   both hold LayaEngine._lock, so an embed also stalls every other request's router")


def main() -> None:
    from backend.brain.engine import LayaEngine

    texts, labels = corpus()
    print(f"probe: {len(texts)} commands, {len(INTENTS)} recurring intents + {len(NOISE)} one-offs")

    start = time.perf_counter()
    engine = LayaEngine()
    print(f"checkpoint loaded in {time.perf_counter() - start:.1f}s")

    vectors = np.asarray(engine.embed(texts), dtype=float)
    report_cosines(cosine_matrix(vectors.tolist()), labels)
    report_sweep(vectors, labels)
    report_centering(vectors, labels)
    report_latency(engine)


if __name__ == "__main__":
    main()
