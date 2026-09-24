"""Grouping the misses: which commands are asking for the same thing.

**One Gemini call is the primary path, and the local vectors are the fallback.**
That is the opposite of how this module started, and it is the measured answer,
not a preference. Both groupers were scored on the same probe and the same
simulated pools (`scripts/calibrate_miner_sim.py`, section 5):

    grouper                      purity  recall  clusters/run
    Laya cosine, single-link      1.000   0.322     0.79
    one Gemini `group` call       0.875   0.808     2.00

The cosine path is pure only because it is nearly a duplicate detector: it finds
a third of the repeated intents in the pool, and the ones it finds are the
near-identical phrasings. Two and a half times the recall is worth 0.875 purity
here, because an impure cluster is not a shipped skill — it still has to survive
``backend/miner/backtest.py``, which is where it dies. A missed cluster is
simply never learned, and "share of commands handled without Gemini" is the
metric the whole project is measured on.

What it costs: one extra ``models.generate_content`` on flash-lite per mining
run — not per cluster — measured at ~0.9 s. The run already spends one of those
per proposal.

Why the cosine cannot carry it (JEB-1548, live ``multilingual`` checkpoint). Its
similarity ranges overlap, so no threshold separates them at any linkage. On the
worked example the *smallest* same-intent pair, "фокус покажи" / "а фокус
умеешь?", sits at 0.76, while "расскажи про квантовую физику" / "а фокус
умеешь?" — different topics entirely — sits at 0.85. Mean-pooled encoder states
are anisotropic; removing a reference corpus mean, whitening, average- and
complete-link and mutual-kNN were all measured and none of them opened a gap.

So the fallback keeps the threshold that is at least *safe*: 0.88, overridable
with ``MINER_SIM``, is the lowest value at which no mixed cluster survived on
that probe, and past ~0.91 nothing reaches ``MINER_MIN_CLUSTER`` at all.
Centering the pool before the cosine is the usual anisotropy fix and is not
used: it shatters a small pool that is genuinely all one intent, because there
the pool mean *is* the intent — the case the miner exists for.

Single-link on a threshold is exactly "connected components of the graph where
an edge means ``cos >= MINER_SIM``", so that is how it is computed — a union-find
over an ``n x n`` matrix. The pool is tens of rows; O(n^2) is the cheap option
here, and sklearn is a very large dependency for thirty lines of numpy.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import numpy as np

from ..brain.engine import DecisionEngine

log = logging.getLogger(__name__)

DEFAULT_SIM = 0.88
DEFAULT_MIN_CLUSTER = 3

#: Grouper = "given these commands, which belong together" -> groups of indices.
Grouper = Callable[[list[str]], list[list[int]]]


def sim_threshold() -> float:
    return float(os.environ.get("MINER_SIM", DEFAULT_SIM))


def min_cluster_size() -> int:
    """Below this a cluster is not mined and its cases stay in the pool.

    Two commands are a coincidence; three are a habit worth a skill. Cases below
    the bar are left ``mined=0`` deliberately — they ripen as more arrive.
    """
    return int(os.environ.get("MINER_MIN_CLUSTER", DEFAULT_MIN_CLUSTER))


def components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Connected components of ``similarity >= threshold``, in input order."""
    size = similarity.shape[0]
    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left in range(size):
        for right in range(left + 1, size):
            if similarity[left, right] >= threshold:
                parent[find(left)] = find(right)

    groups: dict[int, list[int]] = {}
    for node in range(size):
        groups.setdefault(find(node), []).append(node)
    return sorted(groups.values(), key=lambda group: group[0])


def cosine_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leaving the norm at 1 makes it similar to
    # nothing at all, which is the honest answer.
    matrix = matrix / np.where(norms == 0.0, 1.0, norms)
    return matrix @ matrix.T


def group_texts(
    engine: DecisionEngine, texts: list[str], grouper: Grouper | None = None
) -> list[list[int]]:
    """Group commands into clusters of indices into ``texts``.

    The teacher groups; the local vectors only take over when it cannot. An
    empty answer counts as "cannot": ``SkillGenerator.group`` swallows its own
    failures and returns ``[]``, and a run that mines nothing is the one outcome
    worth spending the cheap fallback on.
    """
    if len(texts) < 2:
        return [[0]] if texts else []

    if grouper is not None:
        groups = _validate_groups(grouper(texts), len(texts))
        if groups:
            return groups
        log.warning("miner: the teacher grouped nothing — falling back to local vectors")

    return _group_by_vectors(engine, texts)


def _group_by_vectors(engine: DecisionEngine, texts: list[str]) -> list[list[int]]:
    """Single-link over ``MINER_SIM``, or nothing if the encoder cannot embed."""
    try:
        vectors = engine.embed(texts)
    except Exception as exc:  # noqa: BLE001 — a broken encoder must not stop mining
        # Either the engine says it cannot embed (EmbeddingsUnavailable) or the
        # encoder itself blew up. Both mean the same thing here: no clustering.
        log.warning("miner: no local embeddings either (%r) — nothing to cluster", exc)
        return []

    if len(vectors) != len(texts) or not all(vectors):
        log.warning("miner: the encoder returned no usable vectors — nothing to cluster")
        return []
    return components(cosine_matrix(vectors), sim_threshold())


def _validate_groups(groups: list[list[int]], size: int) -> list[list[int]]:
    """Trust nothing a model returned: indices in range, each used at most once."""
    seen: set[int] = set()
    cleaned = []
    for group in groups:
        members = sorted({i for i in group if isinstance(i, int) and 0 <= i < size} - seen)
        if members:
            seen.update(members)
            cleaned.append(members)
    return sorted(cleaned, key=lambda group: group[0])
