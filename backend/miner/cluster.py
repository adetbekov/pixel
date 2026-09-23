"""Grouping the misses: which commands are asking for the same thing.

Single-link agglomerative clustering on cosine similarity. "Single-link above a
threshold" is exactly "connected components of the graph where an edge means
``cos >= MINER_SIM``", so that is how it is computed — a union-find over an
``n x n`` matrix. The pool is tens of rows; O(n^2) is the cheap option here, and
sklearn is a very large dependency for thirty lines of numpy.

Threshold. 0.88, and ``MINER_SIM`` overrides it. The number is measured against
the real ``multilingual`` vectors (JEB-1509), not chosen. Mean-pooled encoder
states are anisotropic — every cosine comes out high, and on a probe of Russian
commands the across-intent spread (median 0.68, p90 0.81, max 0.86) runs
straight through the within-intent one (min 0.61, median 0.81). Only a narrow
band at the top tells the two apart: over simulated pools 0.75 got 999 of every
1000 mineable clusters mixed and still spent a Gemini call per run, 0.88 is the
lowest value at which no mixed cluster survived, and past ~0.91 nothing reaches
``MINER_MIN_CLUSTER`` at all. What the high bar costs is recall: about a third
of repeated intents group, and they are the near-identical phrasings — "спой
песню" and "давай ты споёшь" do not reach it. That is a limit of these vectors,
not of the threshold.

Centering the pool before the cosine (the usual anisotropy fix) was measured
too and is not used: it separates a large mixed probe better, and it shatters a
small pool that is genuinely all one intent, because there the pool mean *is*
the intent. That is the case the miner exists for.

Fallback. Without sentence vectors the commands are grouped by one Gemini call
instead. It exists so a missing ``embed_fn_from_agent`` degrades instead of
stopping the pipeline — it is not the default path, and it is not free.
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
    """Group commands into clusters of indices into ``texts``."""
    if len(texts) < 2:
        return [[0]] if texts else []

    try:
        vectors = engine.embed(texts)
    except Exception as exc:  # noqa: BLE001 — a broken encoder must not stop mining
        # Either the engine says it cannot embed (EmbeddingsUnavailable) or the
        # encoder itself blew up. Both mean the same thing here: fall back.
        log.warning("miner: no local embeddings (%r), grouping with the teacher instead", exc)
        vectors = []

    if len(vectors) == len(texts) and all(vectors):
        return components(cosine_matrix(vectors), sim_threshold())

    if grouper is None:
        log.warning("miner: no embeddings and no fallback grouper — nothing to cluster")
        return []
    return _validate_groups(grouper(texts), len(texts))


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
