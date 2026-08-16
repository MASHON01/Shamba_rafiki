"""
Cosine similarity and ranking helpers.

Pure, dependency-light functions for scoring and ordering vectors.
The vector stores each compute similarity internally (numpy via
matrix products, FAISS via its index), so these helpers are NOT on
the hot retrieval path. They exist for the cases that sit outside a
store:

- re-ranking or de-duplicating an already-retrieved result set,
- scoring a handful of vectors in tests or diagnostics,
- any future component that needs a similarity number without
  standing up a whole store.

Keeping them here - separate, tiny, and store-agnostic - means the
definition of "cosine similarity" lives in exactly one place and can
be unit-tested in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_L2_EPSILON = 1e-12


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity of two equal-length vectors, in [-1.0, 1.0].

    Raises
    ------
    ValueError
        If the vectors differ in length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must be the same length ({len(a)} != {len(b)})."
        )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom < _L2_EPSILON:
        return 0.0
    return dot / denom


def rank_by_similarity(
    query: Sequence[float],
    candidates: Sequence[Sequence[float]],
) -> list[tuple[int, float]]:
    """
    Score every candidate against `query` and return
    (original_index, score) pairs sorted by descending score.

    The original index is preserved so callers can map a ranking back
    to whatever the vectors represent.
    """
    scored = [
        (i, cosine_similarity(query, candidate))
        for i, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def apply_threshold(
    scored: Sequence[tuple[int, float]],
    threshold: float | None,
) -> list[tuple[int, float]]:
    """
    Drop (index, score) pairs whose score is below `threshold`.
    A `None` threshold keeps everything.
    """
    if threshold is None:
        return list(scored)
    return [(i, s) for i, s in scored if s >= threshold]


def top_k(
    scored: Sequence[tuple[int, float]],
    k: int,
) -> list[tuple[int, float]]:
    """
    Keep the k highest-scoring pairs. Assumes `scored` may be in any
    order; sorts defensively before slicing.
    """
    if k <= 0:
        return []
    ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
    return ordered[:k]