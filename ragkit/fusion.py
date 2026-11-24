"""Combining the vector and keyword candidate lists."""

from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Reciprocal rank fusion, normalised to 0..1.

    RRF only looks at positions, so it needs no score calibration between a
    cosine similarity and a BM25 score. The result is divided by the best
    possible score (rank 1 in every list), so 1.0 means "top of every list".
    """
    weights = list(weights) if weights else [1.0] * len(rankings)
    raw: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            raw[item] = raw.get(item, 0.0) + w / (k + rank)
    best = sum(w / (k + 1) for w in weights) or 1.0
    return {item: s / best for item, s in raw.items()}


def weighted_fusion(
    vector: dict[str, float], keyword: dict[str, float], alpha: float = 0.6
) -> dict[str, float]:
    """Min-max normalise each score list, then mix them: alpha * vector + (1 - alpha) * keyword."""

    def norm(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    v, kw = norm(vector), norm(keyword)
    return {i: alpha * v.get(i, 0.0) + (1 - alpha) * kw.get(i, 0.0) for i in set(v) | set(kw)}
