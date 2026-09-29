"""Weighted Reciprocal Rank Fusion (Cormack, Clarke and Buttcher, SIGIR 2009).

score(d) = sum over legs of  w_leg / (rrf_k + rank_leg(d)),  ranks start at 1.
Rank-based, so dense cosine scores and BM25 scores never need to be calibrated against each other.
"""
from __future__ import annotations


def weighted_rrf(rankings: dict[str, list[str]], weights: dict[str, float], rrf_k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for leg, ids in rankings.items():
        w = weights.get(leg, 1.0)
        for rank, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
