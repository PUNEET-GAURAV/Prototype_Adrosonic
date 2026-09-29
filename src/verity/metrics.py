"""Deterministic IR metrics (binary relevance) and latency statistics."""
from __future__ import annotations

import math

import numpy as np


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return float(any(p in gold for p in ranked[:k]))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold.intersection(ranked[:k])) / len(gold) if gold else 0.0


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for i, p in enumerate(ranked[:k], start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def latency_summary(ms: list[float], budget_ms: float = 300.0) -> dict:
    a = np.asarray(ms, dtype=float)
    p95 = float(np.percentile(a, 95))
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": p95,
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "budget_ms": budget_ms,
            "pass": bool(p95 < budget_ms), "percentile_method": "numpy linear interpolation"}
