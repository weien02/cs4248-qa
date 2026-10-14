"""Confidence estimates from the baseline's top-k span scores.

All functions take a candidates list sorted by score (as written by
scripts/dump_candidates.py). Higher value = more confident.
"""
from __future__ import annotations

import math

MARGIN_KINDS = ("prob_margin", "score_margin", "top1_prob")


def topk_probs(cands: list[dict], k: int) -> list[float]:
    scores = [c["score"] for c in cands[:k]]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [x / z for x in exps]


def confidence(cands: list[dict], k: int = 5, kind: str = "prob_margin") -> float:
    if len(cands) < 2:
        return float("inf")  # only one candidate: nothing to rerank
    if kind == "score_margin":
        return cands[0]["score"] - cands[1]["score"]
    p = topk_probs(cands, k)
    if kind == "prob_margin":
        return p[0] - p[1]
    if kind == "top1_prob":
        return p[0]
    raise ValueError(f"unknown confidence kind {kind!r}; choose from {MARGIN_KINDS}")


def is_uncertain(cands: list[dict], tau: float, k: int = 5, kind: str = "prob_margin") -> bool:
    return confidence(cands, k, kind) < tau
