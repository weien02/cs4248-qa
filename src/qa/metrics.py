"""Official SQuAD v1.1 EM/F1 (same normalisation as the official eval script),
plus helpers for per-question-type breakdowns."""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def max_over_gold(fn, pred: str, golds: list[str]) -> float:
    return max(fn(pred, g) for g in golds) if golds else 0.0


def score_predictions(preds: dict[str, str], golds: dict[str, list[str]]) -> dict:
    """Return {'exact_match', 'f1', 'n'} in percent, like the official script."""
    em = f = 0.0
    for qid, gold in golds.items():
        pred = preds.get(qid, "")
        em += max_over_gold(exact_match, pred, gold)
        f += max_over_gold(f1, pred, gold)
    n = max(len(golds), 1)
    return {"exact_match": 100 * em / n, "f1": 100 * f / n, "n": len(golds)}


_QTYPES = ["how many", "how much", "what", "which", "who", "whom", "whose",
           "when", "where", "why", "how"]


def question_type(q: str) -> str:
    ql = " " + q.lower() + " "
    for t in _QTYPES:  # multi-word types listed first so they win
        if f" {t} " in ql or ql.strip().startswith(t):
            return {"whom": "who", "whose": "who"}.get(t, t)
    return "other"


def breakdown_by_type(preds, golds, questions: dict[str, str]) -> dict[str, dict]:
    buckets = defaultdict(dict)
    for qid, gold in golds.items():
        buckets[question_type(questions[qid])][qid] = gold
    return {t: score_predictions(preds, g) for t, g in sorted(buckets.items())}
