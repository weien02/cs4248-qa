"""Statistical evaluation and confidence-gating sweep helpers."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .confidence import confidence
from .metrics import exact_match, f1, max_over_gold, score_predictions


def _per_example_scores(preds: dict[str, str], golds: dict[str, list[str]], metric) -> np.ndarray:
    return np.array([
        max_over_gold(metric, preds.get(qid, ""), answers)
        for qid, answers in golds.items()
    ], dtype=np.float64)


def paired_bootstrap(
    predictions_a: dict[str, str],
    predictions_b: dict[str, str],
    golds: dict[str, list[str]],
    *,
    n_samples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
    batch_size: int = 128,
) -> dict:
    """Compare two systems by resampling the same questions with replacement.

    Scores and differences are in percentage points. The one-sided empirical
    p-value estimates P(B <= A); add-one smoothing prevents a reported zero.
    """
    if not golds:
        raise ValueError("golds must not be empty")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n = len(golds)
    alpha = (1.0 - confidence_level) / 2.0
    results = {}

    for name, metric in (("exact_match", exact_match), ("f1", f1)):
        a = _per_example_scores(predictions_a, golds, metric)
        b = _per_example_scores(predictions_b, golds, metric)
        per_question_delta = b - a
        bootstrap_differences = np.empty(n_samples, dtype=np.float64)

        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            indices = rng.integers(0, n, size=(size, n), dtype=np.int32)
            bootstrap_differences[start:start + size] = per_question_delta[indices].mean(axis=1) * 100

        non_positive = (np.count_nonzero(bootstrap_differences <= 0) + 1) / (n_samples + 1)
        non_negative = (np.count_nonzero(bootstrap_differences >= 0) + 1) / (n_samples + 1)
        results[name] = {
            "system_a": float(a.mean() * 100),
            "system_b": float(b.mean() * 100),
            "difference": float(per_question_delta.mean() * 100),
            "ci_low": float(np.quantile(bootstrap_differences, alpha)),
            "ci_high": float(np.quantile(bootstrap_differences, 1.0 - alpha)),
            "standard_error": float(bootstrap_differences.std(ddof=1)),
            "p_one_sided_b_not_better": float(non_positive),
            "p_two_sided": float(min(1.0, 2.0 * min(non_positive, non_negative))),
        }

    return {
        "n": n,
        "n_samples": n_samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": results,
    }


def actual_reranker_sweep(
    eval_rows: list[dict],
    heldout_rows: list[dict],
    reranker_choices: dict[str, str],
    fractions: Iterable[float],
    *,
    k: int = 5,
    kind: str = "prob_margin",
) -> list[dict]:
    """Evaluate saved reranker choices at heldout-derived gate fractions."""
    if not eval_rows or not heldout_rows:
        raise ValueError("eval_rows and heldout_rows must not be empty")

    missing = [r["id"] for r in eval_rows if r["id"] not in reranker_choices]
    if missing:
        raise ValueError(f"reranker choices missing {len(missing)} question ids; first={missing[0]}")

    held_conf = np.array([confidence(r["candidates"], k, kind) for r in heldout_rows])
    held_finite = held_conf[np.isfinite(held_conf)]
    if not len(held_finite):
        raise ValueError("heldout set has no finite confidence values")

    eval_conf = {r["id"]: confidence(r["candidates"], k, kind) for r in eval_rows}
    golds = {r["id"]: r["answers"] for r in eval_rows}
    baseline = {r["id"]: r["candidates"][0]["text"] for r in eval_rows}
    rerank_all = {qid: reranker_choices[qid] for qid in golds}
    baseline_metrics = score_predictions(baseline, golds)
    rerank_all_metrics = score_predictions(rerank_all, golds)
    baseline_ok = {
        qid: max_over_gold(exact_match, baseline[qid], answers)
        for qid, answers in golds.items()
    }

    table = []
    for fraction in fractions:
        if not 0 < fraction < 1:
            raise ValueError(f"gating fractions must be between 0 and 1, got {fraction}")
        tau = float(np.quantile(held_finite, fraction))
        gated_ids = {qid for qid, value in eval_conf.items() if value < tau}
        predictions = {
            qid: rerank_all[qid] if qid in gated_ids else baseline[qid]
            for qid in golds
        }
        oracle = {}
        for row in eval_rows:
            qid = row["id"]
            if qid not in gated_ids:
                oracle[qid] = baseline[qid]
                continue
            oracle[qid] = next(
                (candidate["text"] for candidate in row["candidates"][:k]
                 if max_over_gold(exact_match, candidate["text"], row["answers"])),
                baseline[qid],
            )

        metrics = score_predictions(predictions, golds)
        oracle_metrics = score_predictions(oracle, golds)
        system_ok = {
            qid: max_over_gold(exact_match, predictions[qid], answers)
            for qid, answers in golds.items()
        }
        gated_golds = {qid: golds[qid] for qid in gated_ids}
        baseline_subset = score_predictions(baseline, gated_golds)
        reranker_subset = score_predictions(rerank_all, gated_golds)

        table.append({
            "target_gate_pct": 100 * fraction,
            "actual_gate_pct": 100 * len(gated_ids) / len(golds),
            "tau": tau,
            "baseline_em": baseline_metrics["exact_match"],
            "baseline_f1": baseline_metrics["f1"],
            "gated_em": metrics["exact_match"],
            "gated_f1": metrics["f1"],
            "delta_em": metrics["exact_match"] - baseline_metrics["exact_match"],
            "delta_f1": metrics["f1"] - baseline_metrics["f1"],
            "oracle_em": oracle_metrics["exact_match"],
            "oracle_f1": oracle_metrics["f1"],
            "rerank_all_em": rerank_all_metrics["exact_match"],
            "rerank_all_f1": rerank_all_metrics["f1"],
            "baseline_em_on_gated": baseline_subset["exact_match"],
            "reranker_em_on_gated": reranker_subset["exact_match"],
            "wrong_to_right": int(sum(system_ok[q] and not baseline_ok[q] for q in golds)),
            "right_to_wrong": int(sum(baseline_ok[q] and not system_ok[q] for q in golds)),
        })
    return table
