"""Analyse baseline confidence and pick the gating threshold tau.

Run on the HELDOUT candidates (never dev) so tau isn't tuned on the test set.

For each threshold (one per % of questions routed to the reranker), reports:
  - frac_gated        : share of questions sent to the reranker
  - em_confident      : baseline top-1 EM on questions kept as-is
  - em_gated          : baseline top-1 EM on gated questions (should be much lower)
  - oracle_gated      : % of gated questions whose gold is somewhere in the top-k
                        (= best EM a perfect reranker could reach on that subset)
  - oracle_overall_em : overall EM if the reranker were perfect

Example
-------
python scripts/gating_analysis.py --candidates data/candidates/heldout.jsonl \
    --k 5 --out_dir results/gating
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from qa.confidence import MARGIN_KINDS, confidence  # noqa: E402
from qa.data import read_jsonl  # noqa: E402


def analyse(rows, k, kind, fracs):
    conf = np.array([confidence(r["candidates"], k, kind) for r in rows])
    top1_ok = np.array([r["gold_rank"] == 0 for r in rows])
    in_k = np.array([0 <= r["gold_rank"] < k for r in rows])
    finite = conf[np.isfinite(conf)]
    table = []
    for f in fracs:
        tau = float(np.quantile(finite, f)) if f > 0 else -np.inf
        if f >= 1:
            tau = np.inf
        gated = conf < tau
        conf_mask = ~gated
        table.append({
            "kind": kind, "frac_target": f, "tau": tau,
            "frac_gated": gated.mean(),
            "em_confident": top1_ok[conf_mask].mean() if conf_mask.any() else np.nan,
            "em_gated": top1_ok[gated].mean() if gated.any() else np.nan,
            "oracle_gated": in_k[gated].mean() if gated.any() else np.nan,
            "oracle_overall_em": (top1_ok[conf_mask].sum() + in_k[gated].sum()) / len(rows),
        })
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--kinds", nargs="+", default=list(MARGIN_KINDS))
    p.add_argument("--out_dir", default="results/gating")
    args = p.parse_args()

    rows = read_jsonl(args.candidates)
    fracs = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_em = np.mean([r["gold_rank"] == 0 for r in rows]) * 100
    print(f"{len(rows)} questions | baseline top-1 EM {base_em:.2f} | "
          f"gold in top-{args.k}: {np.mean([0 <= r['gold_rank'] < args.k for r in rows]) * 100:.1f}%")

    all_rows = []
    for kind in args.kinds:
        table = analyse(rows, args.k, kind, fracs)
        all_rows += table
        print(f"\n== {kind} ==")
        print(f"{'gated%':>7} {'tau':>10} {'EM_conf':>8} {'EM_gated':>9} {'oracle_g':>9} {'oracle_EM':>10}")
        for t in table:
            print(f"{100 * t['frac_gated']:7.1f} {t['tau']:10.4f} {100 * t['em_confident']:8.2f} "
                  f"{100 * t['em_gated']:9.2f} {100 * t['oracle_gated']:9.2f} {100 * t['oracle_overall_em']:10.2f}")

    with (out / "gating_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {out / 'gating_sweep.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for kind in args.kinds:
            t = [r for r in all_rows if r["kind"] == kind and 0 < r["frac_gated"] < 1]
            ax.plot([100 * r["frac_gated"] for r in t], [100 * r["em_gated"] for r in t], marker="o", label=kind)
        ax.axhline(base_em, ls="--", c="grey", label="overall top-1 EM")
        ax.set_xlabel("% of questions sent to reranker")
        ax.set_ylabel("baseline EM on gated questions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "gating_curve.png", dpi=150)
        print(f"Saved {out / 'gating_curve.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
