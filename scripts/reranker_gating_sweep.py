"""Evaluate saved reranker choices over heldout-derived gating thresholds.

The choices come from run_system.py, so this sweep requires no model inference.
Thresholds are selected on heldout while EM/F1 is measured on dev.

Example
-------
python scripts/reranker_gating_sweep.py --candidates data/candidates/dev.jsonl \
    --tau_from data/candidates/heldout.jsonl \
    --reranker_choices results/dev/reranker_choice_hard.json \
    --tag hard --out_dir results/dev
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.confidence import MARGIN_KINDS  # noqa: E402
from qa.data import read_jsonl  # noqa: E402
from qa.evaluation import actual_reranker_sweep  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, help="evaluation candidates, normally dev.jsonl")
    p.add_argument("--tau_from", required=True, help="heldout candidates used to select thresholds")
    p.add_argument("--reranker_choices", required=True, help="JSON written by run_system.py")
    p.add_argument("--fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--kind", choices=MARGIN_KINDS, default="prob_margin")
    p.add_argument("--tag", default="hard")
    p.add_argument("--out_dir", default="results/dev")
    p.add_argument("--no_plot", action="store_true", help="write CSV/JSON only")
    args = p.parse_args()

    eval_rows = read_jsonl(args.candidates)
    heldout_rows = read_jsonl(args.tau_from)
    choices = json.loads(Path(args.reranker_choices).read_text(encoding="utf-8"))
    table = actual_reranker_sweep(
        eval_rows, heldout_rows, choices, args.fracs, k=args.k, kind=args.kind,
    )
    for row in table:
        row["tag"] = args.tag
        row["kind"] = args.kind
        row["k"] = args.k

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"reranker_gating_sweep_{args.tag}.csv"
    json_path = out / f"reranker_gating_sweep_{args.tag}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    json_path.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Actual-reranker sweep ({args.tag}; thresholds selected on heldout)")
    print(f"{'target%':>8} {'actual%':>8} {'tau':>10} {'EM':>7} {'F1':>7} {'dEM':>7} {'dF1':>7} {'+right':>7} {'-right':>7}")
    for row in table:
        print(
            f"{row['target_gate_pct']:8.1f} {row['actual_gate_pct']:8.1f} {row['tau']:10.4f} "
            f"{row['gated_em']:7.2f} {row['gated_f1']:7.2f} "
            f"{row['delta_em']:+7.2f} {row['delta_f1']:+7.2f} "
            f"{row['wrong_to_right']:7d} {row['right_to_wrong']:7d}"
        )

    if args.no_plot:
        print(f"Saved {csv_path} and {json_path}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [row["actual_gate_pct"] for row in table]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, label in ((axes[0], "em", "Exact Match"), (axes[1], "f1", "F1")):
            ax.plot(x, [row[f"gated_{metric}"] for row in table], marker="o", label="actual reranker")
            ax.plot(x, [row[f"oracle_{metric}"] for row in table], marker="o", ls="--", label="oracle")
            ax.axhline(table[0][f"baseline_{metric}"], color="grey", ls=":", label="baseline")
            ax.set_xlabel("Actual % of dev questions reranked")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[0].legend()
        fig.tight_layout()
        plot_path = out / f"reranker_gating_sweep_{args.tag}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved {plot_path}")
    except ImportError:
        pass

    print(f"Saved {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
