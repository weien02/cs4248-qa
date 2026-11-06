"""Run a paired bootstrap test for two saved QA prediction files.

Example
-------
python scripts/paired_bootstrap.py --candidates data/candidates/dev.jsonl \
    --predictions_a results/dev/predictions_hard_baseline.json --name_a baseline \
    --predictions_b results/dev/predictions_hard_gated.json --name_b hard_gated \
    --tag hard --out_dir results/dev
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.data import read_jsonl  # noqa: E402
from qa.evaluation import paired_bootstrap  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, help="JSONL containing ids and gold answers")
    p.add_argument("--predictions_a", required=True)
    p.add_argument("--predictions_b", required=True)
    p.add_argument("--name_a", default="system_a")
    p.add_argument("--name_b", default="system_b")
    p.add_argument("--samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confidence_level", type=float, default=0.95)
    p.add_argument("--tag", default="comparison")
    p.add_argument("--out_dir", default="results/dev")
    args = p.parse_args()

    rows = read_jsonl(args.candidates)
    golds = {row["id"]: row["answers"] for row in rows}
    predictions_a = json.loads(Path(args.predictions_a).read_text(encoding="utf-8"))
    predictions_b = json.loads(Path(args.predictions_b).read_text(encoding="utf-8"))
    result = paired_bootstrap(
        predictions_a, predictions_b, golds,
        n_samples=args.samples, seed=args.seed, confidence_level=args.confidence_level,
    )
    result["system_a_name"] = args.name_a
    result["system_b_name"] = args.name_b

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"paired_bootstrap_{args.tag}.json"
    csv_path = out / f"paired_bootstrap_{args.tag}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows = []
    for metric, values in result["metrics"].items():
        row = {"metric": metric, "system_a_name": args.name_a, "system_b_name": args.name_b}
        row.update(values)
        csv_rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Paired bootstrap: {args.name_b} - {args.name_a} | n={result['n']} | samples={args.samples}")
    for metric, values in result["metrics"].items():
        level = 100 * args.confidence_level
        print(
            f"{metric:12s}: {values['system_a']:.2f} -> {values['system_b']:.2f} "
            f"diff={values['difference']:+.2f} | {level:.0f}% CI "
            f"[{values['ci_low']:+.2f}, {values['ci_high']:+.2f}] | "
            f"one-sided p={values['p_one_sided_b_not_better']:.6f}"
        )
    print(f"Saved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
