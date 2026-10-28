"""Summarise robustness runs: score drop under distractors and how often the distractor is picked.

Expects the layout written by cluster/robustness.sh:
    data/candidates/robustness/<set>.jsonl                 candidates per test set ("clean" = dev)
    results/robustness/predictions_<set>_<neg>_<mode>.json from run_system.py --tag <set>_<neg>
    results/robustness/metrics_<set>_<neg>.json

Per (set, system) it reports EM/F1, the same questions' clean EM/F1 and the drop, the share
of questions sent to the reranker, and the capture rate: % of predictions that lie inside
the inserted sentence.

  distractor sets (make_distractors.py): compared with clean dev on the same question ids.
  adversarial sets (addsent / addonesent): official protocol, the worst EM/F1 over each
      original question's versions, compared with the originals in the same file.

Example
-------
python scripts/robustness_report.py --results_dir results/robustness
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.confidence import MARGIN_KINDS, confidence  # noqa: E402
from qa.data import read_jsonl  # noqa: E402
from qa.metrics import exact_match, f1, max_over_gold  # noqa: E402

ADVERSARIAL = ("addsent", "addonesent")
# (mode, uses a reranker): baseline and oracle_gated don't depend on the negative type
MODES = (("baseline", False), ("oracle_gated", False), ("rerank_all", True), ("gated", True))


def pred_span(row, text):
    """Character span of the predicted text: the highest-ranked candidate with that text."""
    c = next((c for c in row["candidates"] if c["text"] == text), None)
    return (c["start"], c["end"]) if c else None


def distractor_span(row, orig_ctx):
    if "distractor" in row:  # make_distractors.py inserts " " + sentence at distractor_pos
        at, n = row["distractor_pos"], len(row["distractor"])
        return (0, n) if at == 0 else (at + 1, at + 1 + n)
    if orig_ctx is not None and "-" in row["id"]:  # Adversarial SQuAD appends the sentence
        return (len(orig_ctx), len(row["context"]))
    return None


def scores(rows, preds):
    return {r["id"]: (max_over_gold(exact_match, preds.get(r["id"], ""), r["answers"]),
                      max_over_gold(f1, preds.get(r["id"], ""), r["answers"])) for r in rows}


def worst_case(per_id):
    """Adversarial protocol: min EM and min F1 over all versions of each original question."""
    groups = defaultdict(list)
    for qid, s in per_id.items():
        groups[qid.split("-")[0]].append(s)
    return {base: (min(e for e, _ in v), min(f for _, f in v)) for base, v in groups.items()}


def mean(per_id, i):
    return 100 * sum(v[i] for v in per_id.values()) / max(len(per_id), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates_dir", default="data/candidates/robustness")
    p.add_argument("--results_dir", default="results/robustness")
    p.add_argument("--out_dir", help="default: --results_dir")
    p.add_argument("--tau", type=float,
                   help="gating threshold used by run_system.py (its runs.csv 'tau' column); adds "
                        "gate_vs_capture.md: does the gate route the fooled questions to the reranker?")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--kind", choices=MARGIN_KINDS, default="prob_margin")
    args = p.parse_args()
    res, out = Path(args.results_dir), Path(args.out_dir or args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    def load_preds(tag, mode):
        f = res / f"predictions_{tag}_{mode}.json"
        return json.loads(f.read_text()) if f.exists() else None

    tags = sorted(f.stem[len("metrics_"):] for f in res.glob("metrics_*.json"))
    sets = sorted({t.rsplit("_", 1)[0] for t in tags} - {"clean"})
    negs = sorted({t.rsplit("_", 1)[1] for t in tags})
    if "clean" not in {t.rsplit("_", 1)[0] for t in tags}:
        sys.exit(f"no clean_<neg> run in {res}; robustness.sh runs one on dev")
    clean_rows = {r["id"]: r for r in read_jsonl(Path(args.candidates_dir) / "clean.jsonl")}

    table, gate_rows = [], []
    for s in sets:
        rows = read_jsonl(Path(args.candidates_dir) / f"{s}.jsonl")
        adversarial = s in ADVERSARIAL
        if adversarial:
            orig_ctx = {r["id"]: r["context"] for r in rows if "-" not in r["id"]}
            originals = [r for r in rows if "-" not in r["id"]]
        else:
            orig_ctx = {}
            originals = [clean_rows[r["id"]] for r in rows if r["id"] in clean_rows]
        attacked = [r for r in rows if distractor_span(r, orig_ctx.get(r["id"].split("-")[0]))]

        base = load_preds(f"{s}_{negs[0]}", "baseline")
        if args.tau is not None and base is not None:
            # baseline answers on attacked passages: fooled (inside the distractor) vs correct
            fooled, right = [], []
            for r in attacked:
                sp = pred_span(r, base.get(r["id"], ""))
                d = distractor_span(r, orig_ctx.get(r["id"].split("-")[0]))
                gated = confidence(r["candidates"], args.k, args.kind) < args.tau
                if sp and sp[0] < d[1] and d[0] < sp[1]:
                    fooled.append(gated)
                elif max_over_gold(exact_match, base.get(r["id"], ""), r["answers"]):
                    right.append(gated)
            pct = lambda xs: round(100 * sum(xs) / max(len(xs), 1), 1)  # noqa: E731
            gate_rows.append({"set": s, "attacked": len(attacked), "fooled": len(fooled),
                              "pct_fooled_gated": pct(fooled), "correct": len(right),
                              "pct_correct_gated": pct(right)})

        for mode, needs_rr in MODES:
            for neg in (negs if needs_rr else negs[:1]):
                preds = load_preds(f"{s}_{neg}", mode)
                clean_tag = f"{s}_{neg}" if adversarial else f"clean_{neg}"
                clean_preds = load_preds(clean_tag, mode)
                if preds is None or clean_preds is None:
                    continue
                per_id = scores(rows, preds)
                if adversarial:
                    per_id = worst_case(per_id)
                clean = scores(originals, clean_preds)

                captured = 0
                for r in attacked:
                    sp = pred_span(r, preds.get(r["id"], ""))
                    d = distractor_span(r, orig_ctx.get(r["id"].split("-")[0]))
                    captured += bool(sp and sp[0] < d[1] and d[0] < sp[1])

                metrics_file = res / f"metrics_{s}_{neg}.json"
                pct = json.loads(metrics_file.read_text()).get(mode, {}).get("pct_reranked")
                em, f, cem, cf = mean(per_id, 0), mean(per_id, 1), mean(clean, 0), mean(clean, 1)
                table.append({
                    "set": s, "mode": mode, "neg": neg if needs_rr else "-", "n": len(per_id),
                    "EM": round(em, 2), "F1": round(f, 2), "clean_EM": round(cem, 2), "clean_F1": round(cf, 2),
                    "dEM": round(em - cem, 2), "dF1": round(f - cf, 2),
                    "pct_reranked": None if pct is None else round(pct, 1),
                    "capture_pct": round(100 * captured / max(len(attacked), 1), 2),
                })

    if not table:
        sys.exit(f"no prediction files found in {res}")
    with (out / "robustness_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    cols = list(table[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if t[c] is None else str(t[c]) for c in cols) + " |" for t in table]
    (out / "robustness_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved {out / 'robustness_summary.csv'} and .md")

    if gate_rows:
        cols = list(gate_rows[0])
        md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join(str(t[c]) for c in cols) + " |" for t in gate_rows]
        (out / "gate_vs_capture.md").write_text(
            f"Baseline answers on attacked passages, tau={args.tau} ({args.kind}, k={args.k}). "
            "fooled = answer lies inside the inserted sentence; gated = would be sent to the reranker.\n\n"
            + "\n".join(md) + "\n")
        print("\n" + "\n".join(md))
        print(f"Saved {out / 'gate_vs_capture.md'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        neg = "hard" if "hard" in negs else negs[0]
        systems = [("baseline", "-"), ("gated", neg), ("rerank_all", neg)]
        fig, ax = plt.subplots(figsize=(1.2 * len(sets) + 3, 4))
        width = 0.8 / len(systems)
        for i, (mode, ng) in enumerate(systems):
            vals = [next((t["dF1"] for t in table if t["set"] == s and t["mode"] == mode and t["neg"] == ng),
                         float("nan")) for s in sets]
            ax.bar([j + i * width for j in range(len(sets))], vals, width,
                   label=mode if ng == "-" else f"{mode} ({ng})")
        ax.set_xticks([j + width * (len(systems) - 1) / 2 for j in range(len(sets))])
        ax.set_xticklabels(sets, rotation=30, ha="right")
        ax.axhline(0, c="grey", lw=0.8)
        ax.set_ylabel("F1 change vs clean")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "robustness_drop.png", dpi=150)
        print(f"Saved {out / 'robustness_drop.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
