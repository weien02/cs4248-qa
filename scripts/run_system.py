"""Evaluate the full system and its ablations on a candidates file.

Modes reported in one run:
  baseline     : top-1 baseline span
  rerank_all   : reranker applied to every question (no gating)
  gated        : reranker only when confidence < tau (the proposed system)
  oracle_gated : gated, with a perfect reranker (upper bound)

Appends one row per mode to results/ablations.csv, tagged with --tag, and
writes per-question-type breakdowns + right/wrong flip counts.

Example
-------
python scripts/run_system.py --candidates data/candidates/dev.jsonl \
    --reranker out/reranker-hard/best --tau 0.35 --kind prob_margin \
    --tag hardneg --out_dir results/dev
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from qa.confidence import MARGIN_KINDS, confidence  # noqa: E402
from qa.data import read_jsonl  # noqa: E402
from qa.metrics import breakdown_by_type, exact_match, max_over_gold, score_predictions  # noqa: E402
from qa.utils import get_device, save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True)
    p.add_argument("--reranker", help="dir saved by train_reranker.py (e.g. out/reranker-hard/best)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--tau", type=float, help="confidence threshold (pick on heldout with gating_analysis.py)")
    p.add_argument("--tau_from", help="heldout candidates file to derive tau from (instead of --tau)")
    p.add_argument("--gate_frac", type=float, default=0.2,
                   help="with --tau_from: set tau so this fraction of HELDOUT questions is gated")
    p.add_argument("--kind", choices=MARGIN_KINDS, default="prob_margin")
    p.add_argument("--tag", default="run", help="label for this configuration in ablations.csv")
    p.add_argument("--out_dir", default="results/dev")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    rows = read_jsonl(args.candidates)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    golds = {r["id"]: r["answers"] for r in rows}
    questions = {r["id"]: r["question"] for r in rows}

    if args.tau is None:
        if not args.tau_from:
            p.error("give --tau, or --tau_from heldout.jsonl with --gate_frac")
        import numpy as np
        held = [confidence(r["candidates"], args.k, args.kind) for r in read_jsonl(args.tau_from)]
        held = np.array([c for c in held if np.isfinite(c)])
        args.tau = float(np.quantile(held, args.gate_frac))
        print(f"tau={args.tau:.4f} ({args.kind}) gates {100 * args.gate_frac:.0f}% of heldout")

    conf = {r["id"]: confidence(r["candidates"], args.k, args.kind) for r in rows}
    gated_ids = {qid for qid, c in conf.items() if c < args.tau}

    preds = {"baseline": {r["id"]: r["candidates"][0]["text"] for r in rows}}
    preds["oracle_gated"] = {
        r["id"]: (next((c["text"] for c in r["candidates"][:args.k]
                        if max_over_gold(exact_match, c["text"], r["answers"])),
                       r["candidates"][0]["text"]) if r["id"] in gated_ids else r["candidates"][0]["text"])
        for r in rows
    }

    if args.reranker:
        from qa.reranker import SpanReranker, build_group, score_groups
        model, tok = SpanReranker.load(args.reranker)
        groups = [build_group(r, args.k, train=False) for r in rows]
        logits = score_groups(model, tok, groups, get_device(), args.batch_size, args.max_length,
                              torch.bfloat16 if args.bf16 else None)
        rr = {g.qid: g.texts[max(range(len(lg)), key=lg.__getitem__)] for g, lg in zip(groups, logits)}
        save_json(rr, out / f"reranker_choice_{args.tag}.json")
        preds["rerank_all"] = rr
        preds["gated"] = {qid: (rr[qid] if qid in gated_ids else preds["baseline"][qid]) for qid in golds}

    base_ok = {qid: max_over_gold(exact_match, preds["baseline"][qid], g) for qid, g in golds.items()}
    results, abl_path = {}, Path("results/ablations.csv")
    new_file = not abl_path.exists()
    abl_path.parent.mkdir(parents=True, exist_ok=True)
    with abl_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tag", "candidates", "mode", "k", "kind", "tau", "pct_gated",
                        "EM", "F1", "wrong_to_right", "right_to_wrong"])
        for mode, pr in preds.items():
            m = score_predictions(pr, golds)
            ok = {qid: max_over_gold(exact_match, pr[qid], g) for qid, g in golds.items()}
            m["wrong_to_right"] = sum(ok[q] and not base_ok[q] for q in golds)
            m["right_to_wrong"] = sum(base_ok[q] and not ok[q] for q in golds)
            m["by_question_type"] = breakdown_by_type(pr, golds, questions)
            pct = 100 * len(gated_ids) / len(rows) if mode in ("gated", "oracle_gated") else (
                100.0 if mode == "rerank_all" else 0.0)
            m["pct_reranked"] = pct
            results[mode] = m
            w.writerow([args.tag, Path(args.candidates).name, mode, args.k, args.kind, args.tau,
                        f"{pct:.1f}", f"{m['exact_match']:.2f}", f"{m['f1']:.2f}",
                        m["wrong_to_right"], m["right_to_wrong"]])
            print(f"{mode:13s} EM={m['exact_match']:6.2f} F1={m['f1']:6.2f} reranked={pct:5.1f}% "
                  f"+{m['wrong_to_right']} / -{m['right_to_wrong']}")
            save_json(pr, out / f"predictions_{args.tag}_{mode}.json")

    save_json(results, out / f"metrics_{args.tag}.json")
    print(f"Saved metrics to {out / f'metrics_{args.tag}.json'}; appended rows to {abl_path}")


if __name__ == "__main__":
    main()
