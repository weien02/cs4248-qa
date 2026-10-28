"""Error analysis of the baseline and the gated system on one candidates file (normally dev).

Reads the candidates from dump_candidates.py and the predictions run_system.py saved for
--tag, then writes to --out_dir:

  summary.md         all tables below, plus sampled examples to quote in the report
  error_types.csv    where errors come from:
                       correct      exact match
                       boundary     overlaps the gold answer (F1 > 0) but not exact
                       ranking      no overlap, but gold is in the baseline top-k (rerankable)
                       recall_k     gold only at ranks k..n_best (k too small)
                       recall_all   gold not in the n-best list at all (no reranker can fix it)
  gating.csv         does the confidence gate catch the rerankable errors?
  by_*.csv           EM per question type / gold answer length / confidence decile for
                     baseline, gated, rerank_all and oracle_gated, with wrong->right and
                     right->wrong flips of the gated system
  calibration.png    baseline EM and reranker gain per confidence decile
  flips.csv          gated system's wrong->right / right->wrong flips: was the wrong side a
                     near-miss (overlaps the gold answer) or an unrelated span?
  negatives.csv      are the reranker's training negatives really wrong? Training data has one
                     reference answer per question, so a candidate can be labelled negative while
                     a second annotator would accept it. On dev (several references) this
                     relabels with the FIRST reference only, as training does, and counts top-k
                     ("hard") and rank k..n_best ("easy") negatives that match ANOTHER reference.
                     --train_candidates adds the same overlap breakdown for the real training
                     candidates (fold0/fold1), where only one reference exists.

Example
-------
python scripts/error_analysis.py --candidates data/candidates/dev.jsonl --results_dir results/dev \
    --tag hard --tau_from data/candidates/heldout.jsonl --gate_frac 0.2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from qa.confidence import MARGIN_KINDS, confidence  # noqa: E402
from qa.data import read_jsonl  # noqa: E402
from qa.metrics import exact_match, f1, max_over_gold, normalize_answer, question_type  # noqa: E402

MODES = ("baseline", "gated", "rerank_all", "oracle_gated")
ERROR_TYPES = ("correct", "boundary", "ranking", "recall_k", "recall_all")
NEG_TYPES = ("matches_other_ref", "contains_gold", "inside_gold", "partial_overlap", "no_overlap")


def error_type(row, pred, k):
    if max_over_gold(exact_match, pred, row["answers"]):
        return "correct"
    if max_over_gold(f1, pred, row["answers"]) > 0:
        return "boundary"
    if 0 <= row["gold_rank"] < k:
        return "ranking"
    return "recall_k" if row["gold_rank"] >= k else "recall_all"


def negative_type(text, ref, other_refs):
    """How a candidate labelled negative (not an exact match of ref) relates to the answer."""
    if any(exact_match(text, o) for o in other_refs):
        return "matches_other_ref"
    t, g = normalize_answer(text), normalize_answer(ref)
    if g and g in t:
        return "contains_gold"
    if t and t in g:
        return "inside_gold"
    return "partial_overlap" if f1(text, ref) > 0 else "no_overlap"


def negative_audit(rows, k):
    """Label negatives the way build_group() does, against the first reference only.

    Questions whose first reference isn't in the top-k are skipped, as in training.
    Returns {"hard": Counter, "easy": Counter} of negative_type over all negatives.
    """
    counts = {"hard": Counter(), "easy": Counter()}
    for r in rows:
        if not r["answers"]:
            continue
        ref, others = r["answers"][0], r["answers"][1:]
        cands = r["candidates"]
        if not any(exact_match(c["text"], ref) for c in cands[:k]):
            continue
        for i, c in enumerate(cands):
            if not exact_match(c["text"], ref):
                counts["hard" if i < k else "easy"][negative_type(c["text"], ref, others)] += 1
    return counts


def length_bucket(row):
    n = min(len(a.split()) for a in row["answers"]) if row["answers"] else 0
    return {0: "0", 1: "1", 2: "2", 3: "3"}.get(n, "4-5" if n <= 5 else "6+")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(out + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True)
    p.add_argument("--results_dir", default="results/dev", help="where run_system.py wrote predictions")
    p.add_argument("--tag", required=True, help="the --tag given to run_system.py, e.g. hard")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--kind", choices=MARGIN_KINDS, default="prob_margin")
    p.add_argument("--tau", type=float)
    p.add_argument("--tau_from", help="heldout candidates to derive tau from, as in run_system.py")
    p.add_argument("--gate_frac", type=float, default=0.2)
    p.add_argument("--n_examples", type=int, default=10, help="examples per category in summary.md")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_candidates", nargs="*", default=[],
                   help="reranker training candidates (fold0/fold1.jsonl) for the negatives audit")
    p.add_argument("--out_dir", help="default: results/error_analysis/<tag>")
    args = p.parse_args()
    out = Path(args.out_dir or f"results/error_analysis/{args.tag}")
    out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.candidates)
    res = Path(args.results_dir)
    preds = {}
    for mode in MODES:
        f = res / f"predictions_{args.tag}_{mode}.json"
        if f.exists():
            preds[mode] = json.loads(f.read_text())
    if "gated" not in preds:
        sys.exit(f"{res}/predictions_{args.tag}_gated.json not found: run run_system.py with --reranker first")
    preds["baseline"] = {r["id"]: r["candidates"][0]["text"] for r in rows}

    if args.tau is None:
        if not args.tau_from:
            p.error("give --tau, or --tau_from heldout.jsonl with --gate_frac (same as run_system.py)")
        held = [confidence(r["candidates"], args.k, args.kind) for r in read_jsonl(args.tau_from)]
        args.tau = float(np.quantile([c for c in held if np.isfinite(c)], args.gate_frac))
    conf = {r["id"]: confidence(r["candidates"], args.k, args.kind) for r in rows}
    gated = {qid for qid, c in conf.items() if c < args.tau}

    ok = {m: {r["id"]: max_over_gold(exact_match, pr.get(r["id"], ""), r["answers"]) for r in rows}
          for m, pr in preds.items()}
    n = len(rows)
    by_id = {r["id"]: r for r in rows}
    md = [f"# Error analysis: `{args.tag}` on `{Path(args.candidates).name}`",
          f"{n} questions, k={args.k}, {args.kind} tau={args.tau:.4f}, "
          f"{100 * len(gated) / n:.1f}% sent to the reranker.\n"]

    # 1. error types
    types = {m: Counter(error_type(r, preds[m].get(r["id"], ""), args.k) for r in rows)
             for m in ("baseline", "gated", "rerank_all") if m in preds}
    et_rows = [{"error_type": t, **{m: f"{100 * c[t] / n:.2f}" for m, c in types.items()}}
               for t in ERROR_TYPES]
    write_csv(out / "error_types.csv", et_rows)
    md += ["## Where errors come from (% of all questions)", md_table(et_rows),
           "`ranking` is what a reranker can fix; `recall_*` needs a better baseline or larger k.\n"]

    # 2. does the gate catch the rerankable errors?
    base_type = {r["id"]: error_type(r, preds["baseline"][r["id"]], args.k) for r in rows}
    gate_rows = []
    for t in ERROR_TYPES:
        ids = [q for q, bt in base_type.items() if bt == t]
        g = sum(q in gated for q in ids)
        gate_rows.append({"baseline_outcome": t, "n": len(ids), "gated": g,
                          "pct_gated": f"{100 * g / max(len(ids), 1):.1f}",
                          "gated_and_correct_after": int(sum(q in gated and ok["gated"][q] for q in ids))})
    write_csv(out / "gating.csv", gate_rows)
    md += ["## Does the confidence gate catch the right questions?", md_table(gate_rows),
           "Ideally `ranking` errors have a high `pct_gated` and `correct` a low one.\n"]

    # 3. breakdowns
    def breakdown(key_fn, name, order=None):
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r["id"])
        table = []
        for key in order or sorted(groups):
            ids = groups.get(key, [])
            if not ids:
                continue
            row = {name: key, "n": len(ids)}
            for m in MODES:
                if m in ok:
                    row[f"EM_{m}"] = f"{100 * sum(ok[m][q] for q in ids) / len(ids):.2f}"
            row["wrong_to_right"] = int(sum(ok["gated"][q] and not ok["baseline"][q] for q in ids))
            row["right_to_wrong"] = int(sum(ok["baseline"][q] and not ok["gated"][q] for q in ids))
            row["pct_gated"] = f"{100 * sum(q in gated for q in ids) / len(ids):.1f}"
            table.append(row)
        write_csv(out / f"by_{name}.csv", table)
        return table

    qtype = breakdown(lambda r: question_type(r["question"]), "question_type")
    length = breakdown(length_bucket, "answer_words", ["0", "1", "2", "3", "4-5", "6+"])
    finite = np.array([c for c in conf.values() if np.isfinite(c)])
    edges = np.quantile(finite, np.linspace(0, 1, 11)[1:-1])
    decile = breakdown(lambda r: int(np.searchsorted(edges, conf[r["id"]], side="right")) + 1,
                       "confidence_decile")
    md += ["## By question type", md_table(qtype), "", "## By gold answer length (words)",
           md_table(length), "", "## By baseline confidence decile (1 = least confident)",
           md_table(decile), ""]

    # 4. what kind of answers do the flips swap?
    flip_rows = []
    for flip, cond, wrong_mode in (
            ("wrong_to_right", lambda q: ok["gated"][q] and not ok["baseline"][q], "baseline"),
            ("right_to_wrong", lambda q: ok["baseline"][q] and not ok["gated"][q], "gated")):
        ids = [r["id"] for r in rows if cond(r["id"])]
        kinds = Counter("near_miss" if max_over_gold(f1, preds[wrong_mode][q], by_id[q]["answers"]) > 0
                        else "unrelated" for q in ids)
        flip_rows.append({"flip": flip, "n": len(ids), "wrong_side": wrong_mode,
                          "near_miss": kinds["near_miss"], "unrelated": kinds["unrelated"],
                          "pct_near_miss": f"{100 * kinds['near_miss'] / max(len(ids), 1):.1f}"})
    write_csv(out / "flips.csv", flip_rows)
    md += ["## What do the reranker's flips swap? (gated system)", md_table(flip_rows),
           "`near_miss`: the wrong answer overlaps a reference (span boundary off). "
           "`unrelated`: no overlap.\n"]

    # 5. are the training negatives really wrong?
    neg_rows = []
    sources = [(Path(args.candidates).name + " (first ref only)", rows)]
    if args.train_candidates:
        sources.append(("+".join(Path(f).name for f in args.train_candidates),
                        [r for f in args.train_candidates for r in read_jsonl(f)]))
    for name, src in sources:
        for kind, c in negative_audit(src, args.k).items():
            total = sum(c.values())
            neg_rows.append({"source": name, "negatives": f"{kind} ({'top-k' if kind == 'hard' else 'rank k+'})",
                             "n": total, **{t: f"{100 * c[t] / max(total, 1):.1f}" for t in NEG_TYPES}})
    write_csv(out / "negatives.csv", neg_rows)
    md += ["## Are the reranker's training negatives really wrong? (% of negatives)", md_table(neg_rows),
           "Negatives are labelled as in training: not an exact match of ONE reference answer. "
           "`matches_other_ref` = another annotator gave exactly this answer (a false negative; only "
           "measurable on dev). `contains_gold` / `inside_gold` = a longer / shorter span around the "
           "answer, often also acceptable.\n"]

    # 6. calibration plot: where does reranking help?
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [d["confidence_decile"] for d in decile]
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, style in (("baseline", "o-"), ("rerank_all", "s--"), ("oracle_gated", "^:")):
            if f"EM_{m}" in decile[0]:
                ax.plot(x, [float(d[f"EM_{m}"]) for d in decile], style, label=m)
        tau_decile = int(np.searchsorted(edges, args.tau, side="right")) + 0.5
        ax.axvline(tau_decile, c="grey", ls="--", lw=0.8, label="tau")
        ax.set_xlabel("baseline confidence decile (1 = least confident)")
        ax.set_ylabel("EM")
        ax.set_xticks(x)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "calibration.png", dpi=150)
    except ImportError:
        pass

    # 7. examples
    rng = random.Random(args.seed)
    cats = {
        "Fixed by the reranker (wrong -> right)":
            [q for q in gated if ok["gated"][q] and not ok["baseline"][q]],
        "Broken by the reranker (right -> wrong)":
            [q for q in gated if ok["baseline"][q] and not ok["gated"][q]],
        "Confidently wrong (not gated, gold in top-k)":
            [q for q in by_id if q not in gated and base_type[q] == "ranking"],
        "Gold missing from the n-best list":
            [q for q in by_id if base_type[q] == "recall_all"],
    }
    md.append("## Examples")
    for title, ids in cats.items():
        md.append(f"### {title} ({len(ids)} total)")
        for q in rng.sample(sorted(ids), min(args.n_examples, len(ids))):
            r = by_id[q]
            top = "; ".join(f"{i + 1}. {c['text']!r} ({c['score']:.1f})"
                            for i, c in enumerate(r["candidates"][:args.k]))
            md += [f"- **Q:** {r['question']}  ",
                   f"  **gold:** {r['answers'][0]!r} | **baseline:** {preds['baseline'][q]!r} | "
                   f"**gated:** {preds['gated'].get(q, '')!r} | conf={conf[q]:.3f} | "
                   f"type={question_type(r['question'])}  ",
                   f"  **top-{args.k}:** {top}"]
        md.append("")

    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:2]))
    print(md_table(et_rows))
    print(md_table(neg_rows))
    print(f"\nWrote {out}/summary.md, error_types.csv, gating.csv, by_*.csv, flips.csv, "
          f"negatives.csv, calibration.png")


if __name__ == "__main__":
    main()
