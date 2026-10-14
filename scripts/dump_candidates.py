"""Run a trained baseline on a split and save the n-best candidate spans per question.

Output: one JSON line per question
    {id, title, question, context, answers, answer_starts,
     candidates: [{text, start, end, score, start_logit, end_logit}, ...],   # sorted by score
     gold_rank}   # index of first candidate exactly matching gold, -1 if none

Example
-------
python scripts/dump_candidates.py --model_dir out/baseline-roberta-base/final \
    --split dev --output data/candidates/dev.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: E402

from qa.baseline import candidates_for_split  # noqa: E402
from qa.data import load_split, write_jsonl  # noqa: E402
from qa.metrics import exact_match, max_over_gold, score_predictions  # noqa: E402
from qa.utils import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--split", required=True, help="dev | heldout | fold0 | fold1 | train_main | path.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max_samples", type=int)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    ex = load_split(args.split, args.max_samples)
    amp = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    cands = candidates_for_split(model, tok, ex, get_device(), max_length=args.max_length,
                                 stride=args.stride, n_best=args.n_best,
                                 max_answer_length=args.max_answer_length,
                                 batch_size=args.batch_size, amp_dtype=amp)

    rows, in_topk = [], {1: 0, 5: 0, 10: 0, 20: 0}
    for e in ex:
        cs = cands[e["id"]]
        golds = e["answers"]["text"]
        gold_rank = next((i for i, c in enumerate(cs) if max_over_gold(exact_match, c["text"], golds)), -1)
        for k in in_topk:
            in_topk[k] += int(0 <= gold_rank < k)
        rows.append({
            "id": e["id"], "title": e["title"], "question": e["question"], "context": e["context"],
            "answers": golds, "answer_starts": e["answers"]["answer_start"],
            "candidates": cs, "gold_rank": gold_rank,
        })
    write_jsonl(args.output, rows)

    preds = {r["id"]: r["candidates"][0]["text"] for r in rows}
    m = score_predictions(preds, {r["id"]: r["answers"] for r in rows})
    n = len(rows)
    print(f"Wrote {n} rows to {args.output}")
    print(f"top-1 EM={m['exact_match']:.2f} F1={m['f1']:.2f}")
    print("gold (exact match) within top-k: " +
          ", ".join(f"@{k}={100 * v / n:.1f}%" for k, v in in_topk.items()))


if __name__ == "__main__":
    main()
