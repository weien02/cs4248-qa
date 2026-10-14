"""Train the second-stage span reranker on baseline candidates.

Train on CROSS-FITTED candidates (fold0 predicted by the fold1 model and vice versa),
otherwise the baseline is overconfident on its own training data and you get few
real hard negatives. Validate on heldout candidates.

Example
-------
python scripts/train_reranker.py \
    --train_candidates data/candidates/fold0.jsonl data/candidates/fold1.jsonl \
    --val_candidates data/candidates/heldout.jsonl \
    --encoder roberta-base --neg_mode hard --k 5 --bf16 \
    --output_dir out/reranker-hard
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from qa.data import read_jsonl  # noqa: E402
from qa.metrics import score_predictions  # noqa: E402
from qa.reranker import NEG_MODES, SpanReranker, build_group, collate, score_groups  # noqa: E402
from qa.utils import get_device, save_json, save_run_config, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_candidates", nargs="+", required=True)
    p.add_argument("--val_candidates", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder", default="roberta-base")
    p.add_argument("--neg_mode", choices=NEG_MODES, default="hard")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no_score", action="store_true", help="drop the baseline-score feature")
    p.add_argument("--inject_gold", action="store_true",
                   help="if gold is not in top-k, add it as the positive instead of skipping")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8, help="questions per batch (x k sequences)")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max_train_groups", type=int)
    p.add_argument("--max_val_rows", type=int)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(model, tok, val_rows, k, device, args, amp):
    """Rerank every val question's top-k; return EM/F1 and group accuracy."""
    groups = [build_group(r, k, train=False) for r in val_rows]
    logits = score_groups(model, tok, groups, device, batch_size=args.batch_size * 2,
                          max_length=args.max_length, amp_dtype=amp)
    preds, hits, n_lab = {}, 0, 0
    for g, lg in zip(groups, logits):
        best = max(range(len(lg)), key=lg.__getitem__)
        preds[g.qid] = g.texts[best]
        if g.label >= 0:
            n_lab += 1
            hits += int(best == g.label)
    m = score_predictions(preds, {r["id"]: r["answers"] for r in val_rows})
    m["group_acc_when_gold_in_topk"] = 100 * hits / max(n_lab, 1)
    return m


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_run_config(args, out)
    device = get_device()
    amp = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    rows = [r for f in args.train_candidates for r in read_jsonl(f)]
    groups = [build_group(r, args.k, args.neg_mode, rng, train=True, inject_gold=args.inject_gold)
              for r in rows]
    skipped = sum(g is None for g in groups)
    groups = [g for g in groups if g is not None]
    if args.max_train_groups:
        groups = groups[:args.max_train_groups]
    print(f"{len(rows)} train questions -> {len(groups)} groups ({skipped} skipped: gold not in top-{args.k})")

    val_rows = read_jsonl(args.val_candidates)[:args.max_val_rows]
    base = score_predictions({r["id"]: r["candidates"][0]["text"] for r in val_rows},
                             {r["id"]: r["answers"] for r in val_rows})
    print(f"val baseline top-1: EM={base['exact_match']:.2f} F1={base['f1']:.2f}")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = SpanReranker(args.encoder, use_score=not args.no_score).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(groups) / args.batch_size / args.grad_accum)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type == "cuda"))

    history, best_em = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(groups)
        t0, running = time.time(), 0.0
        n_batches = math.ceil(len(groups) / args.batch_size)
        for bi in range(n_batches):
            chunk = groups[bi * args.batch_size:(bi + 1) * args.batch_size]
            b = {k: v.to(device) for k, v in collate(chunk, tok, args.max_length).items()}
            labels = b.pop("labels")
            ctx = (torch.autocast("cuda", dtype=amp) if amp is not None and device.type == "cuda"
                   else contextlib.nullcontext())
            with ctx:
                logits = model(**b)
            loss = F.cross_entropy(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == n_batches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += loss.item() * args.grad_accum
            if (bi + 1) % 100 == 0:
                print(f"epoch {epoch} step {bi + 1}/{n_batches} loss {running / 100:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                running = 0.0

        m = evaluate(model, tok, val_rows, args.k, device, args, amp)
        m["epoch"] = epoch
        history.append(m)
        print(f"[val epoch {epoch}] rerank-all EM={m['exact_match']:.2f} F1={m['f1']:.2f} "
              f"group_acc={m['group_acc_when_gold_in_topk']:.2f}")
        if m["exact_match"] > best_em:
            best_em = m["exact_match"]
            model.save(out / "best", tok)
            print(f"  saved best -> {out / 'best'}")

    save_json({"val_baseline": base, "history": history}, out / "train_log.json")


if __name__ == "__main__":
    main()
