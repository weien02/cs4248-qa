"""Fine-tune a transformer to predict answer start/end positions (the baseline).

Examples
--------
# Mac smoke test (minutes)
python scripts/train_baseline.py --model distilbert-base-uncased \
    --train_split train_main --max_train_samples 2000 --max_eval_samples 500 \
    --epochs 1 --output_dir out/debug

# Full baseline (GPU)
python scripts/train_baseline.py --model roberta-base --train_split train_main \
    --epochs 2 --batch_size 32 --bf16 --output_dir out/baseline-roberta-base
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,  # noqa: E402
                          TrainingArguments, default_data_collator)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from qa.baseline import candidates_for_split  # noqa: E402
from qa.data import load_split  # noqa: E402
from qa.features import prepare_train_features  # noqa: E402
from qa.metrics import score_predictions  # noqa: E402
from qa.utils import get_device, save_json, save_run_config, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--train_split", default="train_main",
                   help="train | train_main | fold0 | fold1 | path.jsonl")
    p.add_argument("--eval_split", default="dev", help="split to score after training ('none' to skip)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", help="use on A100/H100")
    p.add_argument("--fp16", action="store_true", help="use on V100/T4")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--max_train_samples", type=int)
    p.add_argument("--max_eval_samples", type=int)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_run_config(args, out)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    train_ex = load_split(args.train_split, args.max_train_samples)
    train_feats = train_ex.map(
        prepare_train_features, batched=True, remove_columns=train_ex.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length, "stride": args.stride},
        desc="tokenise train",
    )
    print(f"{len(train_ex)} train examples -> {len(train_feats)} features")

    # transformers 5.x removed warmup_ratio; a float warmup_steps < 1 is now read as a ratio
    warmup = ({"warmup_ratio": args.warmup_ratio}
              if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
              else {"warmup_steps": args.warmup_ratio})
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        **warmup,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_feats,
                      data_collator=default_data_collator)

    # resume automatically if a previous job hit the time limit
    last_ckpt = get_last_checkpoint(targs.output_dir) if Path(targs.output_dir).exists() else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"Saved model to {final}")

    if args.eval_split != "none":
        import torch
        ev = load_split(args.eval_split, args.max_eval_samples)
        amp = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        cands = candidates_for_split(trainer.model, tokenizer, ev, get_device(),
                                     max_length=args.max_length, stride=args.stride, amp_dtype=amp)
        preds = {qid: c[0]["text"] for qid, c in cands.items()}
        golds = {ex["id"]: ex["answers"]["text"] for ex in ev}
        metrics = score_predictions(preds, golds)
        print(f"[{args.eval_split}] EM={metrics['exact_match']:.2f} F1={metrics['f1']:.2f}")
        save_json(metrics, out / f"metrics_{Path(args.eval_split).stem}.json")
        save_json(preds, out / f"predictions_{Path(args.eval_split).stem}.json")


if __name__ == "__main__":
    main()
