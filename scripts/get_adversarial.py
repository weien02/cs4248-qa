"""Download Adversarial SQuAD (Jia & Liang, 2017) as flat JSONL files for load_split().

    data/adversarial/addsent.jsonl      3560 rows (1000 original dev questions + 2560 adversarial)
    data/adversarial/addonesent.jsonl   1787 rows (1000 original + 787 adversarial)

Adversarial ids are "<original id>-<suffix>"; their context is the original passage with
one misleading sentence appended. Score them with scripts/robustness_report.py, which
takes the worst case over each original question's versions (the official protocol).

The original CodaLab links are unreliable, so this reads Hugging Face's parquet conversion
of stanfordnlp/squad_adversarial. Needs internet: run on your Mac or xlogin, not in a GPU job.

Example
-------
python scripts/get_adversarial.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.data import PROJECT_ROOT, write_jsonl  # noqa: E402

HF_PARQUET = ("hf://datasets/stanfordnlp/squad_adversarial@refs%2Fconvert%2Fparquet/"
              "{cfg}/squad_adversarial-validation.parquet")
CONFIGS = {"AddSent": "addsent", "AddOneSent": "addonesent"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "data" / "adversarial"))
    args = p.parse_args()

    from datasets import load_dataset

    for cfg, name in CONFIGS.items():
        ds = load_dataset("parquet", data_files=HF_PARQUET.format(cfg=cfg), split="train")
        rows = [{"id": r["id"], "title": r["title"], "context": r["context"], "question": r["question"],
                 "answers": {"text": r["answers"]["text"],
                             "answer_start": [int(s) for s in r["answers"]["answer_start"]]}}
                for r in ds]
        out = Path(args.out_dir) / f"{name}.jsonl"
        write_jsonl(out, rows)
        n_adv = sum("-" in r["id"] for r in rows)
        print(f"{cfg}: wrote {len(rows)} rows ({n_adv} adversarial) to {out}")


if __name__ == "__main__":
    main()
