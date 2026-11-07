"""Cache pretrained model/tokenizer files before offline cluster jobs.

Run on an internet-enabled compute node, not xlogin. Example:

python scripts/cache_models.py microsoft/deberta-v3-base
"""
from __future__ import annotations

import argparse

from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="+")
    args = p.parse_args()

    for model_name in args.models:
        print(f"Caching {model_name}", flush=True)
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Model cache ready")


if __name__ == "__main__":
    main()
