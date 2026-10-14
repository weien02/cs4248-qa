"""Dataset loading and deterministic splits.

Splits are made by *article title* so no passage appears in two splits.

    train (87.6k)  ->  heldout  (~5% of articles; tune gating threshold, validate reranker)
                   ->  train_main (rest)  ->  fold0 / fold1 (for cross-fitted hard negatives)
    dev            =  official SQuAD v1.1 validation set (final numbers only)

Any split name ending in .jsonl is loaded as a local SQuAD-format file
(e.g. a distractor-augmented dev set from scripts/make_distractors.py).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

SQUAD_HUB_ID = "rajpurkar/squad"
SPLIT_SEED = 13
HELDOUT_FRAC = 0.05
SPLIT_NAMES = ("train", "train_main", "heldout", "fold0", "fold1", "dev")


def _title_groups(train) -> dict[str, list[str]]:
    titles = sorted(set(train["title"]))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(titles)
    n_held = math.ceil(len(titles) * HELDOUT_FRAC)
    heldout = titles[:n_held]
    main = titles[n_held:]
    return {
        "heldout": heldout,
        "train_main": main,
        "fold0": main[0::2],
        "fold1": main[1::2],
    }


def load_split(name: str, max_samples: int | None = None):
    """Load a named split (see SPLIT_NAMES) or a local .jsonl file as a HF Dataset."""
    from datasets import load_dataset  # lazy: analysis scripts don't need it

    if name.endswith(".jsonl") or name.endswith(".json"):
        ds = load_dataset("json", data_files=name, split="train")
    else:
        if name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES} or a .jsonl path")
        squad = load_dataset(SQUAD_HUB_ID)
        if name == "dev":
            ds = squad["validation"]
        elif name == "train":
            ds = squad["train"]
        else:
            keep = set(_title_groups(squad["train"])[name])
            ds = squad["train"].filter(lambda ex: ex["title"] in keep)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
