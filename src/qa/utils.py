from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def save_run_config(args, out_dir: str | Path) -> None:
    """Record exact args + command line next to every output, for reproducibility."""
    cfg = dict(vars(args))
    cfg["_argv"] = " ".join(sys.argv)
    save_json(cfg, Path(out_dir) / "run_config.json")
