"""Inference helpers for the span-prediction baseline."""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .features import extract_candidates, prepare_eval_features

MODEL_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


@torch.no_grad()
def predict_logits(model, features, device, batch_size=64, amp_dtype=None):
    """Run the QA model over eval features; return (start_logits, end_logits) as numpy."""
    model.eval().to(device)
    cols = [c for c in MODEL_INPUTS if c in features.column_names]
    feats = features.with_format("torch", columns=cols)
    starts, ends = [], []
    for i in tqdm(range(0, len(feats), batch_size), desc="predict"):
        batch = {k: v.to(device) for k, v in feats[i:i + batch_size].items()}
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
    return np.concatenate(starts), np.concatenate(ends)


def candidates_for_split(model, tokenizer, examples, device, *, max_length=384, stride=128,
                         n_best=20, max_answer_length=30, batch_size=64, amp_dtype=None):
    """Tokenise examples, run the model, and return {id: sorted candidate list}."""
    features = examples.map(
        prepare_eval_features, batched=True, remove_columns=examples.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "stride": stride},
        desc="tokenise",
    )
    sl, el = predict_logits(model, features, device, batch_size, amp_dtype)
    return extract_candidates(examples, features, sl, el, n_best, max_answer_length)
