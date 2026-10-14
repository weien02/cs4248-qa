"""Tokenisation into model features (sliding window) and n-best span extraction.

Long contexts are split into overlapping windows of `max_length` tokens with
`stride` tokens of overlap, so one example can produce several features.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _questions(examples):
    # some SQuAD questions have leading whitespace, which wastes tokens
    return [q.lstrip() for q in examples["question"]]


def prepare_train_features(examples, tokenizer, max_length=384, stride=128):
    enc = tokenizer(
        _questions(examples), examples["context"],
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets_all = enc.pop("offset_mapping")
    starts, ends = [], []
    for i, offsets in enumerate(offsets_all):
        seq_ids = enc.sequence_ids(i)
        ans = examples["answers"][sample_map[i]]
        a_start = ans["answer_start"][0]
        a_end = a_start + len(ans["text"][0])

        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        # answer not fully inside this window -> point at CLS (index 0)
        if offsets[ctx_start][0] > a_start or offsets[ctx_end][1] < a_end:
            starts.append(0)
            ends.append(0)
            continue
        s = ctx_start
        while s <= ctx_end and offsets[s][0] <= a_start:
            s += 1
        e = ctx_end
        while e >= ctx_start and offsets[e][1] >= a_end:
            e -= 1
        starts.append(s - 1)
        ends.append(e + 1)
    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return enc


def prepare_eval_features(examples, tokenizer, max_length=384, stride=128):
    enc = tokenizer(
        _questions(examples), examples["context"],
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["example_id"] = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["example_id"].append(examples["id"][sample_map[i]])
        # keep offsets only for context tokens so postprocessing can ignore the rest
        enc["offset_mapping"][i] = [
            o if seq_ids[k] == 1 else None for k, o in enumerate(enc["offset_mapping"][i])
        ]
    return enc


def extract_candidates(examples, features, start_logits, end_logits,
                       n_best=20, max_answer_length=30):
    """Return {example_id: [candidate, ...]} sorted by score (start+end logit).

    Each candidate: {text, start, end, score, start_logit, end_logit}, where
    start/end are character offsets into the context. Duplicate character spans
    across overlapping windows keep their highest score.
    """
    feats_per_ex = defaultdict(list)
    for fi, ex_id in enumerate(features["example_id"]):
        feats_per_ex[ex_id].append(fi)
    offsets_col = features["offset_mapping"]  # pull column once; row access on a Dataset is slow

    out = {}
    for ex in examples:
        ctx = ex["context"]
        best = {}
        for fi in feats_per_ex[ex["id"]]:
            sl, el = start_logits[fi], end_logits[fi]
            offsets = offsets_col[fi]
            s_idx = np.argsort(sl)[-1:-n_best - 1:-1]
            e_idx = np.argsort(el)[-1:-n_best - 1:-1]
            for s in s_idx:
                for e in e_idx:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    cs, ce = offsets[s][0], offsets[e][1]
                    score = float(sl[s] + el[e])
                    if (cs, ce) not in best or score > best[(cs, ce)]["score"]:
                        best[(cs, ce)] = {
                            "text": ctx[cs:ce], "start": int(cs), "end": int(ce), "score": score,
                            "start_logit": float(sl[s]), "end_logit": float(el[e]),
                        }
        cands = sorted(best.values(), key=lambda c: c["score"], reverse=True)[:n_best]
        out[ex["id"]] = cands or [{"text": "", "start": 0, "end": 0, "score": -1e9,
                                   "start_logit": 0.0, "end_logit": 0.0}]
    return out
