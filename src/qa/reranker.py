"""Second-stage reranker over the baseline's top-k candidate spans.

Each candidate is encoded as a cross-encoder input:
    question  [SEP]  candidate  [SEP]  sentence with the candidate marked by << >>
The [CLS] vector is concatenated with the candidate's *relative* baseline score
(score - top-1 score, so the top candidate is 0) and mapped to a single logit.
Candidates of one question form a group; training uses a listwise softmax
cross-entropy over the group with the gold candidate as target.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .metrics import exact_match, f1, max_over_gold
from .text import marked_sentence, sentence_spans

NEG_MODES = ("hard", "easy", "random")


# ---------------------------------------------------------------- instances
@dataclass
class Group:
    qid: str
    question: str
    texts: list[str]          # candidate texts
    sentences: list[str]      # marked context sentence per candidate
    rel_scores: list[float]   # baseline score minus top-1 score
    label: int                # index of gold candidate (-1 at inference / unknown)
    targets: list[float] | None = None  # soft target distribution (build_group soft_power)


def _random_span(context: str, golds: list[str], rng: random.Random, max_words=6):
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", context)]
    for _ in range(20):
        n = rng.randint(1, min(max_words, len(words)))
        i = rng.randint(0, len(words) - n)
        s, e = words[i][0], words[i + n - 1][1]
        if not max_over_gold(exact_match, context[s:e], golds):
            return s, e
    return None


def build_group(row: dict, k: int, neg_mode: str = "hard", rng: random.Random | None = None,
                train: bool = True, inject_gold: bool = False,
                soft_power: float | None = None) -> Group | None:
    """Turn one candidates.jsonl row into a reranking group.

    Training negatives (neg_mode):
      hard   - the other top-k baseline candidates (high-scoring mistakes)
      easy   - candidates sampled from baseline ranks k..n_best (low-scoring)
      random - random passage spans, given the lowest n-best score as their score
    At inference (train=False) the group is simply the top-k candidates.

    soft_power: if set, also attach soft targets: weight F1(candidate, gold) ** soft_power,
    1 for the gold candidate, normalised. Train questions have ONE reference answer, so
    near-miss spans (often acceptable to another annotator) are otherwise pure negatives.
    """
    rng = rng or random.Random(0)
    ctx, cands, golds = row["context"], row["candidates"], row["answers"]
    spans = sentence_spans(ctx)
    top = cands[0]["score"]
    is_gold = [bool(max_over_gold(exact_match, c["text"], golds)) for c in cands]

    if not train:
        chosen = cands[:k]
        label = next((i for i, c in enumerate(chosen) if is_gold[i]), -1)
    else:
        pos_idx = next((i for i in range(min(k, len(cands))) if is_gold[i]), None)
        if pos_idx is not None:
            pos = cands[pos_idx]
        elif inject_gold and row.get("answer_starts"):
            s = row["answer_starts"][0]
            pos = {"text": golds[0], "start": s, "end": s + len(golds[0]),
                   "score": min(c["score"] for c in cands[:k])}
        else:
            return None  # gold not reachable; skip question

        if neg_mode == "hard":
            negs = [c for i, c in enumerate(cands[:k]) if not is_gold[i]]
        elif neg_mode == "easy":
            pool = [c for i, c in enumerate(cands) if i >= k and not is_gold[i]]
            negs = rng.sample(pool, min(k - 1, len(pool)))
        elif neg_mode == "random":
            floor = cands[-1]["score"]
            negs = []
            for _ in range(k - 1):
                sp = _random_span(ctx, golds, rng)
                if sp:
                    negs.append({"text": ctx[sp[0]:sp[1]], "start": sp[0], "end": sp[1], "score": floor})
        else:
            raise ValueError(f"neg_mode must be one of {NEG_MODES}")
        negs = negs[:k - 1]
        if not negs:
            return None
        chosen = negs + [pos]
        rng.shuffle(chosen)
        label = chosen.index(pos)

    targets = None
    if soft_power is not None and label >= 0:
        w = [1.0 if i == label else max_over_gold(f1, c["text"], golds) ** soft_power
             for i, c in enumerate(chosen)]
        targets = [x / sum(w) for x in w]

    return Group(
        # SQuAD contains a question with a very large amount of leading
        # whitespace that otherwise dominates tokenisation.
        qid=row["id"], question=row["question"].strip(),
        texts=[c["text"] for c in chosen],
        sentences=[marked_sentence(ctx, c["start"], c["end"], spans) for c in chosen],
        rel_scores=[c["score"] - top for c in chosen],
        label=label,
        targets=targets,
    )


# ---------------------------------------------------------------- model
class SpanReranker(nn.Module):
    def __init__(self, encoder_name: str, use_score: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder_name = encoder_name
        self.use_score = use_score
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(h + int(use_score), 1)

    def forward(self, input_ids, attention_mask, rel_scores, group_mask, token_type_ids=None):
        """input_ids: (B*K, L); rel_scores, group_mask: (B, K). Returns logits (B, K)."""
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        cls = self.encoder(**kw).last_hidden_state[:, 0]
        x = self.dropout(cls)
        B, K = group_mask.shape
        if self.use_score:
            x = torch.cat([x, rel_scores.reshape(B * K, 1).to(x.dtype) / 10.0], dim=-1)
        logits = self.head(x).reshape(B, K).float()
        return logits.masked_fill(~group_mask, float("-inf"))

    def save(self, out_dir: str | Path, tokenizer=None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(out / "encoder")
        torch.save(self.head.state_dict(), out / "head.pt")
        (out / "reranker_config.json").write_text(
            json.dumps({"encoder_name": self.encoder_name, "use_score": self.use_score}, indent=2))

    @classmethod
    def load(cls, out_dir: str | Path):
        out = Path(out_dir)
        cfg = json.loads((out / "reranker_config.json").read_text())
        m = cls(str(out / "encoder"), use_score=cfg["use_score"])
        m.encoder_name = cfg["encoder_name"]
        m.head.load_state_dict(torch.load(out / "head.pt", map_location="cpu"))
        tok = AutoTokenizer.from_pretrained(out / "encoder")
        return m, tok


# ---------------------------------------------------------------- batching
def collate(groups: list[Group], tokenizer, max_length: int = 192) -> dict:
    K = max(len(g.texts) for g in groups)
    sep = f" {tokenizer.sep_token} " if tokenizer.sep_token else " | "
    firsts, seconds, mask, scores, labels, targets = [], [], [], [], [], []
    for g in groups:
        n = len(g.texts)
        for i in range(K):
            j = min(i, n - 1)  # pad by repeating the last candidate (masked out below)
            firsts.append(g.question)
            seconds.append(g.texts[j] + sep + g.sentences[j])
        mask.append([True] * n + [False] * (K - n))
        scores.append(g.rel_scores + [0.0] * (K - n))
        labels.append(g.label)
        t = g.targets or [float(i == g.label) for i in range(n)]  # one-hot unless soft
        targets.append(t + [0.0] * (K - n))
    # Usually the candidate/sentence side is the longer sequence, but a few
    # SQuAD questions are long enough that truncating only the second sequence
    # cannot satisfy max_length. Longest-first handles those outliers while
    # retaining as much of both sides as possible.
    enc = tokenizer(firsts, seconds, truncation="longest_first", max_length=max_length,
                    padding=True, return_tensors="pt")
    batch = dict(enc)
    batch["group_mask"] = torch.tensor(mask)
    batch["rel_scores"] = torch.tensor(scores, dtype=torch.float)
    batch["labels"] = torch.tensor(labels)
    batch["targets"] = torch.tensor(targets, dtype=torch.float)
    return batch


@torch.no_grad()
def score_groups(model, tokenizer, groups, device, batch_size=16, max_length=192, amp_dtype=None):
    """Return a list (per group) of reranker logits."""
    model.eval().to(device)
    out = []
    for i in range(0, len(groups), batch_size):
        chunk = groups[i:i + batch_size]
        b = collate(chunk, tokenizer, max_length)
        b.pop("labels")
        b.pop("targets")
        b = {k: v.to(device) for k, v in b.items()}
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(**b)
        else:
            logits = model(**b)
        for g, row in zip(chunk, logits.cpu().tolist()):
            out.append(row[:len(g.texts)])
    return out
