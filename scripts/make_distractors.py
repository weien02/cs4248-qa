"""Create a distractor-augmented copy of a split for robustness testing.

For every question, one sentence from a DIFFERENT article is inserted into its
passage at a sentence boundary. Answer offsets are shifted so gold answers stay valid.

  --mode random  : a random sentence from another article (easy distractor)
  --mode overlap : the sentence from another article with the most word overlap
                   with the question (harder; similar in spirit to AddSent,
                   Jia & Liang 2017, but without rewriting the sentence)

Sentences containing any gold answer string are never used.

Example
-------
python scripts/make_distractors.py --split dev --mode overlap --position random \
    --output data/distractors/dev_overlap.jsonl
# then dump candidates on it and run run_system.py as usual
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.data import load_split, write_jsonl  # noqa: E402
from qa.metrics import normalize_answer  # noqa: E402
from qa.text import sentence_spans  # noqa: E402

STOP = set("""a an the of in on at to for from by with and or but is are was were be been being
what which who whom whose when where why how many much does do did that this these those it its
as into than then there their they he she his her has have had not no can could would should
""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev")
    p.add_argument("--mode", choices=["random", "overlap"], default="overlap")
    p.add_argument("--position", choices=["end", "start", "random"], default="random")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_samples", type=int)
    args = p.parse_args()
    rng = random.Random(args.seed)

    full = load_split(args.split)
    ds = full.select(range(min(args.max_samples, len(full)))) if args.max_samples else full

    # sentence pool from all distinct passages of the whole split (not just the first
    # max_samples questions, which may all come from one article)
    pool, seen = [], set()
    for ex in full:
        if ex["context"] in seen:
            continue
        seen.add(ex["context"])
        for s, e in sentence_spans(ex["context"]):
            sent = ex["context"][s:e].strip()
            if 5 <= len(sent.split()) <= 60:
                pool.append((ex["title"], sent))
    index = defaultdict(list)  # word -> sentence ids
    pool_words = []
    for i, (_, sent) in enumerate(pool):
        ws = content_words(sent)
        pool_words.append(ws)
        for w in ws:
            index[w].append(i)
    print(f"{len(pool)} candidate distractor sentences")

    out_rows, n_fail = [], 0
    for ex in ds:
        golds = [normalize_answer(t) for t in ex["answers"]["text"]]

        def ok(i):
            title, sent = pool[i]
            ns = normalize_answer(sent)
            return title != ex["title"] and not any(g and g in ns for g in golds)

        choice = None
        if args.mode == "overlap":
            counts = Counter()
            for w in content_words(ex["question"]):
                for i in index.get(w, ()):
                    counts[i] += 1
            for i, _ in counts.most_common(50):
                if ok(i):
                    choice = i
                    break
        if choice is None:  # random mode, or no overlapping sentence found
            for _ in range(100):
                i = rng.randrange(len(pool))
                if ok(i):
                    choice = i
                    break
        if choice is None:
            n_fail += 1
            continue
        sent = pool[choice][1]

        ctx = ex["context"]
        bounds = [0] + [e for _, e in sentence_spans(ctx)]
        at = {"end": len(ctx), "start": 0, "random": rng.choice(bounds)}[args.position]
        if at == 0:
            new_ctx, shift = sent + " " + ctx, len(sent) + 1
        else:
            new_ctx, shift = ctx[:at] + " " + sent + ctx[at:], len(sent) + 1
        starts = [s + shift if s >= at else s for s in ex["answers"]["answer_start"]]
        for t, s in zip(ex["answers"]["text"], starts):
            assert new_ctx[s:s + len(t)] == t, (ex["id"], t)

        out_rows.append({
            "id": ex["id"], "title": ex["title"], "question": ex["question"], "context": new_ctx,
            "answers": {"text": ex["answers"]["text"], "answer_start": starts},
            "distractor": sent, "distractor_pos": at,
        })

    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} examples to {args.output} ({n_fail} skipped)")


if __name__ == "__main__":
    main()
