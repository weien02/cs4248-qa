"""Lightweight sentence helpers (no torch dependency)."""
from __future__ import annotations

import re

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sentence_spans(context: str) -> list[tuple[int, int]]:
    """Character (start, end) spans of sentences, using a simple punctuation rule."""
    spans, start = [], 0
    for m in _SENT_END.finditer(context):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(context)))
    return spans


def marked_sentence(context: str, start: int, end: int, spans=None) -> str:
    """Sentence(s) covering [start, end) with the candidate wrapped in << >>."""
    spans = spans or sentence_spans(context)
    s0 = max((s for s, e in spans if s <= start), default=0)
    e0 = min((e for s, e in spans if e >= end), default=len(context))
    return f"{context[s0:start]}<< {context[start:end]} >>{context[end:e0]}".strip()
