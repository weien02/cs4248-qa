"""Fast tests that need no model download: pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.confidence import confidence  # noqa: E402
from qa.data import _squad_rows  # noqa: E402
from qa.features import extract_candidates  # noqa: E402
from qa.metrics import f1, normalize_answer, question_type, score_predictions  # noqa: E402
from qa.text import marked_sentence, sentence_spans  # noqa: E402


def test_squad_normalisation_and_scores():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert f1("eiffel tower in paris", "the Eiffel Tower") == 2 * (2 / 4) * 1 / (2 / 4 + 1)
    m = score_predictions({"a": "Paris", "b": "x"}, {"a": ["paris", "Paris, France"], "b": ["y"]})
    assert m["exact_match"] == 50.0


def test_question_type():
    assert question_type("How many people live there?") == "how many"
    assert question_type("Whose idea was it?") == "who"
    assert question_type("In what year did it open?") == "what"
    assert question_type("Name the river.") == "other"


def test_confidence():
    c = [{"score": 10.0}, {"score": 10.0}, {"score": 0.0}]
    assert abs(confidence(c, 5, "prob_margin")) < 1e-9
    assert confidence(c, 5, "score_margin") == 0.0
    assert confidence([{"score": 1.0}], 5) == float("inf")


def test_sentences_and_marking():
    ctx = "Paris is in France. It has the Eiffel Tower. Built in 1889."
    spans = sentence_spans(ctx)
    assert [ctx[s:e] for s, e in spans] == ["Paris is in France.", "It has the Eiffel Tower.", "Built in 1889."]
    s = ctx.index("Eiffel Tower")
    assert marked_sentence(ctx, s, s + 12) == "It has the << Eiffel Tower >>."


def test_extract_candidates_dedup_and_offsets():
    ctx = "alpha beta gamma"
    examples = [{"id": "q1", "context": ctx}]
    # two overlapping windows over the same 3 context tokens (+ CLS with None offset)
    offs = [None, (0, 5), (6, 10), (11, 16)]
    features = {"example_id": ["q1", "q1"], "offset_mapping": [offs, offs]}
    sl = np.array([[0, 1, 5, 0], [0, 1, 6, 0]], dtype=float)
    el = np.array([[0, 0, 4, 2], [0, 0, 4, 2]], dtype=float)
    out = extract_candidates(examples, features, sl, el, n_best=5)["q1"]
    assert out[0]["text"] == "beta" and out[0]["score"] == 10.0  # max over windows kept
    assert len({(c["start"], c["end"]) for c in out}) == len(out)  # no duplicate spans
    assert all(ctx[c["start"]:c["end"]] == c["text"] for c in out)
    assert all(c["end"] >= c["start"] for c in out)


def test_original_squad_json_is_flattened(tmp_path):
    path = tmp_path / "train-v1.1.json"
    path.write_text(json.dumps({
        "version": "1.1",
        "data": [{
            "title": "Example",
            "paragraphs": [{
                "context": "Paris is in France.",
                "qas": [{
                    "id": "q1",
                    "question": "Where is Paris?",
                    "answers": [
                        {"text": "France", "answer_start": 12},
                        {"text": "France", "answer_start": 12},
                    ],
                }],
            }],
        }],
    }))

    assert _squad_rows(path) == [{
        "id": "q1",
        "title": "Example",
        "context": "Paris is in France.",
        "question": "Where is Paris?",
        "answers": {"text": ["France", "France"], "answer_start": [12, 12]},
    }]


def test_reranker_soft_targets():
    import random

    import torch
    import torch.nn.functional as F

    from qa.reranker import build_group

    ctx = "The game was played at Levi's Stadium in Santa Clara. It rained."
    def cand(text, score):
        s = ctx.index(text)
        return {"text": text, "start": s, "end": s + len(text), "score": score}
    row = {"id": "q", "question": "  Where was the game?", "context": ctx, "answers": ["Levi's Stadium"],
           "candidates": [cand("Levi's Stadium in Santa Clara", 5.0), cand("Levi's Stadium", 4.9),
                          cand("Santa Clara", 4.0), cand("It rained", 1.0)]}

    g = build_group(row, 4, "hard", random.Random(0), soft_power=2.0)
    assert g.question == "Where was the game?"
    w = dict(zip(g.texts, g.targets))
    assert abs(sum(g.targets) - 1) < 1e-6
    assert w["Levi's Stadium"] == max(g.targets)
    assert 0 < w["Levi's Stadium in Santa Clara"] < w["Levi's Stadium"]   # near-miss: partial credit
    assert w["Santa Clara"] == w["It rained"] == 0                        # no overlap: none
    assert build_group(row, 4, "hard", random.Random(0)).targets is None  # default: exact labels

    # soft loss on one-hot targets == cross-entropy, and padding (-inf, target 0) gives no NaN
    logits = torch.tensor([[2.0, 0.5, -1.0, float("-inf")]])
    mask = torch.tensor([[True, True, True, False]])
    onehot = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    soft = -(onehot * F.log_softmax(logits, -1).masked_fill(~mask, 0.0)).sum(-1).mean()
    assert torch.isfinite(soft) and torch.isclose(soft, F.cross_entropy(logits, torch.tensor([1])))
