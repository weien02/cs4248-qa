"""Fast tests that need no model download: pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qa.confidence import confidence  # noqa: E402
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
