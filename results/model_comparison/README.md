# Baseline Capacity Comparison

This directory compares the same confidence-gated hard-negative reranking
method across three first-stage span predictors. Every experiment uses SQuAD
v1.1, the same article-level splits, top-5 candidates, a target gate rate of
20%, and a RoBERTa-base reranker trained from scratch on hard negatives produced
by that experiment's cross-fitted baseline models.

## Results

| Baseline | Parameters | Baseline EM / F1 | Gated EM / F1 | Gain EM / F1 | Rerank-all EM / F1 |
|---|---:|---:|---:|---:|---:|
| RoBERTa-base | 125M | 85.79 / 92.08 | 86.75 / 92.29 | +0.96 / +0.21 | 85.44 / 91.16 |
| DistilBERT-base | 66M | 75.90 / 84.43 | 79.28 / 86.30 | +3.38 / +1.87 | 82.44 / 88.11 |
| TinyBERT 4-layer | 14.5M | 52.06 / 63.93 | 57.02 / 67.44 | +4.96 / +3.51 | 67.22 / 74.58 |

The paired bootstrap intervals exclude zero for both EM and F1 with DistilBERT
and TinyBERT. For RoBERTa-base, the EM improvement is significant, while the F1
95% interval narrowly includes zero.

The results show an inverse relationship between baseline capacity and
reranker gain. Smaller baselines often retain a correct answer in their top-5
candidate set but rank it poorly, leaving more opportunity for the second stage.
Reranking every candidate set strongly helps TinyBERT and DistilBERT, whereas it
hurts RoBERTa-base; confidence gating is therefore most important for the
stronger baseline.

These are single-seed runs with different baseline architectures and
pretraining procedures, so parameter count alone should not be presented as a
causal explanation. RoBERTa-base remains the strongest system in absolute EM
and F1. The comparison supports the narrower conclusion that the proposed
reranking procedure generalizes across baseline capacities and produces larger
relative gains when first-stage candidate ordering is weaker.

## Files

- `comparison.csv` contains the normalized cross-model table and 95% paired
  bootstrap intervals.
- Each model folder contains `ablations.csv`, the actual-reranker threshold
  sweep, paired bootstrap results, per-question predictions, and reranker
  choices. Available gating-analysis plots and CSVs are included as well.
- The RoBERTa files are copies of the canonical artifacts in `results/dev/`.
- DistilBERT also includes the standalone baseline metrics, predictions, and
  run configuration exported from its training directory.
