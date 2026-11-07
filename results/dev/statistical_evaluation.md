# Statistical Evaluation and Reranker Gating Sweep

These results were generated from the candidate files, predictions, and reranker
choices tracked on `main` at commit `3e9a3c6`. No result artifacts from another
branch were copied into this evaluation.

## Paired bootstrap evaluation

Each comparison uses 10,000 paired bootstrap samples over the 10,570 SQuAD dev
questions. The difference is `gated reranker - baseline`. A 95% confidence
interval that excludes zero provides evidence of a difference between the two
systems.

| Reranker | Baseline EM | Gated EM | EM difference (95% CI) | Baseline F1 | Gated F1 | F1 difference (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Hard | 85.79 | 86.75 | +0.96 [+0.59, +1.36] | 92.08 | 92.29 | +0.21 [-0.01, +0.43] |
| Easy | 85.79 | 86.61 | +0.82 [+0.44, +1.22] | 92.08 | 92.37 | +0.29 [+0.07, +0.52] |
| Random | 85.79 | 82.90 | -2.89 [-3.36, -2.40] | 92.08 | 90.52 | -1.56 [-1.84, -1.29] |
| Hard + soft labels | 85.79 | 86.31 | +0.52 [+0.13, +0.91] | 92.08 | 92.16 | +0.07 [-0.14, +0.30] |

The hard reranker has the largest EM improvement. The easy-negative reranker has
the clearest evidence of improvement on both EM and F1. Hard + soft labels
improves EM, but its F1 interval includes zero. Random-negative training is
significantly worse than the baseline on both metrics.

## Actual reranker threshold sweep

The confidence threshold for each target rate is calibrated on
`data/candidates/heldout.jsonl`, then applied to `data/candidates/dev.jsonl`.
Actual rates differ slightly from the requested rates because thresholds are
selected from finite held-out margins.

| Reranker | Target gate % | Actual gate % | EM | EM change | F1 | F1 change |
|---|---:|---:|---:|---:|---:|---:|
| Hard | 5 | 4.3 | 86.25 | +0.46 | 92.31 | +0.23 |
| Hard | 10 | 9.2 | 86.52 | +0.73 | 92.34 | +0.26 |
| Hard | 20 | 18.6 | 86.75 | +0.96 | 92.29 | +0.21 |
| Hard | 30 | 27.7 | 86.75 | +0.96 | 92.25 | +0.17 |
| Hard | 50 | 48.3 | 86.23 | +0.44 | 91.81 | -0.27 |
| Easy | 5 | 4.3 | 86.22 | +0.43 | 92.28 | +0.19 |
| Easy | 10 | 9.2 | 86.39 | +0.60 | 92.34 | +0.26 |
| Easy | 20 | 18.6 | 86.61 | +0.82 | 92.37 | +0.29 |
| Easy | 30 | 27.7 | 86.52 | +0.73 | 92.34 | +0.25 |
| Easy | 50 | 48.3 | 85.86 | +0.07 | 91.96 | -0.12 |
| Random | 5 | 4.3 | 85.48 | -0.31 | 92.00 | -0.08 |
| Random | 10 | 9.2 | 84.66 | -1.13 | 91.50 | -0.58 |
| Random | 20 | 18.6 | 82.90 | -2.89 | 90.52 | -1.56 |
| Random | 30 | 27.7 | 80.90 | -4.89 | 89.45 | -2.63 |
| Random | 50 | 48.3 | 75.22 | -10.57 | 86.68 | -5.40 |
| Hard + soft labels | 5 | 4.3 | 86.14 | +0.35 | 92.23 | +0.15 |
| Hard + soft labels | 10 | 9.2 | 86.20 | +0.41 | 92.22 | +0.14 |
| Hard + soft labels | 20 | 18.6 | 86.31 | +0.52 | 92.16 | +0.07 |
| Hard + soft labels | 30 | 27.7 | 86.12 | +0.33 | 92.09 | +0.01 |
| Hard + soft labels | 50 | 48.3 | 85.55 | -0.24 | 91.81 | -0.27 |

For the hard and easy rerankers, roughly 20% gating gives the best overall
trade-off. Hard reaches its best EM at 20--30%, while easy reaches its best EM
and F1 at 20%. At 50%, both begin to damage F1, supporting selective rather than
universal reranking.

The full-precision results and answer-flip counts are in the corresponding
`paired_bootstrap_*.csv` and `reranker_gating_sweep_*.csv` files. JSON versions
contain the same results plus run metadata.
