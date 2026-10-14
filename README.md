# CS4248 G11: Confidence-Gated Span Reranking for Extractive QA

SQuAD v1.1. The baseline transformer predicts answer start/end. We keep the top-k spans and, **only when the baseline is uncertain**, let a second-stage reranker choose among them. The reranker is trained on hard negatives mined from the baseline's own high-scoring errors.

```
question + passage ─► baseline (RoBERTa/DeBERTa) ─► top-k spans + scores
                                                   │
                         confidence ≥ τ ──────────►├─► return top-1
                         confidence < τ ──► reranker(question, span, sentence, score) ─► best span
```

## Repo layout

```
src/qa/
  data.py         SQuAD loading + article-level splits (train_main / heldout / fold0 / fold1 / dev)
  features.py     sliding-window tokenisation, n-best span extraction
  baseline.py     batched inference → candidates
  confidence.py   prob_margin / score_margin / top1_prob
  reranker.py     cross-encoder reranker, negative sampling (hard / easy / random)
  metrics.py      official SQuAD EM/F1 + per-question-type breakdown
  text.py         sentence splitting / candidate marking
scripts/
  train_baseline.py    fine-tune span predictor (auto-resumes from checkpoints)
  dump_candidates.py   save top-20 spans per question to JSONL
  gating_analysis.py   sweep τ on heldout → CSV + plot
  train_reranker.py    train reranker on cross-fitted candidates
  run_system.py        baseline / rerank_all / gated / oracle → results/ablations.csv
  make_distractors.py  robustness sets (insert an off-topic sentence)
cluster/
  setup_env.sh  check_gpu.sh  job.sbatch  pipeline.sh     (NUS SoC Slurm cluster)
tests/test_core.py
```

## Data splits (why they look like this)

| Split | What | Used for |
|---|---|---|
| `train_main` | 95% of train **articles** | main baseline |
| `heldout` | 5% of train articles | choosing τ, validating the reranker |
| `fold0` / `fold1` | halves of `train_main` | cross-fitting (below) |
| `dev` | official SQuAD dev | **final numbers only** |

- Splits are by article title so the same passage never appears in two splits.
- **Cross-fitting:** a model scoring its own training data is overconfident and makes few errors, so its hard negatives are weak. The model trained on fold0 predicts fold1 and vice versa, giving realistic mistakes to train the reranker on.

## Setup (Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # model-free unit tests
```

Apple GPU (MPS) is used automatically.

## Smoke test on Mac (~minutes)

```bash
python scripts/train_baseline.py --model distilbert-base-uncased --train_split train_main \
  --max_train_samples 2000 --max_eval_samples 300 --epochs 1 --output_dir out/debug
python scripts/dump_candidates.py --model_dir out/debug/final --split heldout --max_samples 300 \
  --output data/candidates/debug_heldout.jsonl
python scripts/gating_analysis.py --candidates data/candidates/debug_heldout.jsonl --out_dir results/debug
python scripts/train_reranker.py --train_candidates data/candidates/debug_heldout.jsonl \
  --val_candidates data/candidates/debug_heldout.jsonl --encoder distilbert-base-uncased \
  --epochs 1 --output_dir out/debug-rr
python scripts/run_system.py --candidates data/candidates/debug_heldout.jsonl \
  --reranker out/debug-rr/best --tau_from data/candidates/debug_heldout.jsonl --gate_frac 0.2 \
  --tag debug --out_dir results/debug
```

(The smoke test trains and validates on the same file just to check that everything runs. The numbers are meaningless.)

## Full run on the NUS SoC cluster

```bash
ssh <soc_id>@xlogin.comp.nus.edu.sg          # SoC VPN if off-campus
git clone https://github.com/weien02/cs4248-qa.git && cd cs4248-qa
bash cluster/setup_env.sh                     # once: conda env + cache model/data
srun --gpus=a100-40 --time=00:10:00 bash cluster/check_gpu.sh
bash cluster/pipeline.sh                      # submits the whole chain
squeue --me; tail -f logs/*.out
```

`pipeline.sh` submits these jobs, each waiting on the ones it needs:

1. `baseline`: train on `train_main` (2 epochs), score dev
2. `cf0`, `cf1`: cross-fitted baselines → `fold0.jsonl`, `fold1.jsonl`
3. `dump`: heldout + dev candidates, gating sweep (`results/gating/`)
4. `rr-hard`, `rr-easy`, `rr-random`: rerankers + system eval on dev

Knobs: `MODEL=deberta-v3-base`, `K=5`, `GATE_FRAC=0.2`, `PREC=--fp16` (V100/T4). Each job stays under the default 3 h limit on an A100. Training auto-resumes from the last checkpoint if resubmitted.

Run any single step yourself with `sbatch cluster/job.sbatch python scripts/<script>.py ...`

## Ablations

`results/ablations.csv` gets one row per (tag, mode):

| Proposal ablation | Where |
|---|---|
| Baseline | `mode=baseline` |
| No gating (rerank everything) | `mode=rerank_all` |
| Full system | `tag=hard, mode=gated` |
| Without hard negatives | `tag=easy` / `tag=random`, `mode=gated` |
| Upper bound | `mode=oracle_gated` |
| Score feature removed | retrain with `--no_score` |
| Gating threshold | `results/gating/gating_sweep.csv`, or rerun `run_system.py` with other `--gate_frac` |

`metrics_<tag>.json` also has per-question-type EM/F1 and wrong→right / right→wrong flip counts.

## Robustness (distractors)

```bash
python scripts/make_distractors.py --split dev --mode overlap --position random \
  --output data/distractors/dev_overlap.jsonl
python scripts/dump_candidates.py --model_dir out/baseline-roberta-base/final \
  --split data/distractors/dev_overlap.jsonl --output data/candidates/dev_overlap.jsonl --bf16
python scripts/run_system.py --candidates data/candidates/dev_overlap.jsonl \
  --reranker out/reranker-hard/best --tau_from data/candidates/heldout.jsonl \
  --tag hard_overlap --out_dir results/robustness
```

- `--mode random`: an unrelated sentence (easy).
- `--mode overlap`: the off-topic sentence with the most question-word overlap (hard).
- For comparison, also report on Adversarial SQuAD (Jia & Liang, 2017; AddSent / AddOneSent).

## Expected scale

| | Mac M-series (fp32) | A100 (bf16) |
|---|---|---|
| roberta-base baseline, 2 epochs | ~4–10 h | ~30–45 min |
| reranker, 2 epochs | ~1–3 h | ~15–30 min |

Measure `s/it` on the first ~100 steps for real numbers.

## Owners (from proposal)

| Area | Owner |
|---|---|
| Baseline, preprocessing, training pipeline | Bryan |
| Top-k, confidence, gating | Hovan |
| Reranker, hard-negative mining | Edbert |
| Distractors, robustness, error analysis | Zilin |
| Experiment management, eval tools, ablations, integration | Wei En |
