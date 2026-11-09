#!/bin/bash
# Train an alternative baseline plus a hard-negative reranker without
# overwriting the canonical RoBERTa experiment.
#
# Example:
#   MODEL=microsoft/deberta-v3-base EXPERIMENT=deberta-v3-base \
#     BASE_BATCH=16 BASE_GRAD_ACCUM=2 bash cluster/model_comparison.sh
#
# Outputs are namespaced under model_comparison/<experiment>/ in out, data, and
# results. The reranker encoder remains RoBERTa-base so only the span baseline
# and its candidate distribution change.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/slurm

: "${MODEL:?Set MODEL to a Hugging Face model id}"
: "${EXPERIMENT:?Set EXPERIMENT to a short filesystem-safe name}"

if [[ ! "$EXPERIMENT" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EXPERIMENT must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
fi

RERANK_ENC="${RERANK_ENC:-roberta-base}"
K="${K:-5}"
GATE_FRAC="${GATE_FRAC:-0.2}"
PREC="${PREC:---bf16}"
BASE_EPOCHS="${BASE_EPOCHS:-2}"
BASE_BATCH="${BASE_BATCH:-32}"
BASE_GRAD_ACCUM="${BASE_GRAD_ACCUM:-1}"
SEED="${SEED:-42}"

O="out/model_comparison/$EXPERIMENT"
C="data/candidates/model_comparison/$EXPERIMENT"
R="results/model_comparison/$EXPERIMENT"
mkdir -p "$O" "$C" "$R"

sub() {  # sub <name> <dependency or ""> <cmd...>
  local name=$1 dep=$2
  shift 2
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
  sbatch --parsable --job-name="${EXPERIMENT}-${name}" $depflag cluster/job.sbatch "$@"
}

# Main baseline used at inference.
J1=$(sub baseline "" python scripts/train_baseline.py --model "$MODEL" \
      --train_split train_main --epochs "$BASE_EPOCHS" --batch_size "$BASE_BATCH" \
      --grad_accum "$BASE_GRAD_ACCUM" --seed "$SEED" $PREC --output_dir "$O/baseline")

# Cross-fitted baselines generate out-of-fold hard negatives.
J2a=$(sub cf0 "" bash -c "python scripts/train_baseline.py --model '$MODEL' --train_split fold0 \
      --epochs '$BASE_EPOCHS' --batch_size '$BASE_BATCH' --grad_accum '$BASE_GRAD_ACCUM' \
      --seed '$SEED' $PREC --eval_split none --output_dir '$O/cf0' && \
      python scripts/dump_candidates.py --model_dir '$O/cf0/final' --split fold1 \
      --output '$C/fold1.jsonl' $PREC")
J2b=$(sub cf1 "" bash -c "python scripts/train_baseline.py --model '$MODEL' --train_split fold1 \
      --epochs '$BASE_EPOCHS' --batch_size '$BASE_BATCH' --grad_accum '$BASE_GRAD_ACCUM' \
      --seed '$SEED' $PREC --eval_split none --output_dir '$O/cf1' && \
      python scripts/dump_candidates.py --model_dir '$O/cf1/final' --split fold0 \
      --output '$C/fold0.jsonl' $PREC")

# Heldout selects the confidence threshold; dev remains final evaluation only.
J3=$(sub dump "$J1" bash -c "python scripts/dump_candidates.py \
      --model_dir '$O/baseline/final' --split heldout --output '$C/heldout.jsonl' $PREC && \
      python scripts/dump_candidates.py --model_dir '$O/baseline/final' \
      --split dev --output '$C/dev.jsonl' $PREC && \
      python scripts/gating_analysis.py --candidates '$C/heldout.jsonl' --k '$K' \
      --out_dir '$R/gating'")

# Train only the proposed hard-negative reranker for the architecture comparison.
DEPS="$J2a:$J2b:$J3"
J4=$(sub rr-hard "$DEPS" bash -c "python scripts/train_reranker.py \
      --train_candidates '$C/fold0.jsonl' '$C/fold1.jsonl' \
      --val_candidates '$C/heldout.jsonl' --encoder '$RERANK_ENC' \
      --neg_mode hard --k '$K' --seed '$SEED' $PREC --output_dir '$O/reranker-hard' && \
      python scripts/run_system.py --candidates '$C/dev.jsonl' \
      --reranker '$O/reranker-hard/best' --tau_from '$C/heldout.jsonl' \
      --gate_frac '$GATE_FRAC' --k '$K' --tag hard --out_dir '$R' \
      --ablations_csv '$R/ablations.csv' && \
      python scripts/reranker_gating_sweep.py --candidates '$C/dev.jsonl' \
      --tau_from '$C/heldout.jsonl' --reranker_choices '$R/reranker_choice_hard.json' \
      --k '$K' --tag hard --out_dir '$R' && \
      python scripts/paired_bootstrap.py --candidates '$C/dev.jsonl' \
      --predictions_a '$R/predictions_hard_baseline.json' --name_a baseline \
      --predictions_b '$R/predictions_hard_gated.json' --name_b hard_gated \
      --tag hard --out_dir '$R'")

echo "experiment=$EXPERIMENT model=$MODEL seed=$SEED"
echo "baseline=$J1 crossfit=$J2a,$J2b dump=$J3 reranker=$J4"
echo "results=$R"
