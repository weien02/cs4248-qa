#!/bin/bash
# Submit the whole experiment as a chain of Slurm jobs (each waits for the ones it needs).
#   bash cluster/pipeline.sh                   # roberta-base everything
#   MODEL=deberta-v3-base bash cluster/pipeline.sh
# Watch with: squeue --me ; tail -f logs/*.out
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${MODEL:-roberta-base}"
RERANK_ENC="${RERANK_ENC:-roberta-base}"
K="${K:-5}"
GATE_FRAC="${GATE_FRAC:-0.2}"
PREC="${PREC:---bf16}"              # use --fp16 on V100/T4
B=out/baseline-$MODEL
C=data/candidates

sub() {  # sub <name> <dependency or ""> <cmd...>
  local name=$1 dep=$2; shift 2
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
  sbatch --parsable --job-name="$name" $depflag cluster/job.sbatch "$@"
}

# 1. main baseline on train_main (heldout articles excluded), scored on dev
J1=$(sub baseline "" python scripts/train_baseline.py --model "$MODEL" --train_split train_main \
      --epochs 2 --batch_size 32 $PREC --output_dir "$B")

# 2. cross-fitted baselines: train on one fold, predict the other -> realistic hard negatives
J2a=$(sub cf0 "" bash -c "python scripts/train_baseline.py --model $MODEL --train_split fold0 \
      --epochs 2 --batch_size 32 $PREC --eval_split none --output_dir out/cf0-$MODEL && \
      python scripts/dump_candidates.py --model_dir out/cf0-$MODEL/final --split fold1 \
      --output $C/fold1.jsonl $PREC")
J2b=$(sub cf1 "" bash -c "python scripts/train_baseline.py --model $MODEL --train_split fold1 \
      --epochs 2 --batch_size 32 $PREC --eval_split none --output_dir out/cf1-$MODEL && \
      python scripts/dump_candidates.py --model_dir out/cf1-$MODEL/final --split fold0 \
      --output $C/fold0.jsonl $PREC")

# 3. candidates for heldout + dev from the main baseline
J3=$(sub dump "$J1" bash -c "python scripts/dump_candidates.py --model_dir $B/final --split heldout \
      --output $C/heldout.jsonl $PREC && \
      python scripts/dump_candidates.py --model_dir $B/final --split dev --output $C/dev.jsonl $PREC && \
      python scripts/gating_analysis.py --candidates $C/heldout.jsonl --k $K --out_dir results/gating")

# 4. rerankers: hard negatives (proposed) vs easy / random negatives (ablations)
DEPS="$J2a:$J2b:$J3"
for NEG in hard easy random; do
  JR=$(sub "rr-$NEG" "$DEPS" bash -c "python scripts/train_reranker.py \
        --train_candidates $C/fold0.jsonl $C/fold1.jsonl --val_candidates $C/heldout.jsonl \
        --encoder $RERANK_ENC --neg_mode $NEG --k $K $PREC --output_dir out/reranker-$NEG && \
        python scripts/run_system.py --candidates $C/dev.jsonl --reranker out/reranker-$NEG/best \
        --tau_from $C/heldout.jsonl --gate_frac $GATE_FRAC --k $K --tag $NEG --out_dir results/dev")
  echo "reranker-$NEG: $JR"
done

echo "baseline=$J1 crossfit=$J2a,$J2b dump=$J3"
echo "Results land in results/ablations.csv. Robustness: see README (distractors)."
