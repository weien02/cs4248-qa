#!/bin/bash
# Robustness experiments: distractor-augmented dev sets + Adversarial SQuAD, scored with the
# baseline and every trained reranker, then summarised by scripts/robustness_report.py.
#
# Needs the outputs of cluster/pipeline.sh (baseline, heldout/dev candidates, rerankers) and
# data/adversarial/*.jsonl from scripts/get_adversarial.py (run that on xlogin first: needs internet).
#
#   sbatch cluster/job.sbatch bash cluster/robustness.sh      # on the cluster, from xlogin
#   MODEL=deberta-v3-base sbatch cluster/job.sbatch bash cluster/robustness.sh
#
# Local smoke test against the README debug models (numbers are meaningless):
#   BASELINE=out/debug/final RERANKERS="hard=out/debug-rr/best" PREC= MAX_SAMPLES=200 \
#     HELDOUT=data/candidates/debug_heldout.jsonl bash cluster/robustness.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-roberta-base}"
BASELINE="${BASELINE:-out/baseline-$MODEL/final}"
RERANKERS="${RERANKERS:-hard=out/reranker-hard/best easy=out/reranker-easy/best random=out/reranker-random/best}"
HELDOUT="${HELDOUT:-data/candidates/heldout.jsonl}"   # clean heldout: tau is always set here
K="${K:-5}"
GATE_FRAC="${GATE_FRAC:-0.2}"
PREC="${PREC---bf16}"            # --fp16 on V100/T4, empty on Mac
MAX_SAMPLES="${MAX_SAMPLES:-}"  # e.g. 200 for a quick local run
PY="${PY:-python}"

D=data/distractors
C=data/candidates/robustness
R=results/robustness
mkdir -p "$D" "$C" "$R"
LIMIT=${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}
RS_PREC=""                      # run_system.py only has --bf16
[ "$PREC" = "--bf16" ] && RS_PREC=--bf16

# 1. test sets: clean dev, 2 distractor kinds x 3 insert positions, Adversarial SQuAD
SETS=(clean)
for mode in random overlap; do
  for pos in start random end; do
    $PY scripts/make_distractors.py --split dev --mode "$mode" --position "$pos" $LIMIT \
      --output "$D/dev_$mode-$pos.jsonl"
    SETS+=("$mode-$pos")
  done
done
for adv in addsent addonesent; do
  if [ -f "data/adversarial/$adv.jsonl" ]; then
    SETS+=("$adv")
  else
    echo "skipping $adv: run scripts/get_adversarial.py first" >&2
  fi
done

split_for() {
  case "$1" in
    clean) echo dev ;;
    addsent|addonesent) echo "data/adversarial/$1.jsonl" ;;
    *) echo "$D/dev_$1.jsonl" ;;
  esac
}

# 2. baseline candidates for each set, then every reranker (tau from the CLEAN heldout set)
for s in "${SETS[@]}"; do
  $PY scripts/dump_candidates.py --model_dir "$BASELINE" --split "$(split_for "$s")" $LIMIT \
    --output "$C/$s.jsonl" $PREC
  for pair in $RERANKERS; do
    neg=${pair%%=*}
    $PY scripts/run_system.py --candidates "$C/$s.jsonl" --reranker "${pair#*=}" \
      --tau_from "$HELDOUT" --gate_frac "$GATE_FRAC" --k "$K" $RS_PREC \
      --tag "${s}_$neg" --out_dir "$R" --ablations_csv "$R/runs.csv"
  done
done

# 3. summary table + plot
$PY scripts/robustness_report.py --candidates_dir "$C" --results_dir "$R"
