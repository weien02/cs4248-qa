#!/bin/bash
# Generic GPU job: runs whatever command you pass after the script name.
# Submit from the repo root on xlogin:
#   sbatch cluster/job.sbatch python scripts/train_baseline.py --output_dir out/x ...
# Override resources on the command line, e.g.
#   sbatch --gpus=h100-96 --time=01:00:00 cluster/job.sbatch python ...
#SBATCH --job-name=cs4248-qa
#SBATCH --gpus=a100-40
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/conda.sh
export HF_HUB_OFFLINE=1          # use the cache made by setup_env.sh (unset if a download is needed)
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "cmd: $*"
"$@"
echo "end=$(date)"
