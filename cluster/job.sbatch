#!/bin/bash
# Generic GPU job: runs whatever command you pass after the script name.
#   sbatch cluster/job.sbatch python scripts/train_baseline.py --output_dir out/x ...
# Override resources on the command line, e.g.
#   sbatch --gpus=h100-96 --time=01:00:00 cluster/job.sbatch python ...
#SBATCH --job-name=cs4248-qa
#SBATCH --gpus=a100-40
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$HOME/miniforge3/bin/activate" qa
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1          # use the cache made by setup_env.sh
export TOKENIZERS_PARALLELISM=false
cd "${SLURM_SUBMIT_DIR}"

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cmd: $*"
"$@"
echo "end=$(date)"
