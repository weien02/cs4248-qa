#!/bin/bash
# One-time setup. Run this on a COMPUTE node, not on xlogin
# (xlogin has very limited resources; conda installs there can fail with CondaMemoryError):
#
#   salloc -t 60          # reserve a node (may wait in queue)
#   srun --pty bash       # shell on the compute node
#   cd ~/cs4248-qa && bash cluster/setup_env.sh
#   exit; exit            # leave the node + release the allocation
set -euo pipefail
cd "$(dirname "$0")/.."

case "$(hostname)" in
  xlogin*) echo "You're on $(hostname). Run this inside 'salloc -t 60' + 'srun --pty bash'." >&2; exit 1 ;;
esac

ENV_NAME="${QA_ENV:-qa}"
CONDA_DIR="$HOME/miniconda3"

# 1. Miniconda (skipped if you already have conda)
if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ] && [ ! -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  echo ">> Installing Miniconda to $CONDA_DIR"
  (cd "$HOME" && curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
  bash "$HOME/Miniconda3-latest-Linux-x86_64.sh" -b -p "$CONDA_DIR"
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda init bash >/dev/null
fi
for _d in "$CONDA_DIR" "$HOME/miniforge3"; do
  [ -f "$_d/etc/profile.d/conda.sh" ] && source "$_d/etc/profile.d/conda.sh" && break
done
# newer Miniconda asks you to accept the default channels' terms once
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
  --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# 2. Project env
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
pip install -r requirements.txt   # Linux torch wheel includes CUDA

# 3. Cache model + data so GPU jobs run offline
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
grep -q "HF_HOME" ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
python - <<'EOF'
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
load_dataset("rajpurkar/squad")
for m in ["roberta-base", "distilbert-base-uncased"]:
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
print("cache ready")
EOF

mkdir -p logs/slurm
echo ">> Done. Leave the node (exit; exit), then from xlogin check a GPU with:"
echo "   srun --gpus=a100-40 --time=00:10:00 bash cluster/check_gpu.sh"
