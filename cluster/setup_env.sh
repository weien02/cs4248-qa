#!/bin/bash
# One-time setup on the SoC cluster login node (xlogin.comp.nus.edu.sg).
#   bash cluster/setup_env.sh
set -euo pipefail

ENV_NAME=qa
MINIFORGE="$HOME/miniforge3"

if [ ! -d "$MINIFORGE" ]; then
  echo ">> Installing Miniforge to $MINIFORGE"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$MINIFORGE"
fi
source "$MINIFORGE/bin/activate"

if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

cd "$(dirname "$0")/.."
pip install -r requirements.txt   # default torch wheel on Linux includes CUDA

# cache models + data on the login node so GPU jobs don't need the network
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

mkdir -p logs
echo ">> Done. Sanity-check a GPU with:"
echo "   srun --gpus=a100-40 --time=00:10:00 bash cluster/check_gpu.sh"
