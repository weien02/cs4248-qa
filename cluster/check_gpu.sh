#!/bin/bash
# srun --gpus=a100-40 --time=00:10:00 bash cluster/check_gpu.sh
cd "$(dirname "$0")/.."
source cluster/conda.sh
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
