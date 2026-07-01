#!/bin/bash
set -euo pipefail

cd /hkfs/work/workspace/scratch/ujzmd-lerobot/lerobot

mkdir -p logs/pi05_lemon_bowl_224_bs8_1k_single_gpu

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

source .venv/bin/activate
source env_lerobot.sh

echo "Job ${SLURM_JOB_ID:-interactive} running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    sys.exit("Expected exactly one visible CUDA GPU.")
PY

accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --main_process_port=0 \
  "$(which lerobot-train)" \
  --config_path=configs/train_pi05_lemon_bowl_224_1k_single_gpu.json \
  2>&1 | tee logs/pi05_lemon_bowl_224_bs8_1k_single_gpu/train.log
