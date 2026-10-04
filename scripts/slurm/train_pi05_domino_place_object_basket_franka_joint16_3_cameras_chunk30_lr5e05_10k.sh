#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10
#SBATCH -J pi05_domino_place_object_basket_franka_joint16_3cams_chunk30_lr5e05_10k
#SBATCH -o logs/pi05_domino_place_object_basket_franka_joint16_3cams_chunk30_lr5e05_10k/%x_%j.out
#SBATCH -e logs/pi05_domino_place_object_basket_franka_joint16_3cams_chunk30_lr5e05_10k/%x_%j.err

set -e

cd /hkfs/work/workspace/scratch/ujzmd-lerobot/lerobot

mkdir -p logs/pi05_domino_place_object_basket_franka_joint16_3cams_chunk30_lr5e05_10k

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

source .venv/bin/activate
source env_lerobot.sh

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT=${MASTER_PORT}"
nvidia-smi
python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch; aborting instead of falling back to CPU.")
PY

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --config_path=configs/train_pi05_domino_place_object_basket_franka_joint16_3_cameras_chunk30_lr5e05_10k.json
