#!/bin/bash

export PROJECT_WS=/hkfs/work/workspace/scratch/ujzmd-lerobot

# HF
export HF_HOME=$PROJECT_WS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# torch / cache
export TORCH_HOME=$PROJECT_WS/torch_cache
export XDG_CACHE_HOME=$PROJECT_WS/xdg_cache

# tmp
export TMPDIR=$PROJECT_WS/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR

# wandb
export WANDB_MODE=online
export WANDB_DIR=$PROJECT_WS/wandb
export WANDB_CACHE_DIR=$PROJECT_WS/wandb_cache
export WANDB_DATA_DIR=$PROJECT_WS/wandb_data

# stability
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

# avoid CPU thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# mkdir
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"