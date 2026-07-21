#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-4}"
THREADS="${THREADS:-8}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASET_PATH="${DATASET_PATH:-datasets/jarvis/mbj}"
RUN_NAME="${RUN_NAME:-mbj}"
RUN_DIR="${RUN_DIR:-results/mbj}"

export CUDA_VISIBLE_DEVICES

python main.py \
  --name "${RUN_NAME}" \
  --run_dir "${RUN_DIR}" \
  --model "CartNet" \
  --dataset "mbj" \
  --dataset_path "${DATASET_PATH}" \
  --wandb_project "CartNet MBJ Multimodal Fixed Split" \
  --batch "${BATCH}" \
  --batch_accumulation 1 \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --workers "${WORKERS}" \
  --threads "${THREADS}" \
  --radius 5.0 \
  --num_layers 4 \
  --dim_in 256 \
  --device "${DEVICE}" \
  --use_text True \
  --description_file description.csv \
  --text_embedding_file text_embeddings.npy \
  --text_projection_dim 128 \
  --use_late_fusion True \
  --late_fusion_type gated \
  --late_fusion_output_dim 128 \
  --fusion_dropout 0.1 \
  --text_sample_dropout 0.20 \
  --contrastive_weight 0.03 \
  --contrastive_temperature 0.10 \
  --contrastive_projection_dim 128 \
  --use_middle_fusion True \
  --middle_fusion_type residual \
  --middle_fusion_layers "2" \
  --middle_fusion_hidden_dim 256 \
  --middle_fusion_num_heads 2 \
  --middle_fusion_dropout 0.1 \
  --middle_fusion_use_gate_norm True \
  --middle_fusion_use_learnable_scale False \
  --middle_fusion_initial_scale 1.0
