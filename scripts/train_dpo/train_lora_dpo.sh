#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/dhf/chain_critic}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/output/v11-20260408-183210/checkpoint-70669}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/datasets/train_grpo/converted_100k_ge1/dpo_split_95_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/dpo-chaincritic-lora-ge1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"

exec > >(tee -a "${TRAIN_LOG}") 2>&1

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] DATASET_DIR=${DATASET_DIR}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

if [[ ! -f "${DATASET_DIR}/train.jsonl" ]]; then
  echo "[ERROR] Missing ${DATASET_DIR}/train.jsonl"
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/val.jsonl" ]]; then
  echo "[ERROR] Missing ${DATASET_DIR}/val.jsonl"
  exit 1
fi

if command -v swift >/dev/null 2>&1; then
  SWIFT_CMD=(swift rlhf)
else
  SWIFT_CMD=(python3 -m swift.cli.rlhf)
fi

export HF_HOME="${HF_HOME:-/root/.cache/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/.cache/hf_transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${SWIFT_CMD[@]}" \
  --rlhf_type dpo \
  --model "${MODEL_PATH}" \
  --tuner_type lora \
  --dataset "${DATASET_DIR}/train.jsonl" \
  --val_dataset "${DATASET_DIR}/val.jsonl" \
  --model_type qwen2 \
  --template qwen2_5 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 200 \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${TENSORBOARD_DIR}" \
  --dataloader_num_workers 4 \
  --dataset_num_proc 2 \
  --gradient_checkpointing true \
  --max_length 4096 \
  --report_to tensorboard \
  --beta 0.1 \
  --add_version false
