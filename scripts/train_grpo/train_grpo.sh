#!/bin/bash
set -euo pipefail

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/data/dhf/chain_critic}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/output/v11-20260408-183210/checkpoint-70669}"
DATASET="${DATASET:-${PROJECT_ROOT}/datasets/train_grpo/chaincritic_grpo_final_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/grpo-chaincritic-newdata-colocate-full}"
PLUGIN_PATH="${PLUGIN_PATH:-${PROJECT_ROOT}/scripts/train_grpo/grpo_reward_plugin.py}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-${PROJECT_ROOT}/model/Octen-Embedding-8B}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-${PROJECT_ROOT}/model/Qwen3.5-27B}"

mkdir -p "${OUTPUT_DIR}"

# -----------------------------
# Compatibility notes for current environment
# Expected:
#   python        3.11
#   torch         2.6.0+cu118
#   cuda runtime  11.8
#   cxx11abi      False
#   flash-attn    flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
#   vllm          0.8.5
# -----------------------------

# vLLM 0.8.5 does not recognize this variable. Remove it if inherited.
unset VLLM_USE_FLASH_ATTN || true

check_env() {
  echo "[INFO] Checking Python / Torch / FlashAttention / vLLM environment..."
  python - <<'PY'
import sys
import torch

print("python =", sys.version.split()[0])
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("cxx11abi =", torch._C._GLIBCXX_USE_CXX11_ABI)
print("cuda_available =", torch.cuda.is_available())

if not torch.__version__.startswith("2.6.0"):
    raise RuntimeError(f"Expected torch 2.6.0, got {torch.__version__}")

if torch.version.cuda != "11.8":
    raise RuntimeError(f"Expected torch CUDA 11.8, got {torch.version.cuda}")

if torch._C._GLIBCXX_USE_CXX11_ABI is not False:
    raise RuntimeError("Expected cxx11abi=False. Please use cxx11abiFALSE flash-attn wheel.")

import flash_attn_2_cuda
print("flash_attn_2_cuda = ok")

import vllm
print("vllm =", vllm.__version__)
PY
}

check_env

# -----------------------------
# Embedding server
# -----------------------------
EMBEDDING_CUDA_VISIBLE_DEVICES="${EMBEDDING_CUDA_VISIBLE_DEVICES:-1}"
EMBEDDING_HOST="${EMBEDDING_HOST:-127.0.0.1}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8004}"
EMBEDDING_SERVER_LOG="${EMBEDDING_SERVER_LOG:-${OUTPUT_DIR}/embedding_server.log}"
EMBEDDING_EXTRA_ARGS="${EMBEDDING_EXTRA_ARGS:---trust-remote-code}"
EMBEDDING_STARTUP_TIMEOUT="${EMBEDDING_STARTUP_TIMEOUT:-300}"

export CHAINCRITIC_EMBEDDING_BASE_URL="http://${EMBEDDING_HOST}:${EMBEDDING_PORT}/v1"
export CHAINCRITIC_EMBEDDING_MODEL="Octen-Embedding-8B"
export CHAINCRITIC_RS_SIMILARITY_THRESHOLD="${CHAINCRITIC_RS_SIMILARITY_THRESHOLD:-0.8}"
export CHAINCRITIC_RS_WEIGHT="${CHAINCRITIC_RS_WEIGHT:-0.5}"
export CHAINCRITIC_VERIFIER_WEIGHT="${CHAINCRITIC_VERIFIER_WEIGHT:-0.5}"
export CHAINCRITIC_VERIFIER_WORKERS="${CHAINCRITIC_VERIFIER_WORKERS:-8}"

is_server_up() {
  local host="$1"
  local port="$2"
  curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1
}

wait_for_server() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout="$4"
  local log_path="$5"
  local waited=0
  while ! is_server_up "${host}" "${port}"; do
    sleep 2
    waited=$((waited + 2))
    if [[ "${waited}" -ge "${timeout}" ]]; then
      echo "[ERROR] ${name} server did not become ready within ${timeout}s"
      echo "[ERROR] Last 200 lines of ${name} server log:"
      tail -n 200 "${log_path}" || true
      exit 1
    fi
  done
}

STARTED_EMBEDDING_SERVER=0
STARTED_VERIFIER_SERVER=0

cleanup() {
  local exit_code=$?
  if [[ "${STARTED_EMBEDDING_SERVER}" == "1" && -n "${EMBEDDING_SERVER_PID:-}" ]]; then
    echo "[INFO] Stopping embedding server pid=${EMBEDDING_SERVER_PID}"
    kill "${EMBEDDING_SERVER_PID}" >/dev/null 2>&1 || true
    wait "${EMBEDDING_SERVER_PID}" 2>/dev/null || true
  fi
  if [[ "${STARTED_VERIFIER_SERVER}" == "1" && -n "${VERIFIER_SERVER_PID:-}" ]]; then
    echo "[INFO] Stopping verifier server pid=${VERIFIER_SERVER_PID}"
    kill "${VERIFIER_SERVER_PID}" >/dev/null 2>&1 || true
    wait "${VERIFIER_SERVER_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

# Start embedding server if not already running.
# vLLM 0.8.5 does not support: --runner pooling
# Use: --task embed
if is_server_up "${EMBEDDING_HOST}" "${EMBEDDING_PORT}"; then
  echo "[INFO] Embedding server already running at http://${EMBEDDING_HOST}:${EMBEDDING_PORT}"
else
  echo "[INFO] Starting embedding server on GPU ${EMBEDDING_CUDA_VISIBLE_DEVICES}"
  CUDA_VISIBLE_DEVICES="${EMBEDDING_CUDA_VISIBLE_DEVICES}" \
  vllm serve "${EMBEDDING_MODEL_PATH}" \
    --task embed \
    --served-model-name "Octen-Embedding-8B" \
    --host "${EMBEDDING_HOST}" \
    --port "${EMBEDDING_PORT}" \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    ${EMBEDDING_EXTRA_ARGS} \
    > "${EMBEDDING_SERVER_LOG}" 2>&1 &

  EMBEDDING_SERVER_PID=$!
  STARTED_EMBEDDING_SERVER=1
  wait_for_server "Embedding" "${EMBEDDING_HOST}" "${EMBEDDING_PORT}" "${EMBEDDING_STARTUP_TIMEOUT}" "${EMBEDDING_SERVER_LOG}"
  echo "[INFO] Embedding server ready at http://${EMBEDDING_HOST}:${EMBEDDING_PORT}/v1"
fi

# Optional embedding endpoint smoke test.
echo "[INFO] Testing embedding endpoint..."
curl -fsS "http://${EMBEDDING_HOST}:${EMBEDDING_PORT}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model":"Octen-Embedding-8B","input":["hello world"]}' \
  >/dev/null || {
    echo "[ERROR] Embedding endpoint test failed."
    tail -n 200 "${EMBEDDING_SERVER_LOG}" || true
    exit 1
  }
echo "[INFO] Embedding endpoint test passed."

# -----------------------------
# Verifier server
# -----------------------------
VERIFIER_START_SERVER="${VERIFIER_START_SERVER:-true}"
VERIFIER_CUDA_VISIBLE_DEVICES="${VERIFIER_CUDA_VISIBLE_DEVICES:-0}"
VERIFIER_HOST="${VERIFIER_HOST:-127.0.0.1}"
VERIFIER_PORT="${VERIFIER_PORT:-8005}"
VERIFIER_MODEL_NAME="${VERIFIER_MODEL_NAME:-Qwen3.5-27B}"
VERIFIER_SERVER_LOG="${VERIFIER_SERVER_LOG:-${OUTPUT_DIR}/verifier_server.log}"
VERIFIER_EXTRA_ARGS="${VERIFIER_EXTRA_ARGS:---trust-remote-code}"
VERIFIER_STARTUP_TIMEOUT="${VERIFIER_STARTUP_TIMEOUT:-600}"
VERIFIER_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION:-0.90}"
VERIFIER_MAX_MODEL_LEN="${VERIFIER_MAX_MODEL_LEN:-4096}"

export CHAINCRITIC_VERIFIER_BASE_URL="http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"
export CHAINCRITIC_VERIFIER_MODEL="${VERIFIER_MODEL_NAME}"

if is_server_up "${VERIFIER_HOST}" "${VERIFIER_PORT}"; then
  echo "[INFO] Verifier server already running at http://${VERIFIER_HOST}:${VERIFIER_PORT}"
elif [[ "${VERIFIER_START_SERVER}" == "true" ]]; then
  echo "[INFO] Starting verifier server on GPU ${VERIFIER_CUDA_VISIBLE_DEVICES}"
  CUDA_VISIBLE_DEVICES="${VERIFIER_CUDA_VISIBLE_DEVICES}" \
  vllm serve "${VERIFIER_MODEL_PATH}" \
    --served-model-name "${VERIFIER_MODEL_NAME}" \
    --host "${VERIFIER_HOST}" \
    --port "${VERIFIER_PORT}" \
    --dtype auto \
    --gpu-memory-utilization "${VERIFIER_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VERIFIER_MAX_MODEL_LEN}" \
    ${VERIFIER_EXTRA_ARGS} \
    > "${VERIFIER_SERVER_LOG}" 2>&1 &

  VERIFIER_SERVER_PID=$!
  STARTED_VERIFIER_SERVER=1
  wait_for_server "Verifier" "${VERIFIER_HOST}" "${VERIFIER_PORT}" "${VERIFIER_STARTUP_TIMEOUT}" "${VERIFIER_SERVER_LOG}"
  echo "[INFO] Verifier server ready at http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"
else
  echo "[INFO] VERIFIER_START_SERVER=false; expecting verifier at http://${VERIFIER_HOST}:${VERIFIER_PORT}"
fi

echo "[INFO] Testing verifier endpoint..."
curl -fsS "http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${VERIFIER_MODEL_NAME}\",\"messages\":[{\"role\":\"system\",\"content\":\"Output exactly SAME.\"},{\"role\":\"user\",\"content\":\"Return SAME\"}],\"temperature\":0,\"max_tokens\":4}" \
  >/dev/null || {
    echo "[ERROR] Verifier endpoint test failed."
    tail -n 200 "${VERIFIER_SERVER_LOG}" || true
    exit 1
  }
echo "[INFO] Verifier endpoint test passed."

# -----------------------------
# Multi-GPU GRPO training
# ms-swift + vLLM colocate/internal mode
# -----------------------------
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Keep tensor parallel size = 1 by default.
# Qwen/Qwen2/Qwen2.5 hidden size / attention heads are often not divisible by 4.
# TP=4 can easily fail. Use multiple infer workers instead.
NUM_INFER_WORKERS="${NUM_INFER_WORKERS:-${NPROC_PER_NODE}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
DEEPSPEED="${DEEPSPEED:-zero3}"
# SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-false}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"
GC_COLLECT_AFTER_OFFLOAD="${GC_COLLECT_AFTER_OFFLOAD:-true}"

# Current environment has a compatible flash-attn wheel installed.
# If you still hit flash-attn problems, run with: USE_FLASH_ATTN=false bash this_script.sh
USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"
if [[ "${USE_FLASH_ATTN}" == "true" ]]; then
  MODEL_KWARGS='{"attn_implementation":"flash_attention_2"}'
else
  MODEL_KWARGS='{"attn_implementation":"sdpa"}'
fi
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${OUTPUT_DIR}/checkpoint-2000}"
export PYTHONPATH="${PROJECT_ROOT}/scripts/train_grpo:${PROJECT_ROOT}/scripts/train:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[INFO] Starting ms-swift GRPO training on GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] num_infer_workers=${NUM_INFER_WORKERS}"
echo "[INFO] vllm_tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "[INFO] use_flash_attn=${USE_FLASH_ATTN}"

CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --tuner_type full \
  --dataset "${DATASET}" \
  --external_plugins "${PLUGIN_PATH}" \
  --reward_funcs chaincritic_weighted \
  --reward_weights 1.0 \
  --model_type qwen2 \
  --template qwen2_5 \
  --torch_dtype bfloat16 \
  --deepspeed "${DEEPSPEED}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --warmup_ratio 0.05 \
  --save_total_limit 2 \
  --logging_steps 20 \
  --save_steps 200 \
  --eval_steps 200 \
  --output_dir "${OUTPUT_DIR}" \
  --dataloader_num_workers 8 \
  --dataset_num_proc 2 \
  --gradient_checkpointing true \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --num_generations 2 \
  --temperature 0.9 \
  --top_p 0.9 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --offload_model "${OFFLOAD_MODEL}" \
  --offload_optimizer "${OFFLOAD_OPTIMIZER}" \
  --log_completions true \
  --overlong_filter true \
  --add_version false \
  --model_kwargs "${MODEL_KWARGS}" \
  --resume_from_checkpoint "${RESUME_CHECKPOINT}"


# #!/bin/bash
# set -euo pipefail

# # -----------------------------
# # Project paths
# # -----------------------------
# PROJECT_ROOT="${PROJECT_ROOT:-/data/dhf/chain_critic}"
# MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/output/grpo-chaincritic/checkpoint-1000}"
# DATASET="${DATASET:-${PROJECT_ROOT}/datasets/train_grpo/chaincritic_grpo_train.jsonl}"
# OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/grpo-chaincritic}"
# PLUGIN_PATH="${PLUGIN_PATH:-${PROJECT_ROOT}/scripts/train_grpo/grpo_reward_plugin.py}"
# EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-${PROJECT_ROOT}/model/Octen-Embedding-8B}"

# mkdir -p "${OUTPUT_DIR}"

# # -----------------------------
# # Embedding server (GPU 1)
# # -----------------------------
# EMBEDDING_CUDA_VISIBLE_DEVICES="${EMBEDDING_CUDA_VISIBLE_DEVICES:-1}"
# EMBEDDING_HOST="${EMBEDDING_HOST:-127.0.0.1}"
# EMBEDDING_PORT="${EMBEDDING_PORT:-8004}"
# EMBEDDING_SERVER_LOG="${EMBEDDING_SERVER_LOG:-${OUTPUT_DIR}/embedding_server.log}"
# EMBEDDING_EXTRA_ARGS="${EMBEDDING_EXTRA_ARGS:---trust-remote-code}"
# EMBEDDING_STARTUP_TIMEOUT="${EMBEDDING_STARTUP_TIMEOUT:-300}"
# export CHAINCRITIC_EMBEDDING_BASE_URL="http://${EMBEDDING_HOST}:${EMBEDDING_PORT}/v1"
# export CHAINCRITIC_EMBEDDING_MODEL="Octen-Embedding-8B"

# is_server_up() {
#   curl -fsS "http://${EMBEDDING_HOST}:${EMBEDDING_PORT}/health" >/dev/null 2>&1
# }

# wait_for_server() {
#   local waited=0
#   while ! is_server_up; do
#     sleep 2
#     waited=$((waited + 2))
#     if [[ "${waited}" -ge "${EMBEDDING_STARTUP_TIMEOUT}" ]]; then
#       echo "[ERROR] Embedding server did not become ready within ${EMBEDDING_STARTUP_TIMEOUT}s"
#       tail -n 200 "${EMBEDDING_SERVER_LOG}" || true
#       exit 1
#     fi
#   done
# }

# STARTED_EMBEDDING_SERVER=0
# cleanup() {
#   local exit_code=$?
#   if [[ "${STARTED_EMBEDDING_SERVER}" == "1" && -n "${EMBEDDING_SERVER_PID:-}" ]]; then
#     kill "${EMBEDDING_SERVER_PID}" >/dev/null 2>&1 || true
#     wait "${EMBEDDING_SERVER_PID}" 2>/dev/null || true
#   fi
#   exit "${exit_code}"
# }
# trap cleanup EXIT INT TERM

# # Start embedding server if not already running
# if is_server_up; then
#   echo "[INFO] Embedding server already running at http://${EMBEDDING_HOST}:${EMBEDDING_PORT}"
# else
#   echo "[INFO] Starting embedding server on GPU ${EMBEDDING_CUDA_VISIBLE_DEVICES}"
#   CUDA_VISIBLE_DEVICES="${EMBEDDING_CUDA_VISIBLE_DEVICES}" \
#   vllm serve "${EMBEDDING_MODEL_PATH}" \
#     --runner pooling \
#     --served-model-name "Octen-Embedding-8B" \
#     --host "${EMBEDDING_HOST}" \
#     --port "${EMBEDDING_PORT}" \
#     --dtype auto \
#     --gpu-memory-utilization 0.85 \
#     --max-model-len 2048 \
#     ${EMBEDDING_EXTRA_ARGS} \
#     > "${EMBEDDING_SERVER_LOG}" 2>&1 &

#   EMBEDDING_SERVER_PID=$!
#   STARTED_EMBEDDING_SERVER=1
#   wait_for_server
#   echo "[INFO] Embedding server ready at http://${EMBEDDING_HOST}:${EMBEDDING_PORT}/v1"
# fi

# # -----------------------------
# # Multi-GPU GRPO training (ms-swift)
# # -----------------------------
# TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,4}"
# NPROC_PER_NODE="${NPROC_PER_NODE:-3}"

# export PYTHONPATH="${PROJECT_ROOT}/scripts/train_grpo:${PROJECT_ROOT}/scripts/train:${PROJECT_ROOT}:${PYTHONPATH:-}"
# RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint-120"


# echo "[INFO] Resuming ms-swift GRPO training from checkpoint ${RESUME_CHECKPOINT} on GPUs: ${TRAIN_CUDA_VISIBLE_DEVICES}"
# CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
# NPROC_PER_NODE="${NPROC_PER_NODE}" \
# python -m swift.cli.rlhf \
#   --rlhf_type grpo \
#   --model "${MODEL_PATH}" \
#   --resume_from_checkpoint "${RESUME_CHECKPOINT}" \
#   --tuner_type full \
#   --dataset "${DATASET}" \
#   --external_plugins "${PLUGIN_PATH}" \
#   --reward_funcs chaincritic_weighted \
#   --reward_weights 1.0 \
#   --model_type qwen2 \
#   --template qwen2_5 \
#   --torch_dtype bfloat16 \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 4 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 1e-6 \
#   --warmup_ratio 0.05 \
#   --save_total_limit 2 \
#   --logging_steps 20 \
#   --save_steps 20 \
#   --eval_steps 20 \
#   --output_dir "${OUTPUT_DIR}" \
#   --dataloader_num_workers 16 \
#   --dataset_num_proc 4 \
#   --gradient_checkpointing false \
#   --max_length 1024 \
#   --max_completion_length 512 \
#   --num_generations 2 \
#   --temperature 0.9 \
#   --top_p 0.9 \
#   --use_vllm false \
#   --sleep_level 1 \
#   --offload_model false \
#   --offload_optimizer false \
#   --log_completions true \
#   --overlong_filter true \
#   --add_version false
