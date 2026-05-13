#!/bin/bash

MODEL_PATH="${MODEL_PATH:-/data/dhf/chain_critic/model/Qwen3.6-27B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
START_PORT="${START_PORT:-8001}"
GPUS=(${GPUS:-0 1 2 3 4 5 6})

for idx in "${!GPUS[@]}"
do
    PORT=$((START_PORT + idx))
    GPU_ID="${GPUS[$idx]}"

    echo "Starting ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --port "$PORT" \
        --disable-log-requests \
        --enable-prefix-caching \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > "vllm_gpu_${GPU_ID}.log" 2>&1 &

    sleep 5

    if ss -ltn | grep -q ":$PORT "; then
        echo "${MODEL_NAME} started successfully on GPU ${GPU_ID} at port ${PORT}."
    else
        echo "Failed to start ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}."
    fi
done

for idx in "${!GPUS[@]}"
do
    PORT=$((START_PORT + idx))
    if ss -ltn | grep -q ":$PORT "; then
        echo "Port $PORT is open, GPU ${GPUS[$idx]} instance is running."
    else
        echo "Port $PORT is not open, GPU ${GPUS[$idx]} instance has failed to start."
    fi
done

echo "Deployment script finished."
