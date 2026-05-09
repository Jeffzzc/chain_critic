#!/bin/bash

MODEL_PATH="/data/dhf/chain_critic/output/v11-20260408-183210/checkpoint-70669"
MODEL_NAME="chaincritic-v11-70669"
GPU_MEMORY_UTILIZATION=0.95

GPU_IDS=(1 2 3 6)

for idx in "${!GPU_IDS[@]}"
do
    GPU_ID="${GPU_IDS[$idx]}"
    PORT=$((8001 + idx))

    echo "Starting ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size 1 \
        --port "$PORT" \
        --disable-log-requests \
        --enforce-eager \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        > "vllm_gpu_${GPU_ID}.log" 2>&1 &

    sleep 5

    if ss -ltn | grep -q ":$PORT "; then
        echo "${MODEL_NAME} started successfully on GPU ${GPU_ID} at port ${PORT}."
    else
        echo "Failed to start ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}."
    fi
done

for idx in "${!GPU_IDS[@]}"
do
    GPU_ID="${GPU_IDS[$idx]}"
    PORT=$((8001 + idx))

    if ss -ltn | grep -q ":$PORT "; then
        echo "Port $PORT is open, GPU ${GPU_ID} instance is running."
    else
        echo "Port $PORT is not open, GPU ${GPU_ID} instance has failed to start."
    fi
done

echo "Deployment script finished."