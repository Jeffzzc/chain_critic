#!/bin/bash

MODEL_PATH="/data/dhf/chain_critic/model/Octen-Embedding-8B"
MODEL_NAME="Octen-Embedding-8B"
GPU_MEMORY_UTILIZATION=0.95

for i in {1..4}
do
    PORT=$((8000 + i))
    GPU_ID=$i
    LOG_FILE="vllm_gpu_${GPU_ID}.log"

    echo "Starting ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --runner pooling \
        --tensor-parallel-size 1 \
        --port "$PORT" \
        --disable-log-requests \
        --enforce-eager \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        > "$LOG_FILE" 2>&1 &

    sleep 8

    if ss -ltn | grep -q ":$PORT "; then
        echo "${MODEL_NAME} started successfully on GPU ${GPU_ID} at port ${PORT}."
    else
        echo "Failed to start ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}."
        echo "Check log: ${LOG_FILE}"
    fi
done

for i in {1..4}
do
    PORT=$((8000 + i))
    if ss -ltn | grep -q ":$PORT "; then
        echo "Port $PORT is open, instance $i is running."
    else
        echo "Port $PORT is not open, instance $i has failed to start."
    fi
done

echo "Deployment script finished."