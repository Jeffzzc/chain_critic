#!/bin/bash

MODEL_PATH="/data/dhf/chain_critic/model/Qwen3.5-9B"
MODEL_NAME="Qwen3.5-9B"
GPU_MEMORY_UTILIZATION=0.95

for i in {1..4}
do
    PORT=$((8000 + i))
    GPU_ID=$i

    echo "Starting ${MODEL_NAME} on GPU ${GPU_ID} at port ${PORT}..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size 1 \
        --port "$PORT" \
        --no-enable-log-requests \
        --enforce-eager \
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