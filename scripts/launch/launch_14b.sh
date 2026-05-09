#!/bin/bash

MODEL_PATH="/data/dhf/chain_critic/model/Ministral-3-14B-Instruct-2512"
MODEL_NAME="Ministral-3-14B-Instruct-2512"
GPU_MEMORY_UTILIZATION=0.90

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

for i in {0..3}
do
    PORT=$((8000 + i))
    if ss -ltn | grep -q ":$PORT "; then
        echo "Port $PORT is open, instance $i is running."
    else
        echo "Port $PORT is not open, instance $i has failed to start."
    fi
done

echo "Deployment script finished."