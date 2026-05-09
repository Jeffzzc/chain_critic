#!/bin/bash

# 设置模型路径
MODEL_PATH="/data/dhf/chain_critic/model/Qwen3-Omni-30B-A3B-Instruct"

# 设置显存利用率，尝试减少显存占用
GPU_MEMORY_UTILIZATION=0.95  # 设置每个实例占用显存的比例，可以调整为 0.80、0.85 等

# 循环启动 4 个实例 (GPU 0 - 3)
for i in {0..3}
do
    PORT=$((8000 + i))  # 设置端口 8000-8003
    GPU_ID=$i           # 对应 GPU 0-3

    echo "Starting Qwen3-Omni-30B-A3B-Instruct on GPU $GPU_ID at Port $PORT..."

    # 启动模型服务
    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --served-model-name Qwen3-Omni-30B-A3B-Instruct \
        --tensor-parallel-size 1 \
        --port $PORT \
        --trust-remote-code \
        --disable-log-requests \
        --enforce-eager \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        > vllm_gpu_${GPU_ID}.log 2>&1 &

    # 检查启动日志是否成功
    sleep 5  # 等待 5 秒，确保服务启动
    if netstat -an | grep -q ":$PORT "; then
        echo "Qwen3-Omni-30B-A3B-Instruct model started successfully on GPU $GPU_ID at Port $PORT."
    else
        echo "Failed to start Qwen3-Omni-30B-A3B-Instruct model on GPU $GPU_ID at Port $PORT."
    fi
done

# 检查所有实例是否启动成功
for i in {0..3}
do
    PORT=$((8000 + i))
    if netstat -an | grep -q ":$PORT "; then
        echo "Port $PORT is open, instance $i is running."
    else
        echo "Port $PORT is not open, instance $i has failed to start."
    fi
done

echo "Deployment script finished."
