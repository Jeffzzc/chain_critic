#!/bin/bash
# 设置环境变量并执行swift sft训练命令

CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false swift sft \
  --model /data/dhf/model/Qwen2.5-7B-Instruct \
  --dataset /data/dhf/chain_critic/datasets/train/final_train_split.jsonl \
  --model_type qwen2 \
  --tuner_type full \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --logging_steps 50 \
  --max_length 2048 \
  --output_dir output \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 8 \
  --model_author swift \
  --model_name swift-robot \
  --gradient_checkpointing false
  # --resume_from_checkpoint /data/dhf/chain_critic/output/v10-20260407-105024/checkpoint-23000