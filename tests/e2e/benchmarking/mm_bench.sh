#!/bin/bash

vllm bench throughput \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 10 \
  --hf-split train \
  --max_num_batched_tokens 98304
