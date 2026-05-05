#!/bin/bash
# Copyright 2026 Google LLC
#
# Nightly benchmark wrapper for DeepSeek V3 (1k input, 8k output).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
NIGHTLY_SCRIPT="${SCRIPT_DIR}/../../nightly_benchmarking.sh"

# Adjust model-path, max-seqs, and code-hash below when officially serving DeepSeek.
bash "${NIGHTLY_SCRIPT}" \
  --model-path "/mnt/disks/checkpoint/hub/models--deepseek-ai--DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/" \
  --model-name "DeepSeek-R1" \
  --tokenizer "deepseek-ai/DeepSeek-R1" \
  --input-len 1024 \
  --output-len 8192 \
  --tp-size 16 \
  --max-seqs 160 \
  --max-model-len 9216 \
  --max-batched-tokens 512 \
  --num-prompts 2560 \
  --dataset-name "random" \
  --run-type "DAILY" \
  --device "tpu7x-16" \
  --created-by "bm-scheduler" \
  --new-model-design "1" \
  --gpu-memory-utilization "0.95" \
  --enable-expert-parallel \
  --additional-config '{"compilation_sizes": [2560], "sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
  --disable-shared-experts-stream "0" \
  --generation-config "gs://gpolovets-inference/deepseek/generation_configs/DeepSeek-R1" \
  --vllm-mla-disable "0" \
  --moe-requantize-block-size "512" \
  --moe-requantize-weight-dtype "fp4" \
  --moe-all-gather-activation-dtype "fp8" \
  --api-server-count 3 \
  --force-moe-random-routing "1" \
  --run-accuracy "mmlu" \
  --mmlu-output-len "4"
