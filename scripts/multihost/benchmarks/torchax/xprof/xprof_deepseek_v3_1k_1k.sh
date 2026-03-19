#!/bin/bash
# Copyright 2026 Google LLC
#
# Nightly benchmark wrapper for DeepSeek V3 (1k input, 8k output).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
NIGHTLY_SCRIPT="${SCRIPT_DIR}/../../../nightly_benchmarking.sh"

# Adjust model-path, max-seqs, and code-hash below when officially serving DeepSeek.
bash "${NIGHTLY_SCRIPT}" \
  --model-path "gs://tpu-commons-ci/deepseek/r1" \
  --model-name "DeepSeek-R1" \
  --tokenizer "deepseek-ai/DeepSeek-R1" \
  --input-len 1024 \
  --output-len 1024 \
  --tp-size 16 \
  --max-seqs 640 \
  --max-model-len 2048 \
  --max-batched-tokens 640 \
  --num-prompts 10240 \
  --dataset-name "random" \
  --run-type "DAILY" \
  --device "tpu7x-16" \
  --code-hash "deepseek-hash-placeholder" \
  --created-by "bm-scheduler" \
  --new-model-design "1" \
  --gpu-memory-utilization "0.92" \
  --enable-expert-parallel \
  --additional-config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
  --disable-shared-experts-stream "0" \
  --generation-config "gs://gpolovets-inference/deepseek/generation_configs/DeepSeek-R1" \
  --vllm-mla-disable "0" \
  --moe-requantize-block-size "512" \
  --moe-requantize-weight-dtype "fp4" \
  --phased-profiling-dir "gs://tpu-commons-ci/xprof/deepseek-r1/torchax/1k-1k" \
  --skip-db-upload
