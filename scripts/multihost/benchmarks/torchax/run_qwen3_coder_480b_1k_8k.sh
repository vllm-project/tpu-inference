#!/bin/bash
# Copyright 2026 Google LLC
#
# Nightly benchmark wrapper for Qwen3 Coder 480B (1k input, 8k output).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
NIGHTLY_SCRIPT="${SCRIPT_DIR}/../../nightly_benchmarking.sh"

bash "${NIGHTLY_SCRIPT}" \
  --model-path "gs://tpu-commons-ci/qwen/models--Qwen--Qwen3-Coder-480B-A35B-Instruct/snapshots/9d90cf8fca1bf7b7acca42d3fc9ae694a2194069" \
  --model-name "Qwen3-Coder-480B-A35B-Instruct" \
  --tokenizer "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
  --input-len 1024 \
  --output-len 8192 \
  --tp-size 16 \
  --max-seqs 128 \
  --max-model-len 10240 \
  --max-batched-tokens 1024 \
  --num-prompts 128 \
  --dataset-name "random" \
  --run-type "DAILY" \
  --device "tpu7x-16" \
  --created-by "bm-scheduler"
