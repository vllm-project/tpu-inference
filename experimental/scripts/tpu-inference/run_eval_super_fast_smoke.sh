#!/bin/bash
# run_eval_smoke_test.sh
# Extreme fast smoke test to verify quantization and TPU execution parity

set -eo pipefail

ENV_DIR="/mnt/pd/gayar/src/vllm_env"
OUTPUT_DIR="/mnt/pd/gayar/src/eval_logs"

mkdir -p "${OUTPUT_DIR}"

export VLLM_TARGET_DEVICE=tpu
export VLLM_PLATFORM=tpu
export USE_BATCHED_RPA_KERNEL=1
export MODEL_IMPL_TYPE=vllm

MODEL_ARGS_COMPACT='{"pretrained":"Qwen/Qwen3-32B","tensor_parallel_size":4,"kv_cache_dtype":"fp8","max_model_len":8192,"max_num_batched_tokens":4096,"max_num_seqs":256,"gpu_memory_utilization":0.98,"enable_prefix_caching":false,"compilation_config":{"cudagraph_capture_sizes":[]},"additional_config":{"quantization":{"qwix":{"rules":[{"module_path":".*","weight_qtype":"float8_e4m3fn","act_qtype":"float8_e4m3fn"}]}}},"enable_thinking":false}'

echo "Running 5-sample smoke-test on mmlu_pro_philosophy..."
"${ENV_DIR}/bin/python3" -m lm_eval \
  --model vllm \
  --model_args "${MODEL_ARGS_COMPACT}" \
  --tasks mmlu_pro_philosophy \
  --apply_chat_template \
  --verbosity DEBUG \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  --limit 5 \
  --seed 42

