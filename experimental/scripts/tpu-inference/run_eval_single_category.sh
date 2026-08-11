#!/bin/bash
# run_eval_fast_subtask.sh
# Runs a single fast subtask for quick validation

set -eo pipefail

ENV_DIR="/mnt/pd/gayar/vllm_env"
OUTPUT_DIR="/mnt/pd/gayar/eval_logs"

echo "Ensuring output directory exists: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Environment variables
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLATFORM=tpu
export USE_BATCHED_RPA_KERNEL=1
export MODEL_IMPL_TYPE=vllm

# Compact JSON configuration payload
MODEL_ARGS_JSON='{
  "pretrained": "Qwen/Qwen3-32B",
  "tensor_parallel_size": 4,
  "kv_cache_dtype": "fp8",
  "max_model_len": 8192,
  "max_num_batched_tokens": 4096,
  "max_num_seqs": 256,
  "gpu_memory_utilization": 0.98,
  "enable_prefix_caching": false,
  "compilation_config": {
    "cudagraph_capture_sizes": []
  },
  "additional_config": {
    "quantization": {
      "qwix": {
        "rules": [
          {
            "module_path": ".*",
            "weight_qtype": "float8_e4m3fn",
            "act_qtype": "float8_e4m3fn"
          }
        ]
      }
    }
  },
  "enable_thinking": false
}'
MODEL_ARGS_COMPACT=$(echo "${MODEL_ARGS_JSON}" | tr -d '\n ' | sed 's/:\s*/:/g')

echo "Launching quick subtask evaluation..."
"${ENV_DIR}/bin/python3" -m lm_eval \
  --model vllm \
  --model_args "${MODEL_ARGS_COMPACT}" \
  --tasks mmlu_pro_philosophy \
  --apply_chat_template \
  --verbosity DEBUG \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  --limit 10 \
  --seed 42

