#!/bin/bash
# run_eval_robust.sh
# Robust wrapper to verify setup and execute lm_eval on TPU vLLM

set -eo pipefail

# 1. Environment Configurations
ENV_DIR="/mnt/pd/gayar/src/vllm_env"
OUTPUT_DIR="/mnt/pd/gayar/src/eval_logs"
TPU_INF_DIR="/mnt/pd/gayar/src/tpu-inference"

echo "Checking virtual environment directory at: ${ENV_DIR}"
if [ ! -d "${ENV_DIR}" ]; then
  echo "Error: Virtual environment directory not found at ${ENV_DIR}" >&2
  exit 1
fi

echo "Ensuring output directory exists: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 2. Package Verification and Dependency Setup
if ! "${ENV_DIR}/bin/python3" -c "import lm_eval" &>/dev/null; then
  echo "WARNING: lm_eval is not installed inside ${ENV_DIR}."
  echo "Attempting to fix permissions and install lm_eval..."
  # Repair permissions if venv was created as root
  sudo chown -R "${USER}:${USER}" "${ENV_DIR}" || true
  "${ENV_DIR}/bin/python3" -m ensurepip --upgrade
  "${ENV_DIR}/bin/python3" -m pip install "lm_eval[vllm]"
fi

# Add local tpu-inference repository to PYTHONPATH if available
if [ -d "${TPU_INF_DIR}" ]; then
  echo "tpu-inference source directory found. Appending to PYTHONPATH."
  export PYTHONPATH="${TPU_INF_DIR}:${PYTHONPATH:-}"
fi

# 3. Environment Variables
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLATFORM=tpu
export USE_BATCHED_RPA_KERNEL=1
export MODEL_IMPL_TYPE=vllm

# 4. Compact JSON String Parsing
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

# Remove newlines and spacing for a safe model args payload execution
MODEL_ARGS_COMPACT=$(echo "${MODEL_ARGS_JSON}" | tr -d '\n ' | sed 's/:\s*/:/g')

# 5. Run Evaluation Command
echo "Launching lm_eval payload on vLLM TPU..."
"${ENV_DIR}/bin/python3" -m lm_eval \
  --model vllm \
  --model_args "${MODEL_ARGS_COMPACT}" \
  --tasks mmlu_pro \
  --apply_chat_template \
  --verbosity DEBUG \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  --limit 50 \
  --seed 42

echo "Evaluation completed successfully."

