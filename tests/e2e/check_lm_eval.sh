#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script runs the lm_eval model accuracy test and checks the results against a threshold.

set -ex # Exit immediately if a command exits with a non-zero status.

# Function to display usage
usage() {
    echo "Usage: $0 --model_name <model_name> --use_moe_ep_kernel <0|1> --tensor_parallel_size <size> --max_model_len <length> --max_num_batched_tokens <num> --max_gen_toks <num> --enable_expert_parallel <0|1>"
    echo ""
    echo "All parameters are required."
    echo ""
    echo "Options:"
    echo "  --model_name <name>             Model name to evaluate."
    echo "  --use_moe_ep_kernel <0|1>       Whether to use MoE EP kernel."
    echo "  --tensor_parallel_size <size>   Tensor parallel size."
    echo "  --max_model_len <length>        Maximum model length."
    echo "  --max_num_batched_tokens <num>  Maximum number of batched tokens."
    echo "  --max_gen_toks <num>            Maximum number of generated tokens."
    echo "  --enable_expert_parallel <0|1>  Whether to enable expert parallel."
    echo "  -h, --help                      Display this help message."
    exit 1
}

# Initialize variables
MODEL_NAME=""
USE_MOE_EP_KERNEL=""
TENSOR_PARALLEL_SIZE=""
MAX_MODEL_LEN=""
MAX_NUM_BATCHED_TOKENS=""
MAX_GEN_TOKS=""
ENABLE_EXPERT_PARALLEL=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --use_moe_ep_kernel) USE_MOE_EP_KERNEL="$2"; shift ;;
        --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift ;;
        --max_num_batched_tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift ;;
        --max_gen_toks) MAX_GEN_TOKS="$2"; shift ;;
        --enable_expert_parallel) ENABLE_EXPERT_PARALLEL="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if all parameters are provided
if [ -z "$MODEL_NAME" ] || [ -z "$USE_MOE_EP_KERNEL" ] || [ -z "$TENSOR_PARALLEL_SIZE" ] || [ -z "$MAX_MODEL_LEN" ] || [ -z "$MAX_NUM_BATCHED_TOKENS" ] || [ -z "$MAX_GEN_TOKS" ] || [ -z "$ENABLE_EXPERT_PARALLEL" ]; then
    echo "Error: All parameters are required."
    usage
fi

# TODO(xw32): enable async scheduling before merging.
model_args_json=$(printf '{"pretrained": "%s", "tensor_parallel_size": %d, "max_model_len": %d, "max_num_batched_tokens": %d, "max_gen_toks": %d, "enable_expert_parallel": %d, "async_scheduling": 0}' "$MODEL_NAME" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$MAX_GEN_TOKS" "$ENABLE_EXPERT_PARALLEL")
output=$(USE_MOE_EP_KERNEL=${USE_MOE_EP_KERNEL} MODEL_IMPL_TYPE=vllm lm_eval \
    --model vllm \
    --model_args "${model_args_json}" \
    --tasks gsm8k_cot \
    --batch_size auto \
    --apply_chat_template \
    --num_fewshot 8)

echo "Evaluation output:"
echo "$output"

# TODO(xw32): restore these thresholds before merging.
flex_threshold=0.99
strict_threshold=0.99

flex_score=$(echo "$output" | grep "flexible-extract" | awk -F'|' '{print $8}' | xargs)
strict_score=$(echo "$output" | grep "strict-match" | awk -F'|' '{print $8}' | xargs)
echo "Extracted flexible-extract score: $flex_score"
echo "Extracted strict-match score: $strict_score"

if ! [[ "$flex_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: flexible-extract score is not a valid number: $flex_score"
    exit 1
fi
if ! [[ "$strict_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: strict-match score is not a valid number: $strict_score"
    exit 1
fi

is_flex_ok=$(awk -v val="$flex_score" -v threshold="$flex_threshold" 'BEGIN {print (val >= threshold)}')
is_strict_ok=$(awk -v val="$strict_score" -v threshold="$strict_threshold" 'BEGIN {print (val >= threshold)}')

if [ "$is_flex_ok" -eq 1 ] && [ "$is_strict_ok" -eq 1 ]; then
  echo "Accuracy check passed!"
  exit 0
else
  echo "Accuracy check failed!"
  if [ "$is_flex_ok" -ne 1 ]; then
    echo "flexible-extract score $flex_score is below threshold $flex_threshold"
  fi
  if [ "$is_strict_ok" -ne 1 ]; then
    echo "strict-match score $strict_score is below threshold $strict_threshold"
  fi
  exit 1
fi
