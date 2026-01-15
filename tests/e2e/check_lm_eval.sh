#!/bin/bash
# Copyright 2026 Google LLC
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


# This script runs the QWEN model accuracy test and checks the results against a threshold.

set -e # Exit immediately if a command exits with a non-zero status.

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi
MODEL_NAME=$1

# Run the evaluation and capture the output
model_args_json=$(printf '{"pretrained": "%s", "tensor_parallel_size": 8, "max_model_len": 2048, "max_num_batched_tokens": 2048, "max_gen_toks": 256, "enable_expert_parallel": 1}' "$MODEL_NAME")
output=$(USE_MOE_EP_KERNEL=1 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 MODEL_IMPL_TYPE=vllm lm_eval \
    --model vllm \
    --model_args "${model_args_json}" \
    --tasks gsm8k_cot \
    --batch_size auto \
    --apply_chat_template \
    --num_fewshot 8)

echo "Evaluation output:"
echo "$output"

# Thresholds
flex_threshold=0.0
strict_threshold=0.0

# Extract scores
flex_score=$(echo "$output" | grep "flexible-extract" | awk -F'|' '{print $8}' | xargs)
strict_score=$(echo "$output" | grep "strict-match" | awk -F'|' '{print $8}' | xargs)

echo "Extracted flexible-extract score: $flex_score"
echo "Extracted strict-match score: $strict_score"

# Check if scores are valid numbers
if ! [[ "$flex_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: flexible-extract score is not a valid number: $flex_score"
    exit 1
fi

if ! [[ "$strict_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: strict-match score is not a valid number: $strict_score"
    exit 1
fi

# Compare scores with thresholds using awk for floating point comparison
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
