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
    echo "Usage: $0 --model_name <model_name> --use_moe_ep_kernel <0|1> --tensor_parallel_size <size> --max_model_len <length> --max_num_batched_tokens <num> --max_gen_toks <num> --enable_expert_parallel <0|1|true|false> [options]"
    echo ""
    echo "Required Options:"
    echo "  --model_name <name>             Model name to evaluate."
    echo "  --use_moe_ep_kernel <0|1>       Whether to use MoE EP kernel."
    echo "  --tensor_parallel_size <size>   Tensor parallel size."
    echo "  --max_model_len <length>        Maximum model length."
    echo "  --max_num_batched_tokens <num>  Maximum number of batched tokens."
    echo "  --enable_expert_parallel <val>  Whether to enable expert parallel (1/0 or true/false)."
    echo ""
    echo "Evaluation/Task Options:"
    echo "  --tasks <string>                (Optional) Tasks to evaluate (default: gsm8k_cot)."
    echo "  --num_fewshot <int>             (Optional) Number of few-shot examples (default: 8 for gsm8k)."
    echo "  --seed <int>                    (Optional) Random seed."
    echo "  --flex_threshold <float>        (Optional) Threshold for flexible-extract score."
    echo "  --strict_threshold <float>      (Optional) Threshold for strict-match score."
    echo "  --exact_match_threshold <float> (Optional) Threshold for exact_match score (MMLU)."
    echo "  --limit <int>                   (Optional) Limit the number of examples evaluated (e.g., 10)."
    echo ""
    echo "vLLM Model Args Options:"
    echo "  --gpu_memory_utilization<float> (Optional) GPU memory utilization (e.g., 0.8)."
    echo "  --kv_cache_dtype <string>       (Optional) KV cache dtype (e.g., fp8)."
    echo "  --limit_mm_per_prompt <json>    (Optional) Limit multimodal items per prompt."
    echo "  --hf_overrides <json>           (Optional) HuggingFace overrides."
    echo "  --block_size <int>              (Optional) Block size."
    echo "  --max_num_seqs <int>            (Optional) Maximum number of sequences."
    echo "  --enable_thinking <true|false>  (Optional) Enable thinking inside model_args."
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
FLEX_THRESHOLD=""
STRICT_THRESHOLD=""
EXACT_MATCH_THRESHOLD=""
LIMIT_MM_PER_PROMPT=""
HF_OVERRIDES=""
BLOCK_SIZE=""
MAX_NUM_SEQS=""
LIMIT=""
ENABLE_THINKING=""
TASKS="gsm8k_cot"
NUM_FEWSHOT=""
SEED=""
GPU_MEMORY_UTILIZATION=""
KV_CACHE_DTYPE=""

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
        --flex_threshold) FLEX_THRESHOLD="$2"; shift ;;
        --strict_threshold) STRICT_THRESHOLD="$2"; shift ;;
        --exact_match_threshold) EXACT_MATCH_THRESHOLD="$2"; shift ;;
        --limit_mm_per_prompt) LIMIT_MM_PER_PROMPT="$2"; shift ;;
        --hf_overrides) HF_OVERRIDES="$2"; shift ;;
        --block_size) BLOCK_SIZE="$2"; shift ;;
        --max_num_seqs) MAX_NUM_SEQS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --enable_thinking) ENABLE_THINKING="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --num_fewshot) NUM_FEWSHOT="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        --kv_cache_dtype) KV_CACHE_DTYPE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if required parameters are provided
if[ -z "$MODEL_NAME" ] || [ -z "$USE_MOE_EP_KERNEL" ] ||[ -z "$TENSOR_PARALLEL_SIZE" ] || [ -z "$MAX_MODEL_LEN" ] ||[ -z "$MAX_NUM_BATCHED_TOKENS" ] ||[ -z "$MAX_GEN_TOKS" ] || [ -z "$ENABLE_EXPERT_PARALLEL" ]; then
    echo "Error: Required parameters are missing."
    usage
fi

# Default num_fewshot for backward compatibility with original gsm8k_cot runs
if [ -z "$NUM_FEWSHOT" ] &&[ "$TASKS" == "gsm8k_cot" ]; then
    NUM_FEWSHOT="8"
fi

extra_json=""
if [ -n "$LIMIT_MM_PER_PROMPT" ]; then extra_json+=$(printf ', "limit_mm_per_prompt": %s' "$LIMIT_MM_PER_PROMPT"); fi
if[ -n "$HF_OVERRIDES" ]; then extra_json+=$(printf ', "hf_overrides": %s' "$HF_OVERRIDES"); fi
if [ -n "$BLOCK_SIZE" ]; then extra_json+=$(printf ', "block_size": %s' "$BLOCK_SIZE"); fi
if[ -n "$MAX_NUM_SEQS" ]; then extra_json+=$(printf ', "max_num_seqs": %s' "$MAX_NUM_SEQS"); fi
if [ -n "$ENABLE_THINKING" ]; then extra_json+=$(printf ', "enable_thinking": %s' "$ENABLE_THINKING"); fi
if [ -n "$DISABLE_LOG_STATS" ]; then extra_json+=$(printf ', "disable_log_stats": %s' "$DISABLE_LOG_STATS"); fi
if[ -n "$GPU_MEMORY_UTILIZATION" ]; then extra_json+=$(printf ', "gpu_memory_utilization": %s' "$GPU_MEMORY_UTILIZATION"); fi
if [ -n "$ATTENTION_BACKEND" ]; then extra_json+=$(printf ', "attention_backend": "%s"' "$ATTENTION_BACKEND"); fi
if[ -n "$GDN_PREFILL_BACKEND" ]; then extra_json+=$(printf ', "gdn_prefill_backend": "%s"' "$GDN_PREFILL_BACKEND"); fi
if[ -n "$KV_CACHE_DTYPE" ]; then extra_json+=$(printf ', "kv_cache_dtype": "%s"' "$KV_CACHE_DTYPE"); fi

# NOTE: enable_expert_parallel uses %s so it can accept integers (1) or booleans (true) safely
model_args_json=$(printf '{"pretrained": "%s", "tensor_parallel_size": %d, "max_model_len": %d, "max_num_batched_tokens": %d, "max_gen_toks": %d, "enable_expert_parallel": %s%s}' "$MODEL_NAME" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$MAX_GEN_TOKS" "$ENABLE_EXPERT_PARALLEL" "$extra_json")

# Build lm_eval arguments
lm_eval_args=(
    --model vllm
    --model_args "${model_args_json}"
    --tasks "${TASKS}"
    --batch_size auto
    --apply_chat_template
)

# Conditionally append arguments
if [ -n "$NUM_FEWSHOT" ]; then lm_eval_args+=(--num_fewshot "$NUM_FEWSHOT"); fi
if[ -n "$LIMIT" ]; then lm_eval_args+=(--limit "$LIMIT"); fi
if[ "$LOG_SAMPLES" == "true" ]; then lm_eval_args+=(--log_samples); fi
if [ -n "$OUTPUT_PATH" ]; then lm_eval_args+=(--output_path "$OUTPUT_PATH"); fi
if [ -n "$SEED" ]; then lm_eval_args+=(--seed "$SEED"); fi

output=$(VLLM_XLA_CHECK_RECOMPILATION=0 USE_MOE_EP_KERNEL=${USE_MOE_EP_KERNEL} MODEL_IMPL_TYPE=vllm lm_eval "${lm_eval_args[@]}")

echo "Evaluation output:"
echo "$output"

HAS_THRESHOLDS=false
ALL_PASSED=true

# --- Check GSM8K Thresholds ---
if [ -n "$FLEX_THRESHOLD" ] || [ -n "$STRICT_THRESHOLD" ]; then
    HAS_THRESHOLDS=true
    flex_score=$(echo "$output" | grep "flexible-extract" | head -n 1 | awk -F'|' '{print $8}' | xargs)
    strict_score=$(echo "$output" | grep "strict-match" | head -n 1 | awk -F'|' '{print $8}' | xargs)

    echo "Extracted flexible-extract score: $flex_score"
    echo "Extracted strict-match score: $strict_score"

    if ! [[ "$flex_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then echo "Error: flexible-extract score is not a valid number: $flex_score"; exit 1; fi
    if ! [[ "$strict_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then echo "Error: strict-match score is not a valid number: $strict_score"; exit 1; fi

    is_flex_ok=$(awk -v val="$flex_score" -v threshold="${FLEX_THRESHOLD:-0}" 'BEGIN {print (val >= threshold)}')
    is_strict_ok=$(awk -v val="$strict_score" -v threshold="${STRICT_THRESHOLD:-0}" 'BEGIN {print (val >= threshold)}')

    if [ "$is_flex_ok" -ne 1 ]; then echo "flexible-extract score $flex_score is below threshold $FLEX_THRESHOLD"; ALL_PASSED=false; fi
    if [ "$is_strict_ok" -ne 1 ]; then echo "strict-match score $strict_score is below threshold $STRICT_THRESHOLD"; ALL_PASSED=false; fi
fi

# --- Check MMLU Thresholds ---
if[ -n "$EXACT_MATCH_THRESHOLD" ]; then
    HAS_THRESHOLDS=true
    exact_score=$(echo "$output" | grep "exact_match" | head -n 1 | awk -F'|' '{print $8}' | xargs)

    echo "Extracted exact_match score: $exact_score"

    if ! [[ "$exact_score" =~ ^[0-9]+([.][0-9]+)?$ ]]; then echo "Error: exact_match score is not a valid number: $exact_score"; exit 1; fi

    is_exact_ok=$(awk -v val="$exact_score" -v threshold="$EXACT_MATCH_THRESHOLD" 'BEGIN {print (val >= threshold)}')

    if[ "$is_exact_ok" -ne 1 ]; then echo "exact_match score $exact_score is below threshold $EXACT_MATCH_THRESHOLD"; ALL_PASSED=false; fi
fi

# --- Final Evaluation ---
if[ "$HAS_THRESHOLDS" = true ]; then
    if [ "$ALL_PASSED" = true ]; then
        echo "Accuracy check passed!"
        exit 0
    else
        echo "Accuracy check failed!"
        exit 1
    fi
else
    echo "No thresholds provided. Evaluation ran successfully, skipping accuracy threshold check."
    exit 0
fi