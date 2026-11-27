#!/bin/bash
# =============================================================================
# Qwen3.5-397B-A17B Evaluation
# =============================================================================
# Runs a single (config, task) evaluation using lm_eval in offline mode.
#
# Configs:
#   fp8       - FP8 base checkpoint (Qwen/Qwen3.5-397B-A17B-FP8)
#   nvfp4     - NVIDIA pre-quantized FP4 (nvidia/Qwen3.5-397B-A17B-NVFP4)
#   tpufp4    - FP8 requantized to FP4 on TPU
#               Use --block-size and --clip-percentile to configure.
#
# Examples:
#   # FP8 baseline on AIME26
#   bash eval_qwen35_397b.sh fp8 aime26
#
#   # NVIDIA FP4 on GPQA Diamond
#   bash eval_qwen35_397b.sh nvfp4 gpqa_diamond_cot_zeroshot
#
#   # TPU FP4 BS512 with p99.9 clipping on MMLU-Pro
#   bash eval_qwen35_397b.sh tpufp4 mmlu_pro --block-size 512 --clip-percentile 99.9
#
# =============================================================================
set -euo pipefail

usage() {
    sed -n '2,/^# ====/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
}

[[ $# -lt 2 ]] && usage

CONFIG=$1
TASK=$2
shift 2

BLOCK_SIZE=512
CLIP_PERCENTILE="99.9"
SEED=1
OUTPUT_DIR="./eval_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --block-size) BLOCK_SIZE=$2; shift 2 ;;
        --clip-percentile) CLIP_PERCENTILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running."
    exit 1
fi

case $CONFIG in
    fp8)
        MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
        EXTRA_ENVS=()
        TAG="fp8_base"
        ;;
    nvfp4)
        MODEL="nvidia/Qwen3.5-397B-A17B-NVFP4"
        EXTRA_ENVS=("DISABLE_WEIGHT_REQUANTIZATION=1")
        TAG="nvfp4"
        ;;
    tpufp4)
        MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
        EXTRA_ENVS=(
            "MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn"
            "MOE_REQUANTIZE_BLOCK_SIZE=$BLOCK_SIZE"
        )
        TAG="tpufp4_bs${BLOCK_SIZE}"
        if [ -n "$CLIP_PERCENTILE" ]; then
            EXTRA_ENVS+=("MOE_REQUANTIZE_CLIP_PERCENTILE=$CLIP_PERCENTILE")
            TAG="${TAG}_clip${CLIP_PERCENTILE}"
        fi
        ;;
    *)
        echo "Unknown config: $CONFIG (expected fp8, nvfp4, or tpufp4)"
        exit 1
        ;;
esac

LOG_DIR="${OUTPUT_DIR}/${TAG}_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

MODEL_ARGS=$(python3 -c "
import json
print(json.dumps({
    'pretrained': '$MODEL',
    'attention_backend': 'FLASH_ATTN',
    'gdn_prefill_backend': 'triton',
    'kv_cache_dtype': 'fp8',
    'tensor_parallel_size': 8,
    'gpu_memory_utilization': 0.9,
    'max_model_len': 48000,
    'max_num_batched_tokens': 4096,
    'max_num_seqs': 32,
    'block_size': 256,
    'mamba_ssm_cache_dtype': 'float32',
    'language_model_only': True,
    'additional_config': {'sharding': {'sharding_strategy': {'enable_dp_attention': True}}},
    'limit_mm_per_prompt': {'image': 0, 'video': 0},
    'enable_prefix_caching': False,
    'enable_thinking': False,
    'reasoning_parser': 'qwen3',
    'enable_expert_parallel': True,
    'seed': 1,
}))
")

echo "=== Qwen3.5-397B-A17B Eval ==="
echo "Config:  $CONFIG ($TAG)"
echo "Task:    $TASK"
echo "Seed:    $SEED"
echo "Output:  $LOG_DIR"
[ ${#EXTRA_ENVS[@]} -gt 0 ] && echo "Envs:    ${EXTRA_ENVS[*]}"
echo "==============================="

USE_MOE_EP_KERNEL=0 \
    MODEL_IMPL_TYPE=vllm \
    ATTN_BUCKETIZED_NUM_REQS=true \
    ATTN_CUSTOM_NUM_REQS_BUCKETS=8 \
    ONEHOT_MOE_PERMUTE_THRESHOLD=8192 \
    RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d \
    NEW_MODEL_DESIGN=1 \
    HF_TOKEN="$HF_TOKEN" \
    HF_HOME="${HF_HOME:-}" \
    ${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"} \
    lm_eval \
    --model vllm \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --apply_chat_template \
    --batch_size auto \
    --log_samples \
    --output_path "$LOG_DIR" \
    --seed "$SEED" \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,do_sample=true" \
    2>&1 | tee "$LOG_DIR/eval.log"

# Clean up EngineCore processes
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
sleep 3
rm -f /tmp/libtpu_lockfile

echo "=== Done: $TAG | $TASK ==="
echo "Results: $LOG_DIR"
