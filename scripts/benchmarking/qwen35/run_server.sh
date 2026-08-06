#!/usr/bin/env bash
# Serve Qwen3.5-397B-A17B with this branch's optimizations at the pinned
# production settings. This is the exact server side of the recipe the
# published numbers were measured on. Wait for "Application startup complete"
# in the log before starting the client. A cold start takes about an hour
# before the server answers, a warm one about half an hour.
#
#   ./run_server.sh              eight-bit weights, the checkpoint form
#   WEIGHTS=fp4 ./run_server.sh  four-bit MoE weights
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Fail in seconds on a broken install rather than partway into a long start.
command -v vllm >/dev/null 2>&1 \
  || { echo "vllm is not on PATH, install per docs/getting_started/installation.md" >&2; exit 1; }

WEIGHTS="${WEIGHTS:-fp8}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
LOG="${LOG:-$SCRIPT_DIR/server.log}"

# Production base settings, identical across both weight forms.
export MODEL_IMPL_TYPE=vllm
export NEW_MODEL_DESIGN=1
export USE_MOE_EP_KERNEL=0
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export VLLM_MOE_CHUNK_SIZE=256

# The optimizations this branch adds. The fused expert-parallel kernel serves
# every step at or above the token threshold, which is prefill and saturation.
# Below it the grouped matmul path serves, which is decode.
export USE_MOE_FUSED_EP_KERNEL=1
export USE_MOE_FUSED_GMM_KERNEL=1
export MOE_FUSED_EP_KERNEL_MIN_TOKENS=1024

# The recurrent state cache in bfloat16 instead of float32. The kernel widens
# on load and rounds on writeback, so only the stored cache is narrower.
export GDN_BF16_RECURRENT_STATE=1

# Slice the rope table to the model length. Under text-only serving positions
# cannot exceed the model length, so the full table is never needed. Look for
# the line "Sliced rope cache" in the log to confirm it engaged.
export SLICE_ROPE_CACHE=1

# Explicit even where the default already matches, so an inherited shell
# value or a future default change cannot silently alter the recipe.
export LOGITS_ALL_GATHER_CONSERVATIVE=1
export DP_SCHED_BATCH_PREFILL=1

case "$WEIGHTS" in
  fp4)
    # MoE expert weights requantized to four-bit e2m1 at load, quantization
    # block 512, no clip. The checkpoint stays eight-bit, which is why this
    # start is longer than the eight-bit one. Look for the line
    # "re-quantizing MoE weights to float4_e2m1fn" to confirm it engaged.
    export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
    export MOE_REQUANTIZE_BLOCK_SIZE=512
    ;;
  fp8)
    unset MOE_REQUANTIZE_WEIGHT_DTYPE MOE_REQUANTIZE_BLOCK_SIZE || true
    ;;
  *)
    echo "run_server.sh: WEIGHTS must be fp4 or fp8, got '$WEIGHTS'" >&2
    exit 2
    ;;
esac

# The production LIBTPU flag list. The runtime prepends one more flag of
# its own at import.
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'

# The engine start includes the full compile, far past vllm's 600 second
# default readiness timeout.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-10800}"

# A compile inside a measured window invalidates the cell. Keep every cache
# miss loud in the log instead of silent.
export JAX_EXPLAIN_CACHE_MISSES=true

# Persistent compile cache. The first start pays the full compile, every
# later start reuses it.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRIPT_DIR/jax-cache}"
export VLLM_XLA_CACHE_PATH="$JAX_COMPILATION_CACHE_DIR"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "weights=$WEIGHTS model=$MODEL port=$PORT log=$LOG"
exec vllm serve "$MODEL" \
  --max-model-len=9236 \
  --max-num-batched-tokens=1024 \
  --max-num-seqs=64 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization=0.88 \
  --tensor-parallel-size=8 \
  --async-scheduling \
  --port="$PORT" \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  '--limit-mm-per-prompt={"image": 0, "video": 0}' \
  --kv-cache-dtype=fp8 \
  --enable-expert-parallel \
  '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' \
  --block-size=256 \
  2>&1 | tee "$LOG"
