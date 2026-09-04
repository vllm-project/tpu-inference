#!/bin/bash
# Serve Qwen3.5-397B-A17B-FP8 across a v7x-16 slice (2 hosts x 8 devices) at
# tensor-parallel 16 with attention DP left off, so the ATTN_HEAD mesh product
# is exactly 16.
#
# Why 16 specifically. gdn_attention.py:123-126 computes the per-device Gated
# DeltaNet head counts as a bare integer division:
#
#     tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
#     n_kq = n_kq // tp_size          # linear_num_key_heads = 16
#     n_v  = n_v  // tp_size          # linear_num_value_heads = 64
#
# ShardingAxisName.ATTN_HEAD is ('model','expert','dcp'). With linear_num_key_heads
# = 16 -- the same value in the 27B, this 397B and the 2.4T -- tp_size=16 is the
# largest product that still yields a whole key head, and it yields exactly one.
# Everything de-risked so far ran at 4 key heads per device (tp_size=4) or 2
# (tp_size=8); the v7x-32 plan runs at 1, and nothing has ever executed that.
#
# The division is exact, so this is not about truncation. It is about a
# degenerate shape: each device's recurrent state slice becomes
# (slots, 1, 128, 128), and a size-1 head axis inside a jax.shard_map with
# check_vma=False is where TPU vector layout, tiling and padding can take a
# different lowering path than a size-4 axis.
#
# attn_dp is deliberately left at 1 (no --additional-config at all, which is
# also what keeps the auto-heuristic in sharding.py:262-275 from firing -- it
# only runs under enable_dp_attention, and at tp=16 with 2 kv heads it would
# otherwise pick attn_dp=4/model=4 and miss the point of this run entirely).
# That isolates "1 GDN key head per device" from "attn_dp", which the v7x-8
# arms already cleared.
set -euo pipefail

MODEL="${MODEL_PATH:-Qwen/Qwen3.5-397B-A17B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${VLLM_PORT:-8000}"

TP="${TENSOR_PARALLEL_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.88}"
# 128, not 256: USE_BATCHED_RPA_SEQ_ON_LANE selects KVLayout.SEQ_ALONG_LANE,
# whose tile alignment hard-requires page_size==128 (batched_rpa/configs.py:360-363).
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# Same local-disk compilation cache arrangement as the v7x-32 run: run_multihost.sh
# sets up no cache of its own, so the pipeline file bind-mounts one per host.
JAX_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-/root/jax_cache}"
mkdir -p "${JAX_CACHE_DIR}"
export VLLM_XLA_CACHE_PATH="${JAX_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
echo "[jax-cache] dir=${JAX_CACHE_DIR} entries_before=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l)"

echo "=== cap edge: tp=${TP} attn_dp=1 -> ATTN_HEAD product ${TP}, 16/${TP} = $((16 / TP)) GDN key head(s)/device ==="
echo "--- effective tpu-inference env ---"
env | grep -E '^(VLLM_|MODEL_IMPL_TYPE|NEW_MODEL_DESIGN|USE_|ATTN_|SKIP_JAX|MOE_|ONEHOT_|TPU_)' | sort || true

# Tokenizer preflight. Qwen3_5MoeForConditionalGeneration builds a tokenizer in
# its own __init__ (qwen3_5.py:712), so a cache holding weights but no usable
# tokenizer.json takes down every rank ~19 s into model construction with a
# "Couldn't instantiate the backend tokenizer" ValueError whose text blames
# missing sentencepiece/tiktoken -- both of which are installed, and neither of
# which is the problem. That cost build tc#944 ten minutes to reach and a full log
# download to read. Resolve it here instead, in seconds, with the cache listing
# attached.
echo "--- tokenizer preflight (HF_HOME=${HF_HOME:-<unset>}) ---"
python3 - "${MODEL}" <<'PY' || { echo "!!! tokenizer preflight failed; not starting the server"; exit 1; }
import sys, os
model = sys.argv[1]
import transformers, tokenizers
print(f"transformers={transformers.__version__} tokenizers={tokenizers.__version__}")
from huggingface_hub import snapshot_download
# Small files only (~23 MB of index + 13 MB of tokenizer): repairs a snapshot
# that is missing them without touching the 378 GiB of weights.
path = snapshot_download(model, allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*"])
print("snapshot:", path)
for name in sorted(os.listdir(path)):
    full = os.path.join(path, name)
    print(f"   {name:40s} {os.path.getsize(full) if os.path.isfile(full) else '<dir>'}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model)
print(f"tokenizer OK: {type(tok).__name__} vocab={tok.vocab_size}")
PY

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --seed 42 \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --no-enable-prefix-caching \
  --async-scheduling
