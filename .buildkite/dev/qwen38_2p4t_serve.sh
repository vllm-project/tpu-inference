#!/bin/bash
# Serve Qwen3.8-2.4T-A95B-FP8 on a v7x-32 slice (4 hosts x 8 devices).
#
# Runs inside the head node's Ray container, launched by run_multihost.sh.
# Ray copies every VLLM_*-prefixed env var from this driver process to the
# worker actors on the other three hosts, so anything set here that vLLM reads
# lands on all 32 devices. Non-VLLM_ env vars must come from the container
# environment instead (MULTIHOST_ENV_VARS in the pipeline file).
#
# Sharding, in one paragraph: linear_num_key_heads=16 caps attention-head
# parallelism at 16 (gdn_attention.py shards the Gated DeltaNet state by
# ATTN_HEAD and divides n_kq by that product), so plain tp=32 divides 16 heads
# 32 ways and breaks. The leftover factor of 2 goes to attn_dp, which is a
# member of ShardingAxisName.EXPERT and therefore keeps all 512 experts sharded
# 32 ways -- vLLM-level DP is not, and would replicate 2.4 TB per replica.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-gs://tpu-commons-qwen38-2p4t-a95b-fp8/Qwen3.8-2.4T-A95B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${VLLM_PORT:-8000}"

TP="${TENSOR_PARALLEL_SIZE:-32}"
ATTN_DP="${ATTN_DP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.92}"
# 128, not 256: USE_BATCHED_RPA_SEQ_ON_LANE selects KVLayout.SEQ_ALONG_LANE,
# whose tile alignment hard-requires page_size==128
# (batched_rpa/configs.py:360-363). Confirmed the hard way on build #940.
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# ---------------------------------------------------------------------------
# Persistent JAX compilation cache, on local disk.
#
# run_in_docker.sh (the single-host path) rsyncs gs://ullm-ci-cache/jax_cache
# onto /mnt/disks/persist and mounts it; run_multihost.sh does none of that, so
# multi-host runs compile cold every build. The pipeline file bind-mounts a
# host directory here on all four hosts (MULTIHOST_DOCKER_ARGS), so each host
# keeps its own cache across builds. Within a single build this buys nothing --
# the four host processes compile the same programs in lockstep -- the payoff
# is entirely on re-runs.
#
# compilation_manager.py:67 feeds VLLM_XLA_CACHE_PATH into
# jax_compilation_cache_dir; JAX_COMPILATION_CACHE_DIR is the catch-all for
# code paths that never construct a CompilationManager. Both are set to the
# same directory from the pipeline file, matching run_in_docker.sh.
#
# JAX can also take a gs:// URI here, which would give all four hosts one
# shared cache with no host disk at all -- but that needs gcsfs, which is not a
# declared dependency of this image. Deferred until the local path is proven.
# ---------------------------------------------------------------------------
JAX_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-/root/jax_cache}"
mkdir -p "${JAX_CACHE_DIR}"
export VLLM_XLA_CACHE_PATH="${JAX_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
echo "[jax-cache] dir=${JAX_CACHE_DIR} entries_before=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l)"
df -h "${JAX_CACHE_DIR}" || true

echo "--- effective tpu-inference env ---"
env | grep -E '^(VLLM_|MODEL_IMPL_TYPE|NEW_MODEL_DESIGN|USE_|ATTN_|SKIP_JAX|MOE_|ONEHOT_|TPU_|RUNAI_)' | sort || true

# ---------------------------------------------------------------------------
# runai_streamer throughput.
#
# Build #943 streamed 63% of the 287119 tensors in 1:58:40 -- ~218 MB/s per
# host against 2469 GB, i.e. ~3h just to load. Two defaults explain it, and
# both are plain env vars that libstreamer and requests_iterator.py read
# directly (no vLLM CLI plumbing needed, so they reach the Ray workers via
# MULTIHOST_ENV_VARS like everything else):
#
#   RUNAI_STREAMER_CONCURRENCY (default 16) is the reader thread pool. 218 MB/s
#   over 16 threads is 13.6 MB/s each, well under what a single GCS connection
#   sustains in-region, so this is thread-count-bound rather than bandwidth-
#   bound. The library warns it needs ~64 fds per thread; hence the raised
#   nofile below.
#
#   RUNAI_STREAMER_MEMORY_LIMIT (default 40 GB) is the read-ahead buffer, and
#   is the bigger problem. stream_files() is called once with all 213 files and
#   FilesRequestsIterator packs *whole files* into a request until the buffer is
#   full. This checkpoint stores one layer as a 17.2 GB file (gate+up) plus an
#   8.6 GB file (down) = 25.8 GB, so 40 GB holds exactly one layer in flight:
#   nothing for layer N+1 is fetched until every tensor of layer N has been
#   consumed, and consumption blocks on the incremental fp8 path
#   (process_weights_after_loading -> device_put -> jax.effects_barrier ->
#   malloc_trim(0) over a ~26 GB arena, 92 times). That serialises download
#   against processing and matches the observed 448 -> 7.63 -> 139 it/s swings.
#   128 GB keeps ~5 layers in flight so the two overlap. The hosts have ~929 GiB
#   and run_multihost.sh sets no memory cgroup, so the buffer is affordable.
#
# Left as overridable env vars so the pipeline file can sweep them without
# touching this script.
# ---------------------------------------------------------------------------
export RUNAI_STREAMER_CONCURRENCY="${RUNAI_STREAMER_CONCURRENCY:-64}"
export RUNAI_STREAMER_MEMORY_LIMIT="${RUNAI_STREAMER_MEMORY_LIMIT:-128000000000}"
# INFO-level streamer logging is the only confirmation that the two knobs above
# actually took effect -- libstreamer prints its resolved concurrency at start.
export RUNAI_STREAMER_LOG_LEVEL="${RUNAI_STREAMER_LOG_LEVEL:-INFO}"
export RUNAI_STREAMER_LOG_TO_STDERR="${RUNAI_STREAMER_LOG_TO_STDERR:-1}"
ulimit -n 65536 2>/dev/null || true
echo "[streamer] concurrency=${RUNAI_STREAMER_CONCURRENCY} memory_limit=${RUNAI_STREAMER_MEMORY_LIMIT} nofile=$(ulimit -n)"
free -g || true

# --additional-config: attn_dp_size=2 is passed explicitly rather than left to
# the auto-heuristic in ShardingConfigManager.from_vllm_config. With
# USE_BATCHED_RPA_SEQ_ON_LANE the heuristic divides tp by num_kv_heads*2 = 8 and
# would pick attn_dp=4 / model=8, which doubles the replication of the 80 GB of
# attention+GDN projection weights for no KV-cache benefit that the extra
# batch-split does not already provide.
exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tokenizer Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --load-format runai_streamer \
  --port "${PORT}" \
  --seed 42 \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  --additional-config "{\"sharding\": {\"sharding_strategy\": {\"enable_dp_attention\": true, \"attn_dp_size\": ${ATTN_DP}}}}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --async-scheduling
