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
# (batched_rpa/configs.py:360-363). Confirmed the hard way on build tc#940.
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# Attention blocks the *scheduler* is allowed to hand out.
#
# The runner sizes the attention pool against mamba's real footprint (a flat
# `max_num_reqs+1` slots per mamba layer, not one slot per attention block), so
# it physically allocates ~23x more attention blocks than vLLM's uniform sizing
# would. That calculation happens worker-side, inside `initialize_kv_cache`,
# which is *after* the engine-core process has already built the scheduler's
# block pool from its own `get_kv_cache_config()`. The worker also sets
# `cache_config.num_gpu_blocks_override`, but only on its own copy of the
# config -- `get_kv_cache_spec()` returns specs, not config mutations, so that
# value never crosses the process boundary.
#
# Net effect without this flag: the HBM is reserved but the scheduler still
# hands out blocks from the small uniform count, and the surplus is allocated
# and unaddressable. Measured on build tc#960 -- 15,150 blocks allocated,
# `kv_cache_config.num_blocks=661` in use, `GPU KV cache size: 83,822 tokens,
# Maximum concurrency for 40,960 tokens per request: 2.05x`. At conc 32 that is
# ~13x oversubscribed, and the resulting preemption storm cost 28 of 198 GPQA
# questions.
#
# Setting it here puts the value in the engine-core config before
# `get_kv_cache_config()` runs, which is the only place it is read.
#
# #3481 has since made the worker derive its own tensor sizing from
# `kv_cache_config.num_blocks`, so the two sides can no longer allocate and
# admit against different numbers. That removes the wasted-HBM half of the
# problem but not this one: the scheduler still has no way to learn the
# compact-mamba pool size unless it is told on the command line.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"

VLLM_EXTRA_ARGS=()
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  VLLM_EXTRA_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi

# ---------------------------------------------------------------------------
# Where the weights come from.
#
# MODEL_PATH is normally the object-store URI. The pipeline file overrides it
# with a locally-mounted copy of the checkpoint when its preflight has proved
# that *every* host in the slice can read one (see qwen38_2p4t_preflight.sh);
# the mount is discovered at runtime, never hardcoded, because operators attach
# the disk wherever they like.
#
# runai_streamer handles both: _prepare_weights() takes os.path.isdir() as
# "local" and hands the directory straight to list_safetensors(), so the same
# --load-format works either way, just over libstreamer's file backend instead
# of its GCS backend.
#
# --served-model-name pins the name the client asks for, so swapping the path
# does not change the client's --model.
# ---------------------------------------------------------------------------
if [[ "${MODEL_PATH}" != *://* ]]; then
  echo "[weights] source=LOCAL path=${MODEL_PATH}"
  if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
    echo "[weights] FATAL: ${MODEL_PATH}/config.json is not readable in this container." >&2
    echo "[weights] The bind mount did not reach the container. Aborting rather than" >&2
    echo "[weights] burning slice time on a load that cannot succeed." >&2
    exit 1
  fi
  # Globbed into an array rather than `ls | head`: under `set -o pipefail` the
  # head closes the pipe early, ls dies of SIGPIPE, and `set -e` takes the whole
  # script down with exit 141 before vllm is ever reached.
  _shards=("${MODEL_PATH}"/*.safetensors)
  [[ -f "${_shards[0]}" ]] || _shards=()
  echo "[weights] files=${#_shards[@]} safetensors"
  du -sh "${MODEL_PATH}" 2>/dev/null || true
  # Cold-read one file to get a directly comparable number against the ~100-220
  # MB/s per host this checkpoint streamed from GCS. Bounded so a pathological
  # mount costs 60s, not the run.
  _probe_file="${_shards[0]:-}"
  if [[ -n "${_probe_file}" ]]; then
    echo "[weights] read probe on $(basename "${_probe_file}"):"
    timeout 60 dd if="${_probe_file}" of=/dev/null bs=8M count=256 2>&1 | tail -1 || true
  fi
  # Take the tokenizer from the same disk when it is there, so the run needs no
  # network at all. Checked separately: the preflight validates the mount by
  # config.json and safetensors count, which a weights-only sync would pass.
  if [[ -r "${MODEL_PATH}/tokenizer_config.json" ]]; then
    TOKENIZER="${MODEL_PATH}"
    echo "[weights] tokenizer=LOCAL"
  fi
else
  echo "[weights] source=OBJECT-STORE path=${MODEL_PATH}"
fi
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.8-2.4T-A95B-FP8}"

# ---------------------------------------------------------------------------
# Persistent JAX compilation cache, on local disk, with a disk-space guard.
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
# The guard matters now that SKIP_JAX_PRECOMPILE is off: precompilation writes
# an entry per (num_tokens, num_reqs) bucket instead of the handful the lazy
# path produced, and the checkpoint disks are read-only, so the cache has
# nowhere to go but the host root filesystem. Three layers, cheapest first:
#
#   1. JAX_COMPILATION_CACHE_MAX_SIZE caps the directory and makes JAX evict
#      least-recently-used entries itself. JAX's default is -1, unlimited.
#   2. A preflight free-space floor. Below it, the persistent cache is
#      *disabled* rather than the run failed -- recompiling costs minutes,
#      ENOSPC three hours into a load costs the whole build.
#   3. A background watchdog, since the cache is not the only thing that can
#      fill the disk (Ray logs, container layers, core dumps). It reports and,
#      as a last resort, evicts.
#
# JAX can also take a gs:// URI here, which would give all four hosts one
# shared cache with no host disk at all -- but that needs gcsfs, which is not a
# declared dependency of this image. Deferred until the local path is proven.
# ---------------------------------------------------------------------------
JAX_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-/root/jax_cache}"
JAX_CACHE_MAX_GB="${JAX_CACHE_MAX_GB:-20}"
JAX_CACHE_MIN_FREE_GB="${JAX_CACHE_MIN_FREE_GB:-25}"

free_mb() { df -Pm "$1" 2>/dev/null | awk 'NR==2{print $4}'; }

disable_jax_cache() {
  echo "[disk-guard] persistent JAX compilation cache DISABLED: $1"
  export VLLM_DISABLE_COMPILE_CACHE=1
  export JAX_COMPILATION_CACHE_MAX_SIZE=0
  unset JAX_COMPILATION_CACHE_DIR
}

mkdir -p "${JAX_CACHE_DIR}" 2>/dev/null || true
_free=$(free_mb "${JAX_CACHE_DIR}")
_free="${_free:-0}"
echo "[jax-cache] dir=${JAX_CACHE_DIR} entries_before=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l) free=${_free}MB"
df -h "${JAX_CACHE_DIR}" || true

if [[ "${VLLM_DISABLE_COMPILE_CACHE:-0}" == "1" ]]; then
  # The pipeline's preflight already took this decision for the whole slice.
  # Do not second-guess it per host: JAX_COMPILATION_CACHE_DIR would re-enable
  # JAX's own persistent cache behind vLLM's back, leaving the slice half
  # caching and half not.
  disable_jax_cache "VLLM_DISABLE_COMPILE_CACHE=1 from the pipeline preflight"
elif ! touch "${JAX_CACHE_DIR}/.wtest" 2>/dev/null; then
  disable_jax_cache "${JAX_CACHE_DIR} is not writable (read-only mount?)"
elif [[ "${_free}" -lt $((JAX_CACHE_MIN_FREE_GB * 1024)) ]]; then
  rm -f "${JAX_CACHE_DIR}/.wtest"
  disable_jax_cache "only ${_free}MB free, floor is $((JAX_CACHE_MIN_FREE_GB * 1024))MB"
else
  rm -f "${JAX_CACHE_DIR}/.wtest"
  export VLLM_XLA_CACHE_PATH="${JAX_CACHE_DIR}"
  export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
  export JAX_COMPILATION_CACHE_MAX_SIZE=$((JAX_CACHE_MAX_GB * 1024 * 1024 * 1024))
  echo "[jax-cache] enabled, capped at ${JAX_CACHE_MAX_GB}GB (LRU eviction by JAX)"
fi

# ---------------------------------------------------------------------------
# Disk watchdog. Prints a greppable [disk-guard] line every interval and shouts
# CRITICAL when free space crosses the floor, so a failing run says why in the
# log instead of surfacing as an opaque ENOSPC. Past the floor it evicts the
# oldest cache entries, which only costs a recompile.
#
# It watches $$ and exits when that dies. The EXIT trap alone would not do it:
# the last line of this script is `exec vllm serve`, which replaces the shell
# without firing traps. `exec` keeps the PID, so $$ is the server once it takes
# over, and an orphaned watchdog holding this stdout open would stop the
# docker exec from ever returning at teardown.
# ---------------------------------------------------------------------------
DISK_WATCH_INTERVAL_S="${DISK_WATCH_INTERVAL_S:-300}"
SERVE_PID=$$
(
  # No -e/-pipefail in here: the eviction pipeline ends in a `head` that closes
  # the pipe early, and a watchdog that kills itself on SIGPIPE the first time
  # it is needed is worse than no watchdog.
  set +e +o pipefail
  crit=$((JAX_CACHE_MIN_FREE_GB * 1024))
  waited=0
  # Poll in short slices instead of sleeping the whole interval, so the exit
  # check is prompt. Sleeping the full interval would leave the watchdog
  # holding this stdout for up to DISK_WATCH_INTERVAL_S after the server ends,
  # which reads as a hang at teardown.
  while kill -0 "${SERVE_PID}" 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    [ "${waited}" -lt "${DISK_WATCH_INTERVAL_S}" ] && continue
    waited=0
    f=$(free_mb /); f="${f:-0}"
    c=$(free_mb "${JAX_CACHE_DIR}"); c="${c:-0}"
    sz=$(du -sm "${JAX_CACHE_DIR}" 2>/dev/null | awk '{print $1}')
    if [[ "${f}" -lt "${crit}" || "${c}" -lt "${crit}" ]]; then
      echo "[disk-guard] CRITICAL free root=${f}MB cache_fs=${c}MB cache_size=${sz:-0}MB floor=${crit}MB"
      # Oldest-first eviction, half the entries, one pass per tick.
      n=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l)
      if [[ "${n}" -gt 2 ]]; then
        find "${JAX_CACHE_DIR}" -type f -printf '%T@ %p\n' 2>/dev/null \
          | sort -n | head -n $((n / 2)) | cut -d' ' -f2- | xargs -r rm -f
        echo "[disk-guard] evicted $((n / 2))/${n} cache entries"
      fi
    else
      echo "[disk-guard] free root=${f}MB cache_fs=${c}MB cache_size=${sz:-0}MB"
    fi
  done
) &
DISK_WATCH_PID=$!
trap 'kill "${DISK_WATCH_PID}" 2>/dev/null || true' EXIT
echo "[disk-guard] watchdog pid=${DISK_WATCH_PID} interval=${DISK_WATCH_INTERVAL_S}s floor=${JAX_CACHE_MIN_FREE_GB}GB"

echo "--- effective tpu-inference env ---"
env | grep -E '^(VLLM_|MODEL_IMPL_TYPE|NEW_MODEL_DESIGN|USE_|ATTN_|SKIP_JAX|MOE_|ONEHOT_|TPU_|RUNAI_)' | sort || true

# ---------------------------------------------------------------------------
# runai_streamer throughput.
#
# Build tc#943 streamed 63% of the 287119 tensors in 1:58:40 -- ~218 MB/s per
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
  --tokenizer "${TOKENIZER}" \
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
  --async-scheduling \
  ${VLLM_EXTRA_ARGS[@]+"${VLLM_EXTRA_ARGS[@]}"}
