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

#
# Launch a multi-host vllm serve on a TPU v4 pod slice WITHOUT Docker.
#
# Suitable for bare-VM setups where the venv and model are already in place
# on each host (e.g. via rsync_workers.sh).
#
# Usage (run on the head/coordinator node):
#   bash launch_ray_nodocker.sh
#
# Requirements:
#   - SSH key-based access from head to all worker nodes
#   - vllm-env with ray, vllm, tpu-inference installed on all hosts
#   - Model accessible at MODEL_PATH on all hosts (e.g. via GCS FUSE mount)
#   - TPU_MULTIHOST_BACKEND=ray support in tpu-inference
#
# Architecture for TP=4, EP=8 on v4-64 (8 hosts × 4 chips = 32 chips):
#   TP=4  divides attention heads (e.g. 20 heads / 4 = 5 per rank)
#   EP=8  distributes MoE experts across 8 host groups

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
# IP of this (head/coordinator) node, reachable by all workers.
: "${COORDINATOR_IP:?set COORDINATOR_IP to the head-host IP of this pod, reachable from every worker}"

# Ray port.
RAY_PORT="${RAY_PORT:-6379}"

# Ray dashboard is not needed for serving and can stall worker startup on
# some TPU hosts while waiting for metrics export ports.
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"

# Tensor parallel size (must divide the model's num_attention_heads).
TP_SIZE="${TP_SIZE:-4}"

# Expert parallel size (number of hosts; must divide num_routed_experts).
EP_SIZE="${EP_SIZE:-8}"

# Toggle MoE expert parallel routing. Set to 0 for a pure-TP baseline.
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

# Enable the new mesh-native MoE design path (Phase 1-4).
# When set to 1: uses true 5-axis global mesh, no legacy TPU_EP_RANK ownership,
# no router-logit slicing, mesh-native weight sharding via GMM_EP.
NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"

TP_BASELINE_SAFETY_MODE_SET="${TP_BASELINE_SAFETY_MODE+x}"
MAX_MODEL_LEN_SET="${MAX_MODEL_LEN+x}"
MAX_NUM_SEQS_SET="${MAX_NUM_SEQS+x}"
MAX_NUM_BATCHED_TOKENS_SET="${MAX_NUM_BATCHED_TOKENS+x}"
KV_CACHE_DTYPE_SET="${KV_CACHE_DTYPE+x}"
GPU_MEMORY_UTILIZATION_SET="${GPU_MEMORY_UTILIZATION+x}"
SAFETENSORS_LOAD_STRATEGY_SET="${SAFETENSORS_LOAD_STRATEGY+x}"
MOE_REQUANTIZE_EXPERT_CHUNK_SIZE_SET="${MOE_REQUANTIZE_EXPERT_CHUNK_SIZE+x}"

# Worker SSH targets (space-separated), excluding the head node.
# Use IPs by default because hostname resolution on this pod is not reliable.
: "${WORKERS:?set WORKERS to the space-separated IP list of the 7 non-coordinator hosts in this pod}"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no"
RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

# Path to the model (must be accessible on all hosts).
: "${MODEL_PATH:?set MODEL_PATH to the local path of the GLM-5.1-FP8 checkpoint (e.g. a gcsfuse mount reachable on every host)}"

# Path to the Python virtualenv.
: "${VENV:?set VENV to the Python venv with vllm + tpu-inference installed (must exist at the same path on every host)}"

# Maximum model sequence length.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"

# Cap concurrent requests so TPU warmup compiles stay within HBM headroom.
# User only needs batch size 1-2.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"

# Keep the prefill shape well below the model's context window so compile
# warmup, attention scratch buffers, and KV cache setup stay within the TPU
# v4 HBM/VMEM budget.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"

# Use FP8 KV cache to cut the per-layer cache footprint in half.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# GPU (TPU) memory utilization.  GLM-5.1-FP8 fills ~99.5% of HBM with
# weights so we must use 1.0 to leave any room for KV cache.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-1.0}"

# On network-mounted checkpoints, prefetch keeps pages hot while the default
# safetensors path still skips unused EP expert tensors before reading them.
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"

# Requantize FP8 MoE weights in smaller expert chunks to cap peak host memory
# during TP checkpoint loading. Set to 0 to force the legacy whole-layer path.
MOE_REQUANTIZE_EXPERT_CHUNK_SIZE="${MOE_REQUANTIZE_EXPERT_CHUNK_SIZE:-8}"

# Skip FP8→FP32→FP8 requantization for MoE weights. Keeps original checkpoint
# FP8 values + 2D block scales, avoiding quantization error from extra round-trip.
# The GMM kernel dequants FP8→BF16 in VMEM tile-by-tile using the original scales.
VLLM_MOE_SKIP_REQUANTIZATION="${VLLM_MOE_SKIP_REQUANTIZATION:-1}"

# Pure TP loads the full MoE checkpoint on every host. Keep that path in a more
# conservative mode by default so host page cache and TPU KV allocation do not
# become the bottleneck during baseline debugging.
TP_BASELINE_SAFETY_MODE="${TP_BASELINE_SAFETY_MODE:-$([ "$ENABLE_EXPERT_PARALLEL" = "0" ] && echo 1 || echo 0)}"
if [ "$TP_BASELINE_SAFETY_MODE" = "1" ]; then
  if [ -z "$MAX_MODEL_LEN_SET" ]; then
    MAX_MODEL_LEN="256"
  fi
  if [ -z "$MAX_NUM_SEQS_SET" ]; then
    MAX_NUM_SEQS="1"
  fi
  if [ -z "$MAX_NUM_BATCHED_TOKENS_SET" ]; then
    MAX_NUM_BATCHED_TOKENS="64"
  fi
  if [ -z "$KV_CACHE_DTYPE_SET" ]; then
    KV_CACHE_DTYPE="auto"
  fi
  if [ -z "$GPU_MEMORY_UTILIZATION_SET" ]; then
    GPU_MEMORY_UTILIZATION="0.9"
  fi
  if [ -z "$SAFETENSORS_LOAD_STRATEGY_SET" ]; then
    SAFETENSORS_LOAD_STRATEGY="prefetch"
  fi
  if [ -z "$MOE_REQUANTIZE_EXPERT_CHUNK_SIZE_SET" ]; then
    MOE_REQUANTIZE_EXPERT_CHUNK_SIZE="8"
  fi
fi

# Directory for logs.
LOG_DIR="${LOG_DIR:-/tmp/vllm_multih}"
# ──────────────────────────────────────────────────────────────────────────────

wait_for_port() {
  python - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket()
sock.settimeout(1)
try:
  sock.connect((host, port))
except OSError:
  sys.exit(1)
finally:
  sock.close()
PY
}

ray_node_count() {
  python -W ignore - "$1" "$2" <<'PY'
import sys
import ray

address = f"{sys.argv[1]}:{sys.argv[2]}"

try:
  ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
  print(sum(1 for node in ray.nodes() if node.get("Alive")))
finally:
  try:
    ray.shutdown()
  except Exception:
    pass
PY
}

echo "=== Multi-host vllm serve (no Docker) ==="
echo "  Coordinator : $COORDINATOR_IP:$RAY_PORT"
echo "  TP=$TP_SIZE  EP=$EP_SIZE  ($(echo $WORKERS | wc -w)+1 hosts)"
echo "  Enable EP   : $ENABLE_EXPERT_PARALLEL"
echo "  TP safe mode: $TP_BASELINE_SAFETY_MODE"
echo "  Model       : $MODEL_PATH"
echo "  Max seq len : $MAX_MODEL_LEN"
echo "  Dashboard   : $RAY_INCLUDE_DASHBOARD"
echo "  Max num seqs: $MAX_NUM_SEQS"
echo "  Max num tok : $MAX_NUM_BATCHED_TOKENS"
echo "  KV cache    : $KV_CACHE_DTYPE"
echo "  GPU mem util: $GPU_MEMORY_UTILIZATION"
echo "  ST load     : $SAFETENSORS_LOAD_STRATEGY"
echo "  MoE rq chunk: $MOE_REQUANTIZE_EXPERT_CHUNK_SIZE"

mkdir -p "$LOG_DIR"

# ── Step 0: Clean up existing processes ───────────────────────────────────────
echo "[$(date +%T)] Cleaning up existing processes..."
screen -S vllm_multih -X quit 2>/dev/null || true

for host in $WORKERS; do
  ssh $SSH_OPTS -o ConnectTimeout=5 "$host" \
    "source $VENV/bin/activate && ray stop --force 2>/dev/null || true; \
     pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true; \
     pkill -9 -f 'RayWorkerWrapper' 2>/dev/null || true; \
     fuser /dev/accel* 2>/dev/null | xargs -r kill -9 2>/dev/null || true; \
     rm -f /tmp/libtpu_lockfile" &
done
wait

source "$VENV/bin/activate"
export RAY_DEDUP_LOGS
ray stop --force 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "RayWorkerWrapper" 2>/dev/null || true
# Free any dangling TPU handles.  Without this, a stale process can hold
# /dev/accel* open even after pkill, producing a libtpu lockfile error on
# the next launch.
fuser /dev/accel* 2>/dev/null | xargs -r kill -9 2>/dev/null || true
rm -f /tmp/libtpu_lockfile
sleep 3

# ── Step 1: Start Ray HEAD on coordinator ─────────────────────────────────────
# RAY_num_prestart_python_workers=0 suppresses raylet's prewarm pool (Ray
# starts ~num_cpus Python interpreters per raylet for dynamic scheduling;
# vLLM spawns its own actors via placement groups and never uses the
# prewarm pool). Each prestart Python ~80 MB RSS + cold-cache paging slows
# boot substantially, especially on a host still flushing from a prior
# OOM. Set before `ray start` so the raylet inherits it.
export RAY_num_prestart_python_workers=0
# Ray prestarts num_cpus Python workers (services.py:1936). With 240 CPUs this
# spawns 240 python3.12 processes that each import vllm — an I/O storm that
# cripples post-OOM hosts. Cap at 16 (4 TPU chips × some headroom); actors use
# TPU/memory resources, not CPU slots.
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
# Raylet aborts if the dashboard agent can't write its port file in time.
# On hosts recovering from OOM (slow fs), the default (~60s) is too tight.
# Bumped to 5 min so the first-ever launch after an OOM still succeeds.
export RAY_agent_register_timeout_ms="${RAY_agent_register_timeout_ms:-300000}"
# CPU mem during multi-host FP8 MoE post-processing used to peak near 95%
# (JAX async device_put kept CPU source buffers pinned across all 75 MoE
# layers). Now that shard_moe_weights calls jax.block_until_ready before
# returning, the peak falls well under 95% and Ray's default threshold
# is safe. Keep a hook to override if a new model hits the ceiling.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
echo "[$(date +%T)] Starting Ray HEAD on $COORDINATOR_IP (num_cpus=$RAY_NUM_CPUS, agent_timeout=${RAY_agent_register_timeout_ms}ms)..."
ray start --head --port="$RAY_PORT" --dashboard-port=8265 \
  --num-cpus="$RAY_NUM_CPUS" \
  --include-dashboard="$RAY_INCLUDE_DASHBOARD" \
  2>&1 | tail -5

# Wait for GCS to be ready.
echo "[$(date +%T)] Waiting for Ray GCS..."
for _ in $(seq 1 10); do
  if wait_for_port "$COORDINATOR_IP" "$RAY_PORT"; then
    echo "[$(date +%T)] Ray HEAD is healthy."
    break
  fi
  sleep 2
done

# ── Step 2: Start Ray workers ─────────────────────────────────────────────────
echo "[$(date +%T)] Starting Ray workers: $WORKERS"
for host in $WORKERS; do
  ssh $SSH_OPTS -o ConnectTimeout=10 "$host" "
    source $VENV/bin/activate
    export RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS
    export RAY_num_prestart_python_workers=0
    export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-300000}
    export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.95}
    ${TPU_TP_SELECTIVE_LOAD:+export TPU_TP_SELECTIVE_LOAD=$TPU_TP_SELECTIVE_LOAD}
    ray stop --force 2>/dev/null || true
    rm -f /tmp/libtpu_lockfile
    # Drop page cache so post-OOM hosts don't re-fault hot paths
    sync 2>/dev/null || true
    sleep 1
    ray start --address=${COORDINATOR_IP}:${RAY_PORT} --num-cpus=${RAY_NUM_CPUS:-16} 2>&1 | tail -3
    echo 'Ray worker $host started'
  " 2>&1 | sed "s/^/[$host] /" &
done
wait
echo "[$(date +%T)] All Ray workers launched"

# Wait for all nodes to appear in the cluster.
NUM_EXPECTED=$(echo "$WORKERS" | wc -w)
NUM_EXPECTED=$((NUM_EXPECTED + 1))  # include head
echo "[$(date +%T)] Waiting for $NUM_EXPECTED nodes to join..."
for attempt in $(seq 1 60); do
  NUM_NODES=$(ray_node_count "$COORDINATOR_IP" "$RAY_PORT" 2>/dev/null || echo 0)
  echo "[$(date +%T)] Nodes connected: $NUM_NODES/$NUM_EXPECTED"
  if [ "$NUM_NODES" -ge "$NUM_EXPECTED" ]; then
    echo "[$(date +%T)] All $NUM_EXPECTED nodes ready!"
    break
  fi
  sleep 5
done

if [ "$NUM_NODES" -lt "$NUM_EXPECTED" ]; then
  echo "[$(date +%T)] ERROR: Only $NUM_NODES/$NUM_EXPECTED Ray nodes joined. Aborting launch."
  ray_node_count "$COORDINATOR_IP" "$RAY_PORT" 2>/dev/null || true
  exit 1
fi

ray_node_count "$COORDINATOR_IP" "$RAY_PORT" 2>/dev/null || true

# ── Step 3: Launch vllm serve on the coordinator ──────────────────────────────
echo "[$(date +%T)] Launching vllm serve (TP=$TP_SIZE, EP=$EP_SIZE)..."

EXPERT_PARALLEL_FLAG=""
EXPERT_PARALLEL_CONFIG=""
if [ "$ENABLE_EXPERT_PARALLEL" = "1" ]; then
  EXPERT_PARALLEL_FLAG="--enable-expert-parallel"
  # Optional: set ENABLE_DP_ATTENTION=1 to add DP-attention to sharding_strategy
  # (required by MLA models on the upstream path, cf. tpu_platform.py).
  if [ "${ENABLE_DP_ATTENTION:-0}" = "1" ]; then
    EXPERT_PARALLEL_CONFIG="--additional-config '{\"sharding\": {\"sharding_strategy\": {\"expert_parallelism\": $EP_SIZE, \"enable_dp_attention\": true}}}'"
  else
    EXPERT_PARALLEL_CONFIG="--additional-config '{\"sharding\": {\"sharding_strategy\": {\"expert_parallelism\": $EP_SIZE}}}'"
  fi
fi

PP_SIZE="${PP_SIZE:-1}"
PP_FLAG=""
if [ "$PP_SIZE" -gt 1 ]; then
  PP_FLAG="--pipeline-parallel-size $PP_SIZE"
fi

# Pre-compute the serve log path so callers can `tail -f $(cat current_log.txt)`
# without waiting for the screen session to name its own log.
SERVE_LOG="$LOG_DIR/serve_$(date +%Y%m%d_%H%M%S).log"
echo "$SERVE_LOG" > "$LOG_DIR/current_log.txt"

screen -dmS vllm_multih bash -c "
  source $VENV/bin/activate
  export RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS
  export TPU_MULTIHOST_BACKEND=ray
  export MOE_REQUANTIZE_EXPERT_CHUNK_SIZE=$MOE_REQUANTIZE_EXPERT_CHUNK_SIZE
  export VLLM_MOE_SKIP_REQUANTIZATION=$VLLM_MOE_SKIP_REQUANTIZATION
  export NEW_MODEL_DESIGN=$NEW_MODEL_DESIGN
  ${DEBUG_MOE_NORMS:+export DEBUG_MOE_NORMS=$DEBUG_MOE_NORMS}
  ${TPU_TP_SELECTIVE_LOAD:+export TPU_TP_SELECTIVE_LOAD=$TPU_TP_SELECTIVE_LOAD}
  ${TPU_GCS_WEIGHT_LOAD:+export TPU_GCS_WEIGHT_LOAD=$TPU_GCS_WEIGHT_LOAD}
  ${TPU_MIN_TOKEN_BUCKET:+export TPU_MIN_TOKEN_BUCKET=$TPU_MIN_TOKEN_BUCKET}
  ${EP_DEBUG_FULL:+export EP_DEBUG_FULL=$EP_DEBUG_FULL}
  ${TPU_TRUNCATE_LAYERS:+export TPU_TRUNCATE_LAYERS=$TPU_TRUNCATE_LAYERS}
  ${FORCE_LOGICAL_RESHAPE:+export FORCE_LOGICAL_RESHAPE=$FORCE_LOGICAL_RESHAPE}
  ${MLA_S_DTYPE_BF16:+export MLA_S_DTYPE_BF16=$MLA_S_DTYPE_BF16}
  # By default we rely on PJRT distributed to discover the physical
  # 8-host x 4-chip v4-64 topology on its own.  A caller can opt in to
  # explicit TPU_HOST_BOUNDS / TPU_CHIPS_PER_HOST_BOUNDS by exporting
  # them before running this script (useful for some single-host-per-
  # stage layouts).
  unset TPU_HOST_BOUNDS TPU_CHIPS_PER_HOST_BOUNDS
  ${TPU_HOST_BOUNDS_OVERRIDE:+export TPU_HOST_BOUNDS=$TPU_HOST_BOUNDS_OVERRIDE}
  ${TPU_CHIPS_PER_HOST_BOUNDS_OVERRIDE:+export TPU_CHIPS_PER_HOST_BOUNDS=$TPU_CHIPS_PER_HOST_BOUNDS_OVERRIDE}
  # TPU v4 MXU does not support FP8 matmul (Mosaic E2001).
  # gmm_v2 kernel handles FP8→BF16 dequant in Pallas VMEM tile-by-tile,
  # so weights stay as FP8 on HBM (~22.5GB/chip) which fits the budget.
  cd "${SERVE_CWD:-$HOME}"
  exec vllm serve $MODEL_PATH \
    --tensor-parallel-size $TP_SIZE \
    $PP_FLAG \
    $EXPERT_PARALLEL_FLAG \
    $EXPERT_PARALLEL_CONFIG \
    ${TPU_TP_SELECTIVE_LOAD:+--load-format tpu_streaming_loader} \
    --safetensors-load-strategy $SAFETENSORS_LOAD_STRATEGY \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    ${DISABLE_PREFIX_CACHE:+--no-enable-prefix-caching} \
    --port 8000 \
    2>&1 | tee $SERVE_LOG
"

echo ""
echo "[$(date +%T)] vllm serve launched in screen session 'vllm_multih'"
echo "  Log     : $SERVE_LOG"
echo "  Pointer : $LOG_DIR/current_log.txt  -> $(cat $LOG_DIR/current_log.txt)"
echo "  Monitor : tail -f \$(cat $LOG_DIR/current_log.txt)"
echo "  Ray     : ray status"
echo "  Screen  : screen -r vllm_multih"
