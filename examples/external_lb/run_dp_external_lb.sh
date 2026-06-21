#!/usr/bin/env bash
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

# Launch N fully independent vllm-serve instances (ports BASE_PORT..
# BASE_PORT+DP_SIZE-1), each pinned to its own slice of TPU chips, plus
# rr_proxy.py as a pure round-robin reverse proxy on top (PROXY_PORT,
# default 8080).
#
# Why this exists: vLLM's internal DP load balancer scores ranks off
# waiting/running counts that a coordinator only refreshes every 100ms
# (hardcoded). Under bursty/symmetric load this snowballs into uneven
# per-rank request counts. external-lb + a real round-robin proxy removes
# the coordinator from the critical path entirely. See b/525257493 for more
# context.
#
# Usage:
#   [ENV_VAR=...] ./run_dp_external_lb.sh [serve] <model> [extra vllm serve flags...]
# Example (Gemma-4 26B-A4B MoE):
#   MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
#   VLLM_ENGINE_READY_TIMEOUT_S=6000 \
#   USE_BATCHED_RPA_KERNEL=1 \
#   MODEL_IMPL_TYPE=flax_nnx \
#   ./run_dp_external_lb.sh google/gemma-4-26B-A4B-it \
#       --max-model-len=1524 --max-num-seqs=1024 --data-parallel-size 4 \
#       --tensor-parallel-size 2
# --data-parallel-size is read to know how many ranks to launch, then
# stripped (along with any other --data-parallel-* flag) before forwarding
# to vllm serve.

set -ue

usage() {
  echo "Usage: [ENV_VAR=...] $0 [serve] <model> [extra vllm serve flags...]" >&2
  exit 1
}

[ $# -lt 1 ] && usage

# Accept (and ignore) a leading "serve" token so the invocation can mirror
# plain `vllm serve <model> ...` -- this script always runs serve mode.
if [ "$1" = "serve" ]; then
  shift
fi
[ $# -lt 1 ] && usage

MODEL=$1
shift
RAW_ARGS=("$@")

# Pull --data-parallel-size and --tensor-parallel-size out of the passed
# flags: DP_SIZE tells this script how many ranks to launch, TP_SIZE feeds
# the chip-pinning math below. Both fall back to env-var / hardcoded
# defaults if the caller didn't pass them explicitly.
DP_SIZE=${DP_SIZE:-4}
TP_SIZE=${TP_SIZE:-2}
for ((i = 0; i < ${#RAW_ARGS[@]}; i++)); do
  arg=${RAW_ARGS[$i]}
  case "$arg" in
    --data-parallel-size=*)
      DP_SIZE=${arg#--data-parallel-size=}
      ;;
    --data-parallel-size)
      DP_SIZE=${RAW_ARGS[$((i + 1))]}
      ;;
    --tensor-parallel-size=*)
      TP_SIZE=${arg#--tensor-parallel-size=}
      ;;
    --tensor-parallel-size)
      TP_SIZE=${RAW_ARGS[$((i + 1))]}
      ;;
  esac
done

# Strip every --data-parallel-* flag (and its value, if space-separated) --
# each rank below is a fully independent, non-DP-aware vllm serve instance.
# See "Why no --data-parallel-* flags" above. Also strip
# --tensor-parallel-size since it's re-added explicitly below (already
# parsed into TP_SIZE above) -- stripping just avoids passing it twice.
EXTRA_ARGS=()
skip_next=false
for arg in "${RAW_ARGS[@]}"; do
  if $skip_next; then
    skip_next=false
    continue
  fi
  case "$arg" in
    # Boolean (store_true) data-parallel flags take no value -- don't skip
    # the next arg for these.
    --data-parallel-hybrid-lb | --data-parallel-external-lb | \
    --data-parallel-multi-port-external-lb)
      ;;
    --data-parallel-*=*)
      ;;
    --data-parallel-*)
      skip_next=true
      ;;
    --tensor-parallel-size=*)
      ;;
    --tensor-parallel-size)
      skip_next=true
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

# Physical TPU cores per chip (v7x = 2). Each rank needs
# ceil(TP_SIZE / CORES_PER_CHIP) whole chips; with TP_SIZE=2 and
# CORES_PER_CHIP=2 that's exactly 1 chip per rank.
CORES_PER_CHIP=${CORES_PER_CHIP:-2}
CHIPS_PER_RANK=$(( (TP_SIZE + CORES_PER_CHIP - 1) / CORES_PER_CHIP ))

# PROXY_PORT defaults to 8000 (vllm bench serve's own default --port) so
# benchmark clients need no extra --port flag; rank servers start above it.
BASE_PORT=${BASE_PORT:-8100}
PROXY_PORT=${PROXY_PORT:-8000}
STAGGER_S=${STAGGER_S:-20}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
LOG_DIR=${LOG_DIR:-dp_external_lb_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_DIR"
echo "--- Logs -> $LOG_DIR ---"
echo "--- DP_SIZE=${DP_SIZE} TP_SIZE=${TP_SIZE} CORES_PER_CHIP=${CORES_PER_CHIP} CHIPS_PER_RANK=${CHIPS_PER_RANK} ---"

RANK_PORTS=()
RANK_PIDS=()
PROXY_PID=

wait_for_server() {
  local port=$1
  local pid=$2
  timeout 1200 bash -c "
    until curl -s localhost:${port}/health > /dev/null; do
      if ! kill -0 $pid 2>/dev/null; then
        echo \"Error: vLLM server on port $port (PID $pid) crashed or failed to start!\" >&2
        exit 1
      fi
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM/proxy instances..."
  pkill -f "rr_proxy.py" || true
  pkill -f "vllm serve ${MODEL}" || true
  sleep 5
  pkill -9 -f "rr_proxy.py" || true
  pkill -9 -f "vllm serve ${MODEL}" || true
  fuser -k -9 /dev/vfio/* || true
  fuser -k -9 /dev/accel* || true
  rm -rf /tmp/jax_cache_* || true
  rm -f /tmp/libtpu_lockfile || true
}

print_logs_on_exit() {
  echo "--- Script exiting, displaying logs ---"
  for i in "${!RANK_PORTS[@]}"; do
    f="$LOG_DIR/rank${i}.txt"
    echo "--- Tail of $f ---"
    [ -f "$f" ] && tail -n 30 "$f" || echo "File not found."
  done
  echo "--- Tail of $LOG_DIR/proxy.txt ---"
  [ -f "$LOG_DIR/proxy.txt" ] && tail -n 30 "$LOG_DIR/proxy.txt" || echo "File not found."
  echo "--- End of logs ---"
  cleanup_instances
}
trap print_logs_on_exit EXIT

cleanup_instances

echo "--- Starting ${DP_SIZE} independent ranks (staggered ${STAGGER_S}s apart) ---"
for ((rank = 0; rank < DP_SIZE; rank++)); do
  port=$((BASE_PORT + rank))
  chip_start=$((rank * CHIPS_PER_RANK))
  chip_end=$((chip_start + CHIPS_PER_RANK - 1))
  visible_chips=$(seq -s, "$chip_start" "$chip_end")

  TPU_VISIBLE_CHIPS="${visible_chips}" \
  TPU_CHIPS_PER_PROCESS_BOUNDS="1,${CHIPS_PER_RANK},1" \
  TPU_PROCESS_BOUNDS=1,1,1 \
  vllm serve "${MODEL}" \
    "${EXTRA_ARGS[@]}" \
    --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    > "$LOG_DIR/rank${rank}.txt" 2>&1 &

  RANK_PORTS+=("$port")
  RANK_PIDS+=("$!")
  echo "rank=${rank} port=${port} chips=${visible_chips} pid=$!"

  if ((rank < DP_SIZE - 1)); then
    sleep "${STAGGER_S}"
  fi
done

for i in "${!RANK_PORTS[@]}"; do
  port=${RANK_PORTS[$i]}
  echo "Waiting for rank $i on port $port to start..."
  wait_for_server "$port" "${RANK_PIDS[$i]}" || exit 1
done

echo "--- Starting round-robin proxy on port ${PROXY_PORT} ---"
BACKEND_BASE_PORT="${BASE_PORT}" BACKEND_COUNT="${DP_SIZE}" \
  python3 "$SCRIPT_DIR/rr_proxy.py" "${PROXY_PORT}" \
  > "$LOG_DIR/proxy.txt" 2>&1 &
PROXY_PID=$!
wait_for_server "${PROXY_PORT}" "${PROXY_PID}" || exit 1
echo "Proxy healthy on port ${PROXY_PORT} (pid=${PROXY_PID})"

cat <<EOF

The round-robin proxy is listening on: 127.0.0.1:${PROXY_PORT}

>> Send an example request:

curl http://localhost:${PROXY_PORT}/v1/chat/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "model": "${MODEL}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
        "max_tokens": 25,
        "temperature": 0.0
}'

>> Stop the proxy and all rank servers:

pkill -f "vllm serve ${MODEL}" && pkill -f "rr_proxy.py"
EOF

echo "Press Ctrl-C to stop everything (trap will clean up)."
wait
