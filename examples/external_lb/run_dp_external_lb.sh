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

# Launch vLLM with --data-parallel-external-lb: one independent vllm-serve
# process per DP rank (ports BASE_PORT..BASE_PORT+DP_SIZE-1), plus
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
#   MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
#   VLLM_ENGINE_READY_TIMEOUT_S=6000 \
#   USE_BATCHED_RPA_KERNEL=1 \
#   MODEL_IMPL_TYPE=flax_nnx \
#   ./run_dp_external_lb.sh google/gemma-4-26B-A4B-it \
#       --max-model-len=1524 --max-num-seqs=1024
# Any env vars set before the command are inherited as-is (this script does
# not hardcode model-specific ones like USE_BATCHED_RPA_KERNEL --
# only TPU_MULTIPROCESS_DP=1 is forced, since that's required for
# --data-parallel-external-lb itself, not model-specific).

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
EXTRA_ARGS=("$@")

# Pull --data-parallel-size out of the passed-through flags (this script
# needs it to know how many ranks to launch); falls back to DP_SIZE env var
# / 4 if the caller didn't pass it explicitly.
DP_SIZE=${DP_SIZE:-4}
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg=${EXTRA_ARGS[$i]}
  case "$arg" in
    --data-parallel-size=*)
      DP_SIZE=${arg#--data-parallel-size=}
      ;;
    --data-parallel-size)
      DP_SIZE=${EXTRA_ARGS[$((i + 1))]}
      ;;
  esac
done

BASE_PORT=${BASE_PORT:-8000}
PROXY_PORT=${PROXY_PORT:-8080}
STAGGER_S=${STAGGER_S:-20}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
LOG_DIR=${LOG_DIR:-dp_external_lb_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_DIR"
echo "--- Logs -> $LOG_DIR ---"

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

echo "--- Starting ${DP_SIZE} external-LB ranks (staggered ${STAGGER_S}s apart) ---"
for ((rank = 0; rank < DP_SIZE; rank++)); do
  port=$((BASE_PORT + rank))

  # Requires TPU_MULTIPROCESS_DP=1 forced per child -- without it,
  # tpu_inference's tpu_worker.py::_setup_dp_chip_isolation never runs
  # (because dp_supervisor-style external-lb children never get
  # _api_process_rank=-1, so TpuPlatform._resolve_multiprocess_dp's
  # online_serving heuristic resolves to False) and the processes either
  # collide on /tmp/libtpu_lockfile or hit a chip-topology mismatch.
  TPU_MULTIPROCESS_DP=1 \
  vllm serve "${MODEL}" \
    "${EXTRA_ARGS[@]}" \
    --port "${port}" \
    --data-parallel-rank "${rank}" \
    --data-parallel-external-lb \
    > "$LOG_DIR/rank${rank}.txt" 2>&1 &

  RANK_PORTS+=("$port")
  RANK_PIDS+=("$!")
  echo "rank=${rank} port=${port} pid=$!"

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
