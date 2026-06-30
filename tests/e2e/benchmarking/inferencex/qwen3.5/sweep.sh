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

# Sweep: restart the server per concurrency.
# Usage: bash sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers a cold compile)

WORKLOADS=("8192:1024" "1024:1024")
CONC_SHARDING=(
  "4:DP4TP2_EP"
  "8:DP4TP2_EP"
  "16:DP4TP2_EP"
  "32:DP4TP2_EP"
  "64:DP8_EP"
  "128:DP8_EP"
  "256:DP8_EP"
)

stop_server() {
  pkill -TERM -f "vllm serve Qwen/Qwen3.5" 2>/dev/null || true
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
  pkill -KILL -f "vllm serve Qwen/Qwen3.5" 2>/dev/null || true
  pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
  for _ in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://0.0.0.0:${PORT}/health" 2>/dev/null)" = "200" ] || break
    sleep 2
  done
}

start_server() {  # $1 = SHARDING (CONC/ISL/OSL come from the exported env)
  SERVER_LOG="/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
  echo "--- starting server SHARDING=$1 ISL=$ISL OSL=$OSL CONC=$CONC (log: $SERVER_LOG) ---"
  SHARDING="$1" bash "${SCRIPT_DIR}/server.sh" >> "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server exited during startup (see $SERVER_LOG)" >&2; return 1; }
    sleep 5; waited=$((waited + 5))
    [ "$waited" -ge "$READY_TIMEOUT" ] && { echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2; return 1; }
  done
  echo "--- server ready ---"
}

trap stop_server EXIT

for wl in "${WORKLOADS[@]}"; do
  export ISL="${wl%%:*}" OSL="${wl#*:}"
  for cs in "${CONC_SHARDING[@]}"; do
    export CONC="${cs%%:*}"
    sharding="${cs#*:}"
    stop_server
    start_server "$sharding" || exit 1
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC SHARDING=$sharding ##########"
    bash "${SCRIPT_DIR}/bench.sh"
  done
done
echo "########## sweep complete ##########"
