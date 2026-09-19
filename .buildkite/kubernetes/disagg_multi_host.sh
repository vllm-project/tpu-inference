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

# Multi-host DCN disaggregation as processes, not containers: a workload pod has
# no docker daemon and runs under PodSecurity `baseline`, which rejects the
# --privileged that run_disagg_multi_host.sh's `docker run` asks for.
#
#   prefill  4 processes, chips 0-3, Ray cluster on 8100, vLLM on 8400
#   decode   4 processes, chips 4-7, Ray cluster on 9100, vLLM on 9400
#   proxy    1 process, port 8000, no chips
#
# TPU_VISIBLE_CHIPS, CLOUD_TPU_TASK_ID and TPU_PROCESS_PORT differ per process,
# hence the `env` wrapper on each command below.

# shellcheck disable=all
set -e

MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
TPU_VERSION=${TPU_VERSION:=tpu6e}
INPUT_LEN=${INPUT_LEN:=128}
OUTPUT_LEN=${OUTPUT_LEN:=20}
NUM_PROMPTS=${NUM_PROMPTS:=100}
RANDOM_SEED=${RANDOM_SEED:=10}
MAX_CONCURRENCY=${MAX_CONCURRENCY:=10}

LOG_DIR=${LOG_DIR:-$HOME/logs}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

NUM_HOSTS_PER_INSTANCE=4
CORES_PER_CHIP=1
TPU_PROCESS_BOUNDS="2,2,1"
PREFILL_TPU_PORTS=(8476 8477 8478 8479)
DECODE_TPU_PORTS=(9476 9477 9478 9479)
if [ "$TPU_VERSION" = "tpu7x" ]; then
  NUM_HOSTS_PER_INSTANCE=2
  CORES_PER_CHIP=2
  TPU_PROCESS_BOUNDS="1,2,1"
  PREFILL_TPU_PORTS=(8476 8477)
  DECODE_TPU_PORTS=(9476 9477)
fi

# One process per chip, but the mesh is over cores and a v7x chip carries two.
# Sized by the process count, half the devices fall outside the mesh and the
# workers holding them come up with no HBM.
TENSOR_PARALLEL_SIZE=$(( NUM_HOSTS_PER_INSTANCE * CORES_PER_CHIP ))

PREFILL_RAY_PORT=8100
DECODE_RAY_PORT=9100
PREFILL_VLLM_PORT=8400
DECODE_VLLM_PORT=9400
PROXY_PORT=8000
COMMON_SIDE_PORT=8900

PIDS=()

dump_logs() {
  echo "--- Script exiting, displaying logs ---"
  for f in prefill.txt decode.txt proxy.txt benchmark.txt correctness.txt; do
    echo "--- $LOG_DIR/$f ---"
    [ -f "$LOG_DIR/$f" ] && cat "$LOG_DIR/$f" || echo "File not found."
  done
  echo "--- ray ---"
  for d in /tmp/ray-prefill* /tmp/ray-decode*; do
    [ -d "$d" ] && find "$d" -name "raylet.err" -exec tail -20 {} + 2>/dev/null || true
  done
  echo "--- End of logs ---"
}

cleanup() {
  set +e
  dump_logs
  # Ray first: it supervises the per-chip workers, and killing it before the
  # vLLM servers avoids a page of connection errors in the logs above.
  ray stop --force >/dev/null 2>&1
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
  sleep 3
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f toy_proxy_server 2>/dev/null
  rm -f /tmp/libtpu_lockfile
}
trap cleanup EXIT

# EX_TEMPFAIL. The step retries once on exactly this code, so it has to mean
# "the chips were not ready", never "the test failed".
EXIT_TEMPFAIL=75

# Still a heuristic: START_SESSION also fails deterministically for a wrong
# process-bounds setting or chips a previous pod never released, which is why
# the step's retry is capped at one.
exit_if_transient_tpu_init_failure() {
  local log=$1 name=$2
  [ -f "$log" ] || return 0
  if grep -qF -e 'TPU initialization failed: GRPC_ERROR' \
               -e 'START_SESSION failed' "$log" 2>/dev/null; then
    echo "[disagg-harness] ${name}: TPU runtime session failure; exiting ${EXIT_TEMPFAIL} so the step retries." >&2
    exit "$EXIT_TEMPFAIL"
  fi
}

wait_for_server() {
  local port=$1 pid=$2 name=$3 log=$4
  echo "Waiting for $name on port $port (pid $pid)..."
  local end=$((SECONDS + 900))
  while [ $SECONDS -lt $end ]; do
    if curl -fs --max-time 5 "127.0.0.1:${port}/health" >/dev/null; then
      echo "=== $name healthy on port $port ==="
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: $name (pid $pid) died before becoming healthy." >&2
      [ -f "$log" ] && tail -80 "$log"
      exit_if_transient_tpu_init_failure "$log" "$name"
      return 1
    fi
    sleep 2
  done
  echo "Error: $name did not become healthy within the timeout." >&2
  [ -f "$log" ] && tail -80 "$log"
  exit_if_transient_tpu_init_failure "$log" "$name"
  return 1
}

check_failed_requests() {
  local failed
  failed=$(grep "Failed requests:" "$1" | awk '{print $3}' || true)
  if [ -z "$failed" ]; then
    echo "Error: no 'Failed requests:' line in the benchmark output." >&2
    return 1
  fi
  if [ "$failed" -gt 0 ]; then
    echo "Error: benchmark reported $failed failed requests." >&2
    return 1
  fi
  echo "Success: benchmark reported $failed failed requests."
}

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/{prefill,decode,proxy,benchmark,correctness}.txt
rm -rf /tmp/ray-prefill* /tmp/ray-decode* /tmp/libtpu_lockfile
ray stop --force >/dev/null 2>&1 || true

# One Ray cluster per instance, then the vLLM server that drives it.
# `ray start --block` stays in the foreground, hence the backgrounding.
start_instance() {
  local role=$1 ray_port=$2 chip_base=$3 kv_base=$4 tmpdir=$5 vllm_port=$6 kv_role=$7
  shift 7
  local ports=("$@")
  local addrs=()
  for p in "${ports[@]}"; do addrs+=("127.0.0.1:$p"); done
  local joined
  joined=$(IFS=, ; echo "${addrs[*]}")

  for ((i=0; i<NUM_HOSTS_PER_INSTANCE; i++)); do
    # One temp dir per process, not per cluster: a Ray node keeps its session
    # directory, raylet socket and plasma socket under --temp-dir, so raylets
    # sharing one collide and workers attach to the wrong raylet.
    local cmd="ray start --block --temp-dir=${tmpdir}-${i}"
    if [ "$i" -eq 0 ]; then
      cmd="$cmd --head --port=${ray_port}"
      [ "$role" = "decode" ] && cmd="$cmd --min-worker-port=20000 --max-worker-port=29999"
    else
      cmd="$cmd --address=127.0.0.1:${ray_port}"
    fi
    # SKIP_JAX_PRECOMPILE=1 means the first real request necessarily compiles,
    # which VLLM_XLA_CHECK_RECOMPILATION exists to fail; it is inherited from
    # the pod environment, so it has to be overridden per command.
    env \
      VLLM_XLA_CHECK_RECOMPILATION=0 \
      TPU_MULTIHOST_BACKEND=ray \
      TPU_NODE_ID="${i}" \
      TPU_KV_TRANSFER_PORT="$(( kv_base + i ))" \
      TPU_SIDE_CHANNEL_PORT="$(( COMMON_SIDE_PORT + i ))" \
      RAY_DEDUP_LOGS=0 \
      SKIP_JAX_PRECOMPILE=1 \
      TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1" \
      TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS}" \
      TPU_VISIBLE_CHIPS="$(( chip_base + i ))" \
      CLOUD_TPU_TASK_ID="${i}" \
      TPU_PROCESS_ADDRESSES="${joined}" \
      TPU_PROCESS_PORT="${ports[$i]}" \
      $cmd >>"$LOG_DIR/ray-${role}.txt" 2>&1 &
    PIDS+=($!)
    sleep 1
  done

  echo "--- started $role Ray cluster: ${NUM_HOSTS_PER_INSTANCE} processes, chips ${chip_base}-$(( chip_base + NUM_HOSTS_PER_INSTANCE - 1 )) ---"

  env \
    VLLM_XLA_CHECK_RECOMPILATION=0 \
    TPU_MULTIHOST_BACKEND=ray \
    TPU_NODE_ID=0 \
    TPU_KV_TRANSFER_PORT="${kv_base}" \
    TPU_SIDE_CHANNEL_PORT="${COMMON_SIDE_PORT}" \
    RAY_ADDRESS="127.0.0.1:${ray_port}" \
    RAY_DEDUP_LOGS=0 \
    SKIP_JAX_PRECOMPILE=1 \
    TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1" \
    TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS}" \
    TPU_VISIBLE_CHIPS="${chip_base}" \
    CLOUD_TPU_TASK_ID=0 \
    TPU_PROCESS_ADDRESSES="${joined}" \
    TPU_PROCESS_PORT="${ports[0]}" \
    vllm serve "$MODEL" \
      --port "${vllm_port}" \
      --gpu-memory-utilization 0.8 \
      --no-enable-prefix-caching \
      --max-num-batched-tokens 1024 \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"${kv_role}\"}" \
      >"$LOG_DIR/${role}.txt" 2>&1 &
  PIDS+=($!)
  eval "${role^^}_VLLM_PID=$!"
}

start_instance prefill "$PREFILL_RAY_PORT" 0 8200 /tmp/ray-prefill \
  "$PREFILL_VLLM_PORT" kv_producer "${PREFILL_TPU_PORTS[@]}"
start_instance decode "$DECODE_RAY_PORT" "$NUM_HOSTS_PER_INSTANCE" 9200 /tmp/ray-decode \
  "$DECODE_VLLM_PORT" kv_consumer "${DECODE_TPU_PORTS[@]}"

wait_for_server "$PREFILL_VLLM_PORT" "$PREFILL_VLLM_PID" "prefill vllm" "$LOG_DIR/prefill.txt"
wait_for_server "$DECODE_VLLM_PORT" "$DECODE_VLLM_PID" "decode vllm" "$LOG_DIR/decode.txt"

# 127.0.0.1, not localhost: localhost resolves to ::1 first and the pod has no
# IPv6 loopback.
python3 "$SCRIPT_DIR/toy_proxy_server.py" --host 127.0.0.1 --port "$PROXY_PORT" \
  >"$LOG_DIR/proxy.txt" 2>&1 &
PROXY_PID=$!
PIDS+=($PROXY_PID)
wait_for_server "$PROXY_PORT" "$PROXY_PID" "toy_proxy_server" "$LOG_DIR/proxy.txt"

echo "--- benchmark ---"
vllm bench serve \
  --backend vllm --host 127.0.0.1 --port "$PROXY_PORT" --model "$MODEL" \
  --dataset-name random --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" \
  --request-rate inf --max-concurrency "$MAX_CONCURRENCY" \
  --trust-remote-code --seed "$RANDOM_SEED" \
  >"$LOG_DIR/benchmark.txt" 2>&1
check_failed_requests "$LOG_DIR/benchmark.txt"

echo "--- done ---"
