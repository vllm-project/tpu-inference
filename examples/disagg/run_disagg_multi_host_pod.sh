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

# Multi-host DCN disaggregation, run as processes in one pod.
#
# The same test as run_disagg_multi_host.sh, which starts one docker container
# per TPU process. That script cannot run here: it drives `docker run`, and a
# workload pod has no docker daemon and runs under PodSecurity `baseline`,
# which rejects the --privileged it asks for.
#
# The conversion is small, because "multi-host" here means multiple TPU
# processes rather than multiple machines. All eight containers already run
# --network host and address each other as 127.0.0.1, so a pod - one network
# namespace, all 8 chips of a ct6e-standard-8t - is the same environment with
# the container boundaries removed. `docker run` becomes a background process
# and `docker exec` becomes a plain command.
#
#   prefill  4 processes, chips 0-3, Ray cluster on 8100, vLLM on 8400
#   decode   4 processes, chips 4-7, Ray cluster on 9100, vLLM on 9400
#   proxy    1 process, port 8000, no chips
#
# Two things the container boundary was providing for free, which have to be
# arranged explicitly now:
#
#   Ray temp dirs, one per process. A Ray node keeps its session directory and
#   its raylet and plasma sockets under --temp-dir, so two raylets sharing one
#   collide. The docker script gets this free from the mount namespace: even
#   with --network host, each container has a private /tmp.
#
#   Process identity. TPU_VISIBLE_CHIPS, CLOUD_TPU_TASK_ID and TPU_PROCESS_PORT
#   differ per process and were per-container env. Here they are set per
#   command, which is why each `ray start` is wrapped in `env`.

# shellcheck disable=all
set -e

MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
TPU_VERSION=${TPU_VERSION:=tpu6e}
INPUT_LEN=${INPUT_LEN:=128}
OUTPUT_LEN=${OUTPUT_LEN:=20}
NUM_PROMPTS=${NUM_PROMPTS:=100}
RANDOM_SEED=${RANDOM_SEED:=10}
MAX_CONCURRENCY=${MAX_CONCURRENCY:=10}
# 1 benchmark, 2 correctness, 3 both. Same meaning as the docker script.
TEST_MODE=${TEST_MODE:=1}

LOG_DIR=${LOG_DIR:-$HOME/logs}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# v6e has 4 chips per instance; tpu7x has 2. Mirrors run_disagg_multi_host.sh.
NUM_HOSTS_PER_INSTANCE=4
TPU_PROCESS_BOUNDS="2,2,1"
PREFILL_TPU_PORTS=(8476 8477 8478 8479)
DECODE_TPU_PORTS=(9476 9477 9478 9479)
if [ "$TPU_VERSION" = "tpu7x" ]; then
  NUM_HOSTS_PER_INSTANCE=2
  TPU_PROCESS_BOUNDS="1,2,1"
  PREFILL_TPU_PORTS=(8476 8477)
  DECODE_TPU_PORTS=(9476 9477)
fi

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

# Wait for an HTTP service, failing early if the process behind it died. The
# docker script did this with `docker exec ... kill -0`; here the PID is ours.
wait_for_server() {
  local port=$1 pid=$2 name=$3 log=$4
  echo "Waiting for $name on port $port (pid $pid)..."
  local end=$((SECONDS + 900))
  while [ $SECONDS -lt $end ]; do
    # 127.0.0.1, not localhost: localhost resolves to ::1 first and the pod
    # has no IPv6 loopback, which is what broke single-host disagg on kube.
    if curl -fs --max-time 5 "127.0.0.1:${port}/health" >/dev/null; then
      echo "=== $name healthy on port $port ==="
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: $name (pid $pid) died before becoming healthy." >&2
      [ -f "$log" ] && tail -80 "$log"
      return 1
    fi
    sleep 2
  done
  echo "Error: $name did not become healthy within the timeout." >&2
  [ -f "$log" ] && tail -80 "$log"
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
#
# `ray start --block` stays in the foreground, so each process goes to the
# background here - the docker script got the same effect from `docker run -d`.
start_instance() {
  local role=$1 ray_port=$2 chip_base=$3 kv_base=$4 tmpdir=$5 vllm_port=$6 kv_role=$7
  shift 7
  local ports=("$@")
  local addrs=()
  for p in "${ports[@]}"; do addrs+=("127.0.0.1:$p"); done
  local joined
  joined=$(IFS=, ; echo "${addrs[*]}")

  for ((i=0; i<NUM_HOSTS_PER_INSTANCE; i++)); do
    # One temp dir per process, not per cluster. A Ray node keeps its session
    # directory, raylet socket and plasma socket under --temp-dir, so raylets
    # sharing one collide and workers attach to the wrong raylet, and the
    # prefill workers deadlock in xla_bridge backend init racing for one chip.
    #
    # The docker script gets this from the mount namespace: --network host
    # shares the network but each container still has a private /tmp, so every
    # raylet already had its own /tmp/ray. That is also what a real Ray cluster
    # looks like - one session dir per node. Only the network was ever shared.
    local cmd="ray start --block --temp-dir=${tmpdir}-${i}"
    if [ "$i" -eq 0 ]; then
      cmd="$cmd --head --port=${ray_port}"
      [ "$role" = "decode" ] && cmd="$cmd --min-worker-port=20000 --max-worker-port=29999"
    else
      cmd="$cmd --address=127.0.0.1:${ray_port}"
    fi
    env \
      # 0, not the 1 run.sh exports for every other step. This test sets
      # SKIP_JAX_PRECOMPILE=1, so the first real request necessarily compiles,
      # and the check exists to fail a run that does exactly that: every
      # request fails with "JAX compilation occurred".
      #
      # Bare metal never sees it: `docker run` passes an explicit env list that
      # omits the flag, so the containers start clean. Processes in a pod
      # inherit the pod's environment instead. run_disagg_single_host.sh sets
      # it to 0 for the same reason.
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

  # vLLM attaches to node 0 of that cluster, exactly as `docker exec ...-0` did.
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
      --tensor-parallel-size "${NUM_HOSTS_PER_INSTANCE}" \
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

# 127.0.0.1 rather than localhost: localhost resolves to ::1 first, and the
# pod has no IPv6 loopback. That is what broke single-host disagg on kube.
python3 "$SCRIPT_DIR/toy_proxy_server.py" --host 127.0.0.1 --port "$PROXY_PORT" \
  >"$LOG_DIR/proxy.txt" 2>&1 &
PROXY_PID=$!
PIDS+=($PROXY_PID)
wait_for_server "$PROXY_PORT" "$PROXY_PID" "toy_proxy_server" "$LOG_DIR/proxy.txt"

if [ "$TEST_MODE" = "1" ] || [ "$TEST_MODE" = "3" ]; then
  echo "--- benchmark ---"
  vllm bench serve \
    --backend vllm --host 127.0.0.1 --port "$PROXY_PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" \
    --request-rate inf --max-concurrency "$MAX_CONCURRENCY" \
    --trust-remote-code --seed "$RANDOM_SEED" \
    >"$LOG_DIR/benchmark.txt" 2>&1
  check_failed_requests "$LOG_DIR/benchmark.txt"
fi

if [ "$TEST_MODE" = "2" ] || [ "$TEST_MODE" = "3" ]; then
  echo "--- correctness ---"
  python3 "$SCRIPT_DIR/test_disagg_correctness.py" \
    --baseline_url "http://127.0.0.1:${DECODE_VLLM_PORT}/v1/completions" \
    --disagg_url "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
    --model "$MODEL" --num_requests "$NUM_PROMPTS" \
    --input_length "$INPUT_LEN" --output_length "$OUTPUT_LEN" \
    >"$LOG_DIR/correctness.txt" 2>&1
fi

echo "--- done ---"
