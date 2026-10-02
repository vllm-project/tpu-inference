#!/bin/bash
# Copyright 2025 Google LLC
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


# shellcheck disable=all
set -e

print_logs_on_exit() {
  echo "--- Script exiting, displaying logs ---"

  LOG_DIR=$HOME/logs

  if [ -d "$LOG_DIR" ]; then
    echo "--- Contents of $LOG_DIR/prefill_0.txt ---"
    if [ -f "$LOG_DIR/prefill_0.txt" ]; then
      cat "$LOG_DIR/prefill_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/decode_0.txt ---"
    if [ -f "$LOG_DIR/decode_0.txt" ]; then
      cat "$LOG_DIR/decode_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/proxy_0.txt ---"
    if [ -f "$LOG_DIR/proxy_0.txt" ]; then
      cat "$LOG_DIR/proxy_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/benchmark_0.txt ---"
    if [ -f "$LOG_DIR/benchmark_0.txt" ]; then
      cat "$LOG_DIR/benchmark_0.txt"
    else
      echo "File not found."
    fi
  else
    echo "Log directory '$LOG_DIR' not found."
  fi
  echo "--- End of logs ---"
}

trap print_logs_on_exit EXIT

MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
INPUT_LEN=${INPUT_LEN:=512}
OUTPUT_LEN=${OUTPUT_LEN:=128}
NUM_PROMPTS=${NUM_PROMPTS:=200}
REQUEST_RATE=${REQUEST_RATE:=4}


NUM_PREFILL_INSTANCES=1
NUM_DECODE_INSTANCES=1
TPU_VERSION=${TPU_VERSION:=tpu7x}
if [ "${TPU_VERSION:-}" = "tpu7x" ]; then
    PREFILLER_TP_SIZE=2
    DECODER_TP_SIZE=2
else
    PREFILLER_TP_SIZE=1
    DECODER_TP_SIZE=1
fi
echo "TPU_VERSION=${TPU_VERSION:-<unset>} | PREFILLER_TP_SIZE=$PREFILLER_TP_SIZE | DECODER_TP_SIZE=$DECODER_TP_SIZE"

PREFILL_HOSTS=()
PREFILL_PORTS=()
DECODE_HOSTS=()
DECODE_PORTS=()
PREFILL_PIDS=()
DECODE_PIDS=()

wait_for_server() {
  local port=$1
  local pid=$2
  timeout 1200 bash -c "
    until curl -s 127.0.0.1:${port}/health > /dev/null; do
      if ! kill -0 $pid 2>/dev/null; then
        echo \"Error: vLLM server on port $port (PID $pid) crashed or failed to start!\" >&2
        exit 1
      fi
      sleep 1
    done" && return 0 || return 1
}

check_failed_requests() {
  local log_file="$1"
  local failed_requests
  failed_requests=$(grep "Failed requests:" "$log_file" | awk '{print $3}' || true)

  if [ -z "$failed_requests" ]; then
    echo "Error: Could not find 'Failed requests:' in the benchmark output." >&2
    return 1
  fi

  if [ "$failed_requests" -gt 0 ]; then
    echo "Error: Benchmark reported $failed_requests failed requests." >&2
    return 1
  fi
  
  echo "Success: Benchmark reported $failed_requests failed requests." >&2
  return 0
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm" || true
  pkill -f "toy_proxy_server" || true
  sleep 5
  pkill -9 -f "vllm" || true
  pkill -9 -f "toy_proxy_server" || true
  fuser -k -9 /dev/vfio/* || true
  fuser -k -9 /dev/accel* || true
  rm -rf /tmp/jax_cache_* || true
  rm -f /tmp/libtpu_lockfile || true
}

LOG_DIR=$HOME/logs

echo "--- The HOME variable is : $HOME ---"

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
else
  # Delete old log files to avoid printing stale logs at the end
  rm -f $LOG_DIR/prefill_0.txt $LOG_DIR/decode_0.txt $LOG_DIR/benchmark_0.txt $LOG_DIR/proxy_0.txt
fi

cleanup_instances

for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    PORT=$((8400 + i))
    KV_PORT=$((7100 + i))
    SIDE_PORT=$((6100 + i))
    JAX_PORT=$((1200 + i))

    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_CHIPS=0 \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    SKIP_JAX_PRECOMPILE=1 \
    VLLM_XLA_CHECK_RECOMPILATION=0 \
    VLLM_XLA_CACHE_PATH="/tmp/jax_cache_$PORT" \
    JAX_COORDINATOR_ADDRESS="127.0.0.1:$JAX_PORT" \
    JAX_PROCESS_ID=0 \
    JAX_NUM_PROCESSES=1 \
    \
    vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.3 \
    --max-num-batched-tokens 1024 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}" \
    > $LOG_DIR/prefill_$i.txt 2>&1 &

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
    PREFILL_PIDS+=($!)
done


for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    PORT=$((9400 + i))
    KV_PORT=$((7200 + i))
    # Same as prefill SIDE_PORT
    SIDE_PORT=$((6100 + i))
    JAX_PORT=$((1210 + i))

    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_CHIPS=1 \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    SKIP_JAX_PRECOMPILE=1 \
    VLLM_XLA_CHECK_RECOMPILATION=0 \
    VLLM_XLA_CACHE_PATH="/tmp/jax_cache_$PORT" \
    JAX_COORDINATOR_ADDRESS="127.0.0.1:$JAX_PORT" \
    JAX_PROCESS_ID=0 \
    JAX_NUM_PROCESSES=1 \
    \
    vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.3 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 1024 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}" \
    > $LOG_DIR/decode_$i.txt 2>&1 &

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
    DECODE_PIDS+=($!)
done

for i in "${!PREFILL_PORTS[@]}"; do
    PORT=${PREFILL_PORTS[$i]}
    echo "Waiting for prefill on port $PORT to start..."
    wait_for_server $PORT ${PREFILL_PIDS[$i]}
done

for i in "${!DECODE_PORTS[@]}"; do
    PORT=${DECODE_PORTS[$i]}
    echo "Waiting for decode on port $PORT to start..."
    wait_for_server $PORT ${DECODE_PIDS[$i]}
done

echo "starting proxy server"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# 127.0.0.1, not localhost: localhost resolves to ::1 first and a pod has no
# IPv6 loopback, so uvicorn fails to bind and the proxy dies at startup.
python $SCRIPT_DIR/toy_proxy_server.py \
--host 127.0.0.1 \
--port 8000 \
--prefiller-hosts ${PREFILL_HOSTS[@]} \
--prefiller-ports ${PREFILL_PORTS[@]} \
--decoder-hosts ${DECODE_HOSTS[@]} \
--decoder-ports ${DECODE_PORTS[@]} \
> $LOG_DIR/proxy_0.txt 2>&1 &
PROXY_PID=$!

wait_for_server 8000 $PROXY_PID

LOG_FILE="$LOG_DIR/benchmark_0.txt"
echo "--- Running Disagg Benchmark ---" > $LOG_FILE

set -x
vllm bench serve \
  --model=$MODEL \
  --num-warmups=3 \
  --dataset-name=random \
  --random-input-len=${INPUT_LEN} \
  --random-output-len=${OUTPUT_LEN} \
  --num-prompts=${NUM_PROMPTS} \
  --ignore-eos \
  --host=127.0.0.1 \
  --port 8000 \
  --request-rate=${REQUEST_RATE} \
  >> $LOG_FILE 2>&1
set +x

check_failed_requests "$LOG_FILE"

cat <<'EOF'
The proxy server has been launched on: 127.0.0.1:8000

>> Send example request:

curl http://127.0.0.1:8000/v1/completions -X POST -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "what is your pet name",
    "max_tokens": 10,
    "temperature": 0.0
}'

>> Stop the proxy server and all prefill/decode instances:

pkill -f "vllm serve" && pkill -f "toy_proxy_server" && pkill -f "disagg_single_host"
EOF
