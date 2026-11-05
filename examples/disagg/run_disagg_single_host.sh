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

# Function to print logs on exit
print_logs_on_exit() {
  echo "--- Script exiting, displaying logs ---"

  # The logs are written inside containers to /root/logs, which is mapped from $LOG_DIR on the host.
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

    echo "--- Contents of $LOG_DIR/benchmark_0.txt ---"
    if [ -f "$LOG_DIR/benchmark_0.txt" ]; then
      cat "$LOG_DIR/benchmark_0.txt"
    else
      echo "File not found."
    fi

    if [ -f "$LOG_DIR/controller_0.txt" ]; then
      echo "--- Contents of $LOG_DIR/controller_0.txt ---"
      cat "$LOG_DIR/controller_0.txt"
    fi
  else
    echo "Log directory '$LOG_DIR' not found."
  fi
  echo "--- End of logs ---"
}

# Register the cleanup function to be called on script exit (normal or error)
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
    PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:=2}
    DECODER_TP_SIZE=${DECODER_TP_SIZE:=2}
else
    PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:=1}
    DECODER_TP_SIZE=${DECODER_TP_SIZE:=1}
fi

# Which physical chips each side gets, and the process grid over them. Not
# every subset is valid: libtpu requires a chip's index_on_host to match its
# position, so only single chips and axis-aligned blocks initialize. The grid
# is derived from how many chips the side was given, not from its TP degree --
# a chip may hold more than one core. Override *_CHIP_BOUNDS when the default
# guess does not match the host's grid.
PREFILL_CHIPS=${PREFILL_CHIPS:="0"}
DECODE_CHIPS=${DECODE_CHIPS:="1"}
chip_count() {
  local IFS=','
  # shellcheck disable=SC2086
  set -- $1
  echo $#
}
chip_bounds() {
  case "$1" in
    1) echo "1,1,1" ;;
    2) echo "1,2,1" ;;
    4) echo "2,2,1" ;;
    8) echo "2,4,1" ;;
    *) echo "1,$1,1" ;;
  esac
}
PREFILL_CHIP_COUNT=$(chip_count "$PREFILL_CHIPS")
DECODE_CHIP_COUNT=$(chip_count "$DECODE_CHIPS")
PREFILL_CHIP_BOUNDS=${PREFILL_CHIP_BOUNDS:=$(chip_bounds $PREFILL_CHIP_COUNT)}
DECODE_CHIP_BOUNDS=${DECODE_CHIP_BOUNDS:=$(chip_bounds $DECODE_CHIP_COUNT)}

# KV page size per side. They may differ only on the controller path below;
# the symmetric connector index-matches blocks and requires them equal.
# Decode is left unset by default so the platform picks its own page size.
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:=128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-}

# Connector selection. The default is the stock symmetric connector, so this
# script's behaviour is unchanged unless you ask for something else.
KV_CONNECTOR_MODULE=${KV_CONNECTOR_MODULE:="tpu_inference.distributed.tpu_connector"}

# Controller path: route the transfer through Raiden's byte-span reshard
# planner instead of the index-matched pull, which is what allows the two
# sides to differ in TP degree and page size. Requires the raiden connector.
KV_CONTROLLER=${KV_CONTROLLER:=0}
KV_CONTROLLER_PORT=${KV_CONTROLLER_PORT:=9700}
PREFILL_LISTENER_PORT=${PREFILL_LISTENER_PORT:=9800}
DECODE_LISTENER_PORT=${DECODE_LISTENER_PORT:=9900}
if [ "$KV_CONTROLLER" != "0" ]; then
  KV_CONNECTOR_MODULE=${KV_CONNECTOR_MODULE_OVERRIDE:="tpu_inference.distributed.tpu_raiden_connector"}
  KV_CONTROLLER_ADDRESS="localhost:$KV_CONTROLLER_PORT"
elif { [ -n "$DECODE_BLOCK_SIZE" ] && [ "$PREFILL_BLOCK_SIZE" != "$DECODE_BLOCK_SIZE" ]; } ||
     [ "$PREFILLER_TP_SIZE" != "$DECODER_TP_SIZE" ]; then
  echo "Error: prefill and decode geometries differ (TP $PREFILLER_TP_SIZE vs $DECODER_TP_SIZE," \
       "block $PREFILL_BLOCK_SIZE vs ${DECODE_BLOCK_SIZE:-<platform default>}) but KV_CONTROLLER=0." \
       "The symmetric connector index-matches blocks and would transfer the wrong bytes." >&2
  exit 1
fi

# The raiden connector needs the decode side's TP degree while the prefill
# engine is planning, so it rides along with the other connector options.
# The stock connector gets exactly the transfer config it always got.
KV_EXTRA_CONFIG=""
case "$KV_CONNECTOR_MODULE" in
  *tpu_raiden_connector)
    KV_EXTRA_CONFIG=",\"kv_connector_extra_config\":{\"decode_tp_size\":$DECODER_TP_SIZE}"
    ;;
esac

echo "TPU_VERSION=${TPU_VERSION:-<unset>} | PREFILLER_TP_SIZE=$PREFILLER_TP_SIZE | DECODER_TP_SIZE=$DECODER_TP_SIZE"
echo "chips: prefill=[$PREFILL_CHIPS] ($PREFILL_CHIP_BOUNDS) decode=[$DECODE_CHIPS] ($DECODE_CHIP_BOUNDS)"
echo "block_size: prefill=$PREFILL_BLOCK_SIZE decode=${DECODE_BLOCK_SIZE:-<platform default>} | connector=$KV_CONNECTOR_MODULE"
echo "controller=${KV_CONTROLLER_ADDRESS:-<disabled>}"

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
    until curl -s localhost:${port}/health > /dev/null; do
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
  pkill -f "raiden_controller_sidecar" || true
  sleep 5
  pkill -9 -f "vllm" || true
  pkill -9 -f "toy_proxy_server" || true
  pkill -9 -f "raiden_controller_sidecar" || true
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
  rm -f $LOG_DIR/prefill_0.txt $LOG_DIR/decode_0.txt $LOG_DIR/benchmark_0.txt $LOG_DIR/proxy_0.txt \
        $LOG_DIR/controller_0.txt
fi

cleanup_instances

# Start the Raiden controller sidecar. It owns the plan and moves no data, so
# one process serves the pair; it lives outside the engines because they own
# TPU chips and restart far more often than the control plane does.
if [ "$KV_CONTROLLER" != "0" ]; then
  JAX_PLATFORMS=cpu python -m tpu_inference.distributed.raiden_controller_sidecar \
    --port $KV_CONTROLLER_PORT > $LOG_DIR/controller_0.txt 2>&1 &
  CONTROLLER_PID=$!
  timeout 60 bash -c "
    until grep -q RAIDEN_CONTROLLER_PORT $LOG_DIR/controller_0.txt; do
      if ! kill -0 $CONTROLLER_PID 2>/dev/null; then
        echo 'Error: Raiden controller sidecar failed to start!' >&2
        exit 1
      fi
      sleep 1
    done"
  echo "Raiden controller sidecar up on $KV_CONTROLLER_ADDRESS (PID $CONTROLLER_PID)"
fi

# libtpu takes a whole-host advisory lock at /tmp/libtpu_lockfile as soon as a
# process claims more than one chip, so the second engine aborts with "The TPU
# is already in use by process with pid N" even though the two TPU_VISIBLE_CHIPS
# sets are disjoint. One chip per side does not trip it, which is why TP1 -> TP1
# needs none of this. Dropping the file between the two launches is the
# workaround libtpu's own error message points at: prefill keeps its open inode,
# decode creates a fresh one, and `fuser /dev/vfio/*` confirms the two processes
# end up holding disjoint iommu groups.
#
# Only do this when the chip sets really are disjoint -- that lock is the only
# thing stopping two engines from claiming the same chip.
PARTITION_NEEDS_LOCK_RELEASE=0
if [ "$PREFILL_CHIPS" != "$DECODE_CHIPS" ] &&
   { [ "$PREFILL_CHIP_COUNT" -gt 1 ] || [ "$DECODE_CHIP_COUNT" -gt 1 ]; }; then
  PARTITION_NEEDS_LOCK_RELEASE=1
  # Clear any lock left behind by a crashed run, so the wait below observes the
  # file prefill is about to create rather than a stale one.
  rm -f /tmp/libtpu_lockfile
fi

# Start prefill instances
for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    PORT=$((8400 + i))
    KV_PORT=$((7100 + i))
    SIDE_PORT=$((6100 + i))
    JAX_PORT=$((1200 + i))

    # os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    # os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    # os.environ[TPU_VISIBLE_CHIPS] = "0,1,2,3"

    TPU_CHIPS_PER_PROCESS_BOUNDS=$PREFILL_CHIP_BOUNDS \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_CHIPS=$PREFILL_CHIPS \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    TPU_KV_CONTROLLER_ADDRESS=${KV_CONTROLLER_ADDRESS:-} \
    TPU_KV_LISTENER_PORT=$((PREFILL_LISTENER_PORT + i * 16)) \
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
    --block-size $PREFILL_BLOCK_SIZE \
    --no-enable-prefix-caching \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"$KV_CONNECTOR_MODULE\",\"kv_role\":\"kv_producer\"$KV_EXTRA_CONFIG}" \
    > $LOG_DIR/prefill_$i.txt 2>&1 &

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
    PREFILL_PIDS+=($!)
done

if [ "$PARTITION_NEEDS_LOCK_RELEASE" = "1" ]; then
  # Wait for prefill to take the lock, then drop the file so decode can take
  # one of its own. Prefill keeps its open inode, so its own lock is unaffected.
  echo "waiting for prefill to claim the TPU, then releasing the whole-host lock"
  for _ in $(seq 1 300); do
    [ -e /tmp/libtpu_lockfile ] && break
    sleep 1
  done
  rm -f /tmp/libtpu_lockfile
fi

# Start decode instances
for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    PORT=$((9400 + i))
    KV_PORT=$((7200 + i))
    # Same as prefill SIDE_PORT
    SIDE_PORT=$((6100 + i))
    JAX_PORT=$((1210 + i))

    # os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    # os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    # os.environ[TPU_VISIBLE_CHIPS] = "4,5,6,7"

    TPU_CHIPS_PER_PROCESS_BOUNDS=$DECODE_CHIP_BOUNDS \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_CHIPS=$DECODE_CHIPS \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    TPU_KV_CONTROLLER_ADDRESS=${KV_CONTROLLER_ADDRESS:-} \
    TPU_KV_LISTENER_PORT=$((DECODE_LISTENER_PORT + i * 16)) \
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
    ${DECODE_BLOCK_SIZE:+--block-size $DECODE_BLOCK_SIZE} \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"$KV_CONNECTOR_MODULE\",\"kv_role\":\"kv_consumer\"}" \
    > $LOG_DIR/decode_$i.txt 2>&1 &

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
    DECODE_PIDS+=($!)
done

# Wait for all instances to start
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
# Start proxy server
python $SCRIPT_DIR/toy_proxy_server.py \
--host localhost \
--port 8000 \
--prefiller-hosts ${PREFILL_HOSTS[@]} \
--prefiller-ports ${PREFILL_PORTS[@]} \
--decoder-hosts ${DECODE_HOSTS[@]} \
--decoder-ports ${DECODE_PORTS[@]} \
> $LOG_DIR/proxy_0.txt 2>&1 &

# run benchmark for both disagg and non-disagg
LOG_FILE="$LOG_DIR/benchmark_0.txt"
echo "--- Running Disagg Benchmark ---" > $LOG_FILE

# run ben for disagg
set -x
vllm bench serve \
  --model=$MODEL \
  --num-warmups=3 \
  --dataset-name=random \
  --random-input-len=${INPUT_LEN} \
  --random-output-len=${OUTPUT_LEN} \
  --num-prompts=${NUM_PROMPTS} \
  --ignore-eos \
  --host=localhost \
  --port 8000 \
  --request-rate=${REQUEST_RATE} \
  >> $LOG_FILE 2>&1
set +x

check_failed_requests "$LOG_FILE"

cat <<'EOF'
The proxy server has been launched on: 127.0.0.1:8000

>> Send example request:

curl http://localhost:8000/v1/completions -X POST -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "what is your pet name",
    "max_tokens": 10,
    "temperature": 0.0
}'

>> Stop the proxy server and all prefill/decode instances:

pkill -f "vllm serve" && pkill -f "toy_proxy_server" && pkill -f "run_disagg_single_host"
EOF
