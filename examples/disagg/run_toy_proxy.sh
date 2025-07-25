#!/bin/bash

# shellcheck disable=all
set -xe

MODEL="/mnt/disks/data/Qwen/Qwen2.5-1.5B-Instruct"

NUM_PREFILL_INSTANCES=1
NUM_DECODE_INSTANCES=1
PREFILLER_TP_SIZE=1
DECODER_TP_SIZE=1

PREFILL_HOSTS=()
PREFILL_PORTS=()
DECODE_HOSTS=()
DECODE_PORTS=()

wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Start prefill instances
for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    PORT=$((8100 + i))

    vllm serve $MODEL \
    --port $PORT \
    --disable-log-requests \
    --gpu-memory-utilization 0.2 \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_role\":\"kv_producer\"}'

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
done


# Start decode instances
for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    PORT=$((8200 + i))

    vllm serve $model_name \
    --port $PORT \
    --disable-log-requests \
    --gpu-memory-utilization 0.2 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_role\":\"kv_consumer\"}'

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
done

# Wait for all instances to start
for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
done

for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
done


# Start proxy server
python $HOME/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
--host localhost \
--port 7080 \
--prefiller-hosts ${PREFILL_HOSTS[@]} \
--prefiller-ports ${PREFILL_PORTS[@]} \
--decoder-hosts ${DECODE_HOSTS[@]} \
--decoder-ports ${DECODE_PORTS[@]}
