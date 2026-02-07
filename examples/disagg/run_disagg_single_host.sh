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

MODEL="Qwen/Qwen3-0.6B"

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
    until curl -s localhost:${port}/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  pkill -f "toy_proxy_server" || true
  sleep 1
}

mkdir -p $HOME/logs
cleanup_instances

# Start prefill instances
for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    PORT=$((8400 + i))
    KV_PORT=$((7100 + i))
    SIDE_PORT=$((6100 + i))

    # os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    # os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    # os.environ[TPU_VISIBLE_CHIPS] = "0,1,2,3"

    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_CHIPS=0 \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    SKIP_JAX_PRECOMPILE=1 \
    \
    vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.2 \
    --max-num-batched-tokens 8192 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}" \
    > $HOME/logs/prefill_$i.txt 2>&1 &

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
done


# Start decode instances
for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    PORT=$((9400 + i))
    KV_PORT=$((7200 + i))
    # Same as prefill SIDE_PORT
    SIDE_PORT=$((6100 + i))

    # os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    # os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    # os.environ[TPU_VISIBLE_CHIPS] = "4,5,6,7"

    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_CHIPS=1 \
    \
    TPU_KV_TRANSFER_PORT=$KV_PORT \
    TPU_SIDE_CHANNEL_PORT=$SIDE_PORT \
    SKIP_JAX_PRECOMPILE=1 \
    \
    vllm serve $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.6 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    --block-size 128 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}" \
    > $HOME/logs/decode_$i.txt 2>&1 &

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
done

# Wait for all instances to start
for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill on port $PORT to start..."
    wait_for_server $PORT
done

for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode on port $PORT to start..."
    wait_for_server $PORT
done

echo "starting proxy server"
# Start proxy server
python $HOME/tpu-inference/examples/disagg/toy_proxy_server.py \
--host localhost \
--port 8000 \
--prefiller-hosts ${PREFILL_HOSTS[@]} \
--prefiller-ports ${PREFILL_PORTS[@]} \
--decoder-hosts ${DECODE_HOSTS[@]} \
--decoder-ports ${DECODE_PORTS[@]} \
> $HOME/logs/proxy_s.txt 2>&1 &

# run benchmark for both disagg and non-disagg
LOG_FILE="$HOME/logs/benchmark_single_host.txt"
echo "--- Running Disagg Benchmark ---" > $LOG_FILE

# run ben for disagg
set -x
vllm bench serve \
--model=$MODEL \
--num-warmups=3 \
--dataset-name=random \
--random-input-len=4096 \
--random-output-len=128 \
--num-prompts=300 \
--ignore-eos \
--host=localhost \
--port 8000 \
--request-rate 4 \
>> $LOG_FILE 2>&1

echo -e "\n\n--- Running Non-Disagg Benchmark ---" >> $LOG_FILE
# run ben for non-disagg
vllm bench serve \
--model=$MODEL \
--num-warmups=3 \
--dataset-name=random \
--random-input-len=4096 \
--random-output-len=128 \
--num-prompts=100 \
--ignore-eos \
--host=localhost \
--port 9400 \
--request-rate 4 \
>> $LOG_FILE 2>&1
set +x

cat <<'EOF'
The proxy server has been launched on: 127.0.0.1:8000

>> Send example request:

curl http://localhost:8000/v1/completions -X POST -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "what is your pet name",
    "max_tokens": 10
}'

>> Stop the proxy server and all prefill/decode instances:

pkill -f "vllm serve" && pkill -f "toy_proxy_server" && pkill -f "run_disagg_single_host"
EOF
