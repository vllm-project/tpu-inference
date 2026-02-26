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

# Parameters may come from external
# docker related
CONTAINER_PREFIX=${CONTAINER_PREFIX:="disagg-node"}
RUN_IN_BUILDKITE=${RUN_IN_BUILDKITE:=false}
TPU_VERSION=${TPU_VERSION:=tpu6e}
MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
DOCKER_IMAGE=${DOCKER_IMAGE:="vllm-tpu:000"}

# benchmark related
INPUT_LEN=${INPUT_LEN:=128}
OUTPUT_LEN=${OUTPUT_LEN:=20}
NUM_PROMPTS=${NUM_PROMPTS:=10}
RANDOM_SEED=${RANDOM_SEED:=10}
MAX_CONCURRENCY=${MAX_CONCURRENCY:=1}
TEST_MODE=${TEST_MODE:=1} # if 1, run benchmark; if 2, run correctness; if 3, run both
############################

echo "--- The HOME variable is : $HOME ---"

wait_for_server() {
  local port=$1
  local container_name=$2
  local service_name=$3
  local log_path=$4

  echo "port: $port, container_name: $container_name, service_name: $service_name, log_path: $log_path"


  # 1. Get the PID inside the container
  local pid=$(docker exec $container_name pgrep -n -f "$service_name")

  # Handle case where process didn't even start fast enough or failed immediately
  if [[ -z "$pid" ]]; then
      echo "Error: Could not find PID for $service_name immediately after start."
      docker exec "$container_name" cat "$log_path"
      return 1
  fi

  echo "Waiting for $service_name on port $port (Container PID: $pid) to become healthy..."

  local end_time=$((SECONDS + 600))
  while [[ $SECONDS -lt $end_time ]]; do
    # 2. Check health (Assuming port is mapped to localhost)
    if curl -fs "localhost:${port}/health" > /dev/null; then
      echo "=====$service_name is healthy on port: $port. ==="
      return 0
    fi

    # 3. FIX: Check if PID is alive INSIDE the container
    # We use 'docker exec' to run the kill command inside the container's namespace
    if ! docker exec "$container_name" kill -0 "$pid" 2>/dev/null; then
      echo "Error: $service_name on $port (PID $pid) died inside container while waiting for health check."
      echo "Displaying logs from $container_name:$log_path"
      docker exec "$container_name" cat "$log_path"
      return 1
    fi

    sleep 1
  done

  echo "Error: $service_name on $port failed to become healthy within the timeout."
  echo "Displaying logs from $container_name:$log_path"
  docker exec "$container_name" cat "$log_path"
  return 1
}

# clear existing container if there is
CONTAINERS=$(docker ps -a --filter "name=${CONTAINER_PREFIX}*" -q)
if [ -n "$CONTAINERS" ]; then
  docker stop $CONTAINERS
  docker rm -f $CONTAINERS
fi

# The docker image is generated outside if in buildkite
if [ "$RUN_IN_BUILDKITE" = "false" ]; then
    echo "Running in local mode, building image."
    docker image prune -f
    docker build -f docker/Dockerfile -t ${DOCKER_IMAGE} .
fi

# log folder $HOME/logs
LOG_DIR=$HOME/logs
if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
fi

# Define local mounts for non-Buildkite environments
# mount the image into local source code
local_mounts=()
if [ "$RUN_IN_BUILDKITE" = "false" ]; then
  echo "Running in local mode, mounting local vllm and tpu-inference directories."
  local_mounts=(
    -v "$HOME/vllm:/workspace/vllm"
    -v "$HOME/tpu-inference:/workspace/tpu_inference"
  )
fi

# General configs
HOST_HF_HOME="/mnt/disks/persist/models"
COMMON_SIDE_PORT=8900

# v6ex has 4 hosts with 4 TPUs, while v7x has 2 hosts with 2 TPU chips (4 cores). Adjust configs accordingly.
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
######## Prefill hosts setup ########

# Start ray cluster on 4 hosts.
PREFILL_TPU_ADDRS=()
for port in "${PREFILL_TPU_PORTS[@]}"; do
  PREFILL_TPU_ADDRS+=("127.0.0.1:$port")
done
PREFILL_TPU_ADDRS=$(IFS=, ; echo "${PREFILL_TPU_ADDRS[*]}")

PREFILL_RAY_PORT=8100

for ((i=0; i<NUM_HOSTS_PER_INSTANCE; i++)); do
    tpu_port=${PREFILL_TPU_PORTS[$i]}

    if [[ i -eq 0 ]]; then
        DOCKER_CMD="ray start --block --head --port=${PREFILL_RAY_PORT}"
    else
        DOCKER_CMD="ray start --block --address=127.0.0.1:${PREFILL_RAY_PORT}"
    fi

    KV_PORT=$((8200 + i))
    SIDE_PORT=$((COMMON_SIDE_PORT + i))

    set -x
    docker run -d \
        --privileged \
        --network host \
        --shm-size 16G \
        --name "${CONTAINER_PREFIX}-${i}" \
        \
        -e TPU_MULTIHOST_BACKEND="ray" \
        -e TPU_NODE_ID="${i}" \
        -e TPU_KV_TRANSFER_PORT="${KV_PORT}" \
        -e TPU_SIDE_CHANNEL_PORT="${SIDE_PORT}" \
        -e RAY_DEDUP_LOGS="0" \
        -e SKIP_JAX_PRECOMPILE="1" \
        \
        -e TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1" \
        -e TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS}" \
        -e TPU_VISIBLE_CHIPS="${i}" \
        -e CLOUD_TPU_TASK_ID="${i}" \
        -e TPU_PROCESS_ADDRESSES="${PREFILL_TPU_ADDRS}" \
        -e TPU_PROCESS_PORT="${tpu_port}" \
        \
        -e HF_HOME="/root/hf" \
        -e TPU_VERSION="${TPU_VERSION}" \
        -v "${HOST_HF_HOME}:/root/hf" \
        -v $LOG_DIR:/root/logs \
        "${local_mounts[@]}" \
        --entrypoint /bin/bash \
        "${DOCKER_IMAGE}" -c "${DOCKER_CMD}"
    sleep 1
    set +x
done

# Start vllm on host-0

PREFILL_VLLM_PORT="8400"

set -x
docker exec -d ${CONTAINER_PREFIX}-0 /bin/bash -c \
    "vllm serve $MODEL \
    --port ${PREFILL_VLLM_PORT} \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}' \
    --no-async-scheduling \
    > /root/logs/prefill.txt 2>&1"
set +x

######## Decode hosts setup ########

# Start ray cluster on 4 hosts.

DECODE_TPU_ADDRS=()
for port in "${DECODE_TPU_PORTS[@]}"; do
  DECODE_TPU_ADDRS+=("127.0.0.1:$port")
done
DECODE_TPU_ADDRS=$(IFS=, ; echo "${DECODE_TPU_ADDRS[*]}")

DECODE_RAY_PORT=9100

for ((i=0; i<NUM_HOSTS_PER_INSTANCE; i++)); do
    tpu_port=${DECODE_TPU_PORTS[$i]}
    tpu_index=$((i + NUM_HOSTS_PER_INSTANCE))

    if [[ i -eq 0 ]]; then
        DOCKER_CMD="ray start --block --head --port=${DECODE_RAY_PORT} --min-worker-port=20000 --max-worker-port=29999"
    else
        DOCKER_CMD="ray start --block --address=127.0.0.1:${DECODE_RAY_PORT}"
    fi

    KV_PORT=$((9200 + i))
    SIDE_PORT=$((COMMON_SIDE_PORT + i))

    set -x
    docker run -d \
        --privileged \
        --network host \
        --shm-size 16G \
        --name "${CONTAINER_PREFIX}-2-${i}" \
        \
        -e TPU_MULTIHOST_BACKEND="ray" \
        -e TPU_NODE_ID="${i}" \
        -e TPU_KV_TRANSFER_PORT="${KV_PORT}" \
        -e TPU_SIDE_CHANNEL_PORT="${SIDE_PORT}" \
        -e RAY_DEDUP_LOGS="0" \
        -e SKIP_JAX_PRECOMPILE="1" \
        \
        -e TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1" \
        -e TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS}" \
        -e TPU_VISIBLE_CHIPS="${tpu_index}" \
        -e CLOUD_TPU_TASK_ID="${i}" \
        -e TPU_PROCESS_ADDRESSES="${DECODE_TPU_ADDRS}" \
        -e TPU_PROCESS_PORT="${tpu_port}" \
        \
        -e HF_HOME="/root/hf" \
        -e TPU_VERSION="${TPU_VERSION}" \
        -v "${HOST_HF_HOME}:/root/hf" \
        -v $LOG_DIR:/root/logs \
        "${local_mounts[@]}" \
        --entrypoint /bin/bash \
        "${DOCKER_IMAGE}" -c "${DOCKER_CMD}"
    sleep 1
    set +x
done

# Start vllm on host-20

DECODE_VLLM_PORT="9400"

set -x
docker exec -d ${CONTAINER_PREFIX}-2-0 /bin/bash -c \
    "vllm serve $MODEL \
    --port ${DECODE_VLLM_PORT} \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}' \
    --no-async-scheduling \
    > /root/logs/decode.txt 2>&1"
set +x

wait_for_server "$PREFILL_VLLM_PORT" "${CONTAINER_PREFIX}-0" "vllm serve" "/root/logs/prefill.txt"

wait_for_server "$DECODE_VLLM_PORT" "${CONTAINER_PREFIX}-2-0" "vllm serve" "/root/logs/decode.txt"


# Start a long-running container for proxy and benchmark
echo "Starting proxy/benchmark container..."
set -x
docker run -d \
    --privileged \
    --network host \
    --shm-size 16G \
    --name "${CONTAINER_PREFIX}-proxy-benchmark" \
    -e HF_HOME="/root/hf" \
    -v "${HOST_HF_HOME}:/root/hf" \
    -v $LOG_DIR:/root/logs \
    "${local_mounts[@]}" \
    --entrypoint /bin/bash \
    "${DOCKER_IMAGE}" -c "tail -f /dev/null"
set +x

# Start proxy server in the container
echo "Starting proxy server in container..."
set -x
docker exec -d ${CONTAINER_PREFIX}-proxy-benchmark /bin/bash -c "python /workspace/tpu_inference/examples/disagg/toy_proxy_server.py --host localhost --port 8000 > /root/logs/proxy.txt 2>&1"
set +x

# Wait for proxy server to start
 wait_for_server 8000 "${CONTAINER_PREFIX}-proxy-benchmark" "toy_proxy_server" "/root/logs/proxy.txt"

# Run benchmark inside the proxy-benchmark-node container
if [ "$TEST_MODE" = "1" ] || [ "$TEST_MODE" = "3" ]; then
    echo "Running benchmark test in container."
    set -x
    docker exec ${CONTAINER_PREFIX}-proxy-benchmark /bin/bash -c "python3 /workspace/tpu_inference/scripts/vllm/benchmarking/benchmark_serving.py \
        --backend vllm \
        --host localhost \
        --port 8000 \
        --model ${MODEL} \
        --dataset-name random \
        --random-input-len ${INPUT_LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --num-prompts ${NUM_PROMPTS} \
        --request-rate inf \
        --max-concurrency ${MAX_CONCURRENCY} \
        --trust-remote-code \
        --seed ${RANDOM_SEED} > /root/logs/benchmark.txt 2>&1"
    set +x
fi

# Run correctness test inside the proxy-benchmark-node container
if [ "$TEST_MODE" = "2" ] || [ "$TEST_MODE" = "3" ]; then
    echo "Running correctness test in container."
    set -x
    docker exec ${CONTAINER_PREFIX}-proxy-benchmark /bin/bash -c "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py \
        --baseline_url http://localhost:9400/v1/completions \
        --disagg_url http://localhost:8000/v1/completions \
        --model ${MODEL} \
        --num_requests ${NUM_PROMPTS} \
        --input_length ${INPUT_LEN} \
        --output_length ${OUTPUT_LEN} > /root/logs/correctness.txt 2>&1"
    set +x
fi

# Clean up

cat <<'EOF'
The proxy server has been launched on: 127.0.0.1:8000
can send request like:

curl http://localhost:8000/v1/completions -X POST -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "what is your pet name",
    "max_tokens": 10
}'

>> Stop the proxy server and all prefill/decode instances:
docker stop $(docker ps -a --filter "name=disagg-node*" -q)
docker rm -f $(docker ps -a --filter "name=disagg-node*" -q)
EOF
