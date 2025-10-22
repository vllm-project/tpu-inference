#!/bin/bash

# shellcheck disable=all

set -e

echo "--- DEBUG: The HOME variable is set to: $HOME ---"

CONTAINERS=$(docker ps -a --filter "name=node*" -q)
if [ -n "$CONTAINERS" ]; then
  docker stop $CONTAINERS
  docker rm -f $CONTAINERS
fi

# NOTE: Strange ray race condition between dock build and ray start in single machine
# if you find error, comments these line and rerun this .sh file
#  _wait_until_pg_ready(current_placement_group)
# tpu-inference/tpu_inference/executors/ray_distributed_executor.py

docker image prune -f
docker build -f docker/Dockerfile -t ullm:test .
DOCKER_IMAGE="ullm:test"

HOST_HF_HOME="/mnt/disks/data/hf-docker"
NUM_HOSTS_PER_INSTANCE=4
COMMON_SIDE_PORT=8900
MODEL="Qwen/Qwen3-0.6B"

######## Prefill hosts setup ########

# Start ray cluster on 4 hosts.

PREFILL_TPU_PORTS=(8476 8477 8478 8479)
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
        --name "node-${i}" \
        \
        -e TPU_MULTIHOST_BACKEND="ray" \
        -e TPU_NODE_ID="${i}" \
        -e TPU_KV_TRANSFER_PORT="${KV_PORT}" \
        -e TPU_SIDE_CHANNEL_PORT="${SIDE_PORT}" \
        -e RAY_DEDUP_LOGS="0" \
        -e SKIP_JAX_PRECOMPILE="1" \
        \
        -e TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1" \
        -e TPU_PROCESS_BOUNDS="2,2,1" \
        -e TPU_VISIBLE_CHIPS="${i}" \
        -e CLOUD_TPU_TASK_ID="${i}" \
        -e TPU_PROCESS_ADDRESSES="${PREFILL_TPU_ADDRS}" \
        -e TPU_PROCESS_PORT="${tpu_port}" \
        \
        -e HF_HOME="/root/hf" \
        -v "${HOST_HF_HOME}:/root/hf" \
        -v $HOME/test:/root/test \
        -v $HOME/logs:/root/logs \
        -v $HOME/vllm:/workspace/vllm \
        -v $HOME/tpu-inference:/workspace/tpu_inference \
        --entrypoint /bin/bash \
        "${DOCKER_IMAGE}" -c "${DOCKER_CMD}"
    sleep 2
    set +x
done

# Start vllm on host-0

PREFILL_VLLM_PORT="8400"

set -x
docker exec node-0 /bin/bash -c \
    "vllm serve $MODEL \
    --port ${PREFILL_VLLM_PORT} \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}' \
    > /root/logs/prefill.txt 2>&1 &"
set +x


######## Decode hosts setup ########

# Start ray cluster on 4 hosts.

DECODE_TPU_PORTS=(9476 9477 9478 9479)
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
        DOCKER_CMD="ray start --block --head --port=${DECODE_RAY_PORT}"
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
        --name "node-2${i}" \
        \
        -e TPU_MULTIHOST_BACKEND="ray" \
        -e TPU_NODE_ID="${i}" \
        -e TPU_KV_TRANSFER_PORT="${KV_PORT}" \
        -e TPU_SIDE_CHANNEL_PORT="${SIDE_PORT}" \
        -e RAY_DEDUP_LOGS="0" \
        -e SKIP_JAX_PRECOMPILE="1" \
        \
        -e TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1" \
        -e TPU_PROCESS_BOUNDS="2,2,1" \
        -e TPU_VISIBLE_CHIPS="${tpu_index}" \
        -e CLOUD_TPU_TASK_ID="${i}" \
        -e TPU_PROCESS_ADDRESSES="${DECODE_TPU_ADDRS}" \
        -e TPU_PROCESS_PORT="${tpu_port}" \
        \
        -e HF_HOME="/root/hf" \
        -v "${HOST_HF_HOME}:/root/hf" \
        -v $HOME/test:/root/test \
        -v $HOME/logs:/root/logs \
        -v $HOME/vllm:/workspace/vllm \
        -v $HOME/tpu-inference:/workspace/tpu_inference \
        --entrypoint /bin/bash \
        "${DOCKER_IMAGE}" -c "${DOCKER_CMD}"
    sleep 2
    set +x
done

# Start vllm on host-20

DECODE_VLLM_PORT="9400"

set -x
docker exec node-20 /bin/bash -c \
    "vllm serve $MODEL \
    --port ${DECODE_VLLM_PORT} \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}' \
    > /root/logs/decode.txt 2>&1 &"
set +x
