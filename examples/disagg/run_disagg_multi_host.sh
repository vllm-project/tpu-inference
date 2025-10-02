#!/bin/bash

# shellcheck disable=all

set -e

docker stop $(docker ps -a --filter "name=node*" -q) && \
docker rm -f $(docker ps -a --filter "name=node*" -q)

docker image prune -f
docker build -f docker/Dockerfile -t ullm:test .
DOCKER_IMAGE="ullm:test"

HOST_HF_HOME="/mnt/disks/data/hf-docker"
NUM_HOSTS_PER_INSTANCE=4

MODEL="Qwen/Qwen3-0.6B"

######## Prefill hosts setup ########

# Start ray cluster on 4 hosts.

PREFILL_TPU_PORTS=(8476 8477 8478 8479)
PREFILL_TPU_ADDRS=()
for port in "${PREFILL_TPU_PORTS[@]}"; do
  PREFILL_TPU_ADDRS+=("127.0.0.1:$port")
done
PREFILL_TPU_ADDRS=$(IFS=, ; echo "${PREFILL_TPU_ADDRS[*]}")

PREFILL_RAY_PORT=9100

for ((i=0; i<NUM_HOSTS_PER_INSTANCE; i++)); do
    tpu_port=${PREFILL_TPU_PORTS[$i]}

    if [[ i -eq 0 ]]; then
        DOCKER_CMD="ray start --block --head --port=${PREFILL_RAY_PORT}"
    else
        DOCKER_CMD="ray start --block --address=127.0.0.1:${PREFILL_RAY_PORT}"
    fi

    KV_PORT=$((9200 + i))
    SIDE_PORT=$((9300 + i))

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
        --entrypoint /bin/bash \
        "${DOCKER_IMAGE}" -c "${DOCKER_CMD}"
    set +x
done


# Start vllm on host-0

PREFILL_VLLM_PORT="7400"

set -x
docker exec node-0 /bin/bash -c \
    "vllm serve $MODEL \
    --port ${PREFILL_VLLM_PORT} \
    --gpu-memory-utilization 0.2 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_commons.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}' \
    > /root/logs/prefill.txt 2>&1 &"
set +x


# TODO: add decode
