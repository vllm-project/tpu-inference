#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Multi-host P/D disaggregation plus speculative decoding.
# The default model is the larger MoE model used by repository multi-host CI.
# Speculative decoding is intentionally enabled only on the Decode server.

set -euo pipefail

export SSH_USER="${SSH_USER:-$(whoami)}"
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"

# Benchmark related defaults
MODEL="${MODEL:-Qwen/Qwen3-8B}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"
RANDOM_SEED="${RANDOM_SEED:-10}"
TEST_MODE="${TEST_MODE:-3}" # 1: benchmark, 2: correctness, 3: both
NODE_CONTAINER_NAME="node"
PROXY_CONTAINER_NAME="disagg-proxy-benchmark"
PREFILL_HOSTS_COUNT="${PREFILL_HOSTS_COUNT:-1}"
DECODE_HOSTS_COUNT="${DECODE_HOSTS_COUNT:-1}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-ngram}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"ngram\",\"prompt_lookup_max\":5,\"prompt_lookup_min\":3,\"num_speculative_tokens\":3}}"
DRAFT_MODEL_IMPL_TYPE="${DRAFT_MODEL_IMPL_TYPE:-auto}"

# Log directory setup
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/{prefill,decode,proxy,benchmark,correctness}.txt

get_metadata_value() {
    local path=$1
    curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/${path}" 2>/dev/null || true
}

get_current_internal_ip() {
    local metadata_ip
    metadata_ip="$(get_metadata_value "instance/network-interfaces/0/ip")"
    if [[ -n "$metadata_ip" ]]; then
        echo "$metadata_ip"
        return 0
    fi

    hostname -I | awk '{print $1}'
}

# Automatic Worker IP Discovery
if [[ -z "${WORKER_IPS:-}" ]]; then
    echo "⚠️  WORKER_IPS not provided. Attempting to discover via gcloud..."

    if command -v gcloud &> /dev/null; then
        ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
        TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"

        if [[ -n "$TPU_NAME" && -n "$ZONE" ]]; then
            echo "   -> Found TPU_NAME: $TPU_NAME, ZONE: $ZONE"
            ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)")
            ALL_IPS="${ALL_IPS//;/ }"
            ALL_IPS="${ALL_IPS//,/ }"

            # shellcheck disable=SC2206
            ALL_IPS_ARRAY=($ALL_IPS)

            if [[ -z "${HEAD_INTERNAL_IP:-}" ]]; then
                HEAD_INTERNAL_IP="$(get_current_internal_ip)"
                echo "   -> Current VM internal IP: $HEAD_INTERNAL_IP"
            fi

            CURRENT_IP_IN_SLICE=0
            WORKER_IPS_LIST=()
            for ip in "${ALL_IPS_ARRAY[@]}"; do
                if [[ "$ip" == "$HEAD_INTERNAL_IP" ]]; then
                    CURRENT_IP_IN_SLICE=1
                elif [[ -n "$ip" ]]; then
                    WORKER_IPS_LIST+=("$ip")
                fi
            done

            if (( CURRENT_IP_IN_SLICE != 1 )); then
                echo "❌ Current VM IP (${HEAD_INTERNAL_IP}) is not in discovered TPU endpoints: ${ALL_IPS_ARRAY[*]}"
                exit 1
            fi

            WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
            echo "   -> Discovered Worker IPs: $WORKER_IPS"

            ACCELERATOR_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(acceleratorType)" 2>/dev/null || echo "")
            echo "   -> Detected Accelerator Type: $ACCELERATOR_TYPE"
            if [[ -z "${TPU_VERSION:-}" ]]; then
                if [[ "$ACCELERATOR_TYPE" == *"tpu7"* ]]; then
                    export TPU_VERSION="tpu7x"
                    echo "   -> Setting TPU_VERSION=tpu7x"
                elif [[ "$ACCELERATOR_TYPE" == *"6e"* ]] || [[ "$ACCELERATOR_TYPE" == *"tpu6"* ]]; then
                    export TPU_VERSION="tpu6e"
                    echo "   -> Setting TPU_VERSION=tpu6e"
                fi
            fi
        else
            echo "❌ Could not determine TPU_NAME or ZONE from metadata. Please set WORKER_IPS manually."
            exit 1
        fi
    else
        echo "❌ gcloud not found. Please set WORKER_IPS environment variable manually."
        exit 1
    fi
fi

if [[ -z "${WORKER_IPS:-}" ]]; then
    echo "ERROR: Failed to discover WORKER_IPS. Please provide it manually."
    exit 1
fi

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_current_internal_ip)}"

# Always ensure ACCELERATOR_TYPE is populated if not specified in the environment
if [[ -z "${ACCELERATOR_TYPE:-}" ]] && command -v gcloud &> /dev/null && command -v curl &> /dev/null; then
    ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}' || echo "")}"
    TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"

    if [[ -n "$TPU_NAME" && -n "$ZONE" ]]; then
        ACCELERATOR_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(acceleratorType)" 2>/dev/null || echo "")
        echo "   -> Detected Accelerator Type: $ACCELERATOR_TYPE"
    fi
fi

# Auto-discover TPU_VERSION if not specified and ACCELERATOR_TYPE is present
if [[ -z "${TPU_VERSION:-}" && -n "${ACCELERATOR_TYPE:-}" ]]; then
    if [[ "$ACCELERATOR_TYPE" == *"tpu7"* ]]; then
        export TPU_VERSION="tpu7x"
        echo "   -> Setting TPU_VERSION=tpu7x"
    elif [[ "$ACCELERATOR_TYPE" == *"6e"* ]] || [[ "$ACCELERATOR_TYPE" == *"tpu6"* ]]; then
        export TPU_VERSION="tpu6e"
        echo "   -> Setting TPU_VERSION=tpu6e"
    fi
fi

if [[ -z "${TPU_VERSION:-}" ]]; then
    echo "❌ Error: TPU_VERSION environment variable is not set and could not be automatically discovered."
    exit 1
fi

echo "Running on TPU_VERSION: ${TPU_VERSION}"

IFS=',' read -r -a worker_array <<< "$WORKER_IPS"
ALL_IPS_ARRAY=("$HEAD_INTERNAL_IP")
for ip in "${worker_array[@]}"; do
  [[ -n "$ip" && "$ip" != "$HEAD_INTERNAL_IP" ]] && ALL_IPS_ARRAY+=("$ip")
done
NUM_HOSTS=${#ALL_IPS_ARRAY[@]}
echo "Discovered TPU hosts in launch order: ${ALL_IPS_ARRAY[*]}"
echo "Current/local head IP: ${HEAD_INTERNAL_IP}"
echo "Total TPU hosts available: ${NUM_HOSTS}"
[[ "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
(( PREFILL_HOSTS_COUNT + DECODE_HOSTS_COUNT <= NUM_HOSTS )) || exit 1

PREFILL_HOSTS=("${ALL_IPS_ARRAY[@]:0:PREFILL_HOSTS_COUNT}")
DECODE_HOSTS=("${ALL_IPS_ARRAY[@]:PREFILL_HOSTS_COUNT:DECODE_HOSTS_COUNT}")
PREFILL_HEAD_IP="${PREFILL_HOSTS[0]}"
DECODE_HEAD_IP="${DECODE_HOSTS[0]}"
PREFILL_WORKERS=("${PREFILL_HOSTS[@]:1}")
DECODE_WORKERS=("${DECODE_HOSTS[@]:1}")
echo "Prefill hosts: ${PREFILL_HOSTS[*]}"
echo "Decode hosts: ${DECODE_HOSTS[*]}"

if [[ "$ACCELERATOR_TYPE" == *4t* || "$ACCELERATOR_TYPE" == *-4* || "$TPU_VERSION" == tpu7x ]]; then
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-4}"
else
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-8}"
fi
if [[ "$TPU_VERSION" == tpu7x ]]; then CORES_PER_CHIP="${CORES_PER_CHIP:-2}"; else CORES_PER_CHIP="${CORES_PER_CHIP:-1}"; fi
PREFILL_TP=$((PREFILL_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP))
DECODE_TP=$((DECODE_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP))
echo "Calculated PREFILL_TENSOR_PARALLEL_SIZE: ${PREFILL_TP}"
echo "Calculated DECODE_TENSOR_PARALLEL_SIZE: ${DECODE_TP}"

if [[ ! -f "$HOME/.ssh/id_rsa" ]]; then
  mkdir -p "$HOME/.ssh"
  ssh-keygen -t rsa -b 4096 -N "" -f "$HOME/.ssh/id_rsa" -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i "$HOME/.ssh/id_rsa")

PROJECT="$(gcloud config get-value project 2>/dev/null)"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference/vllm-tpu-multi-disagg-speculative"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
echo "--- Building and pushing speculative decoding Docker image ---"
cleanup() {
  local code=$?
  echo "🧹 Cleaning up speculative disaggregation containers..."
  for ip in "${ALL_IPS_ARRAY[@]}"; do
    [[ "$ip" == "$HEAD_INTERNAL_IP" ]] || ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
      "docker rm -f '$NODE_CONTAINER_NAME' >/dev/null 2>&1 || true" || true
  done
  docker rm -f "$NODE_CONTAINER_NAME" "$PROXY_CONTAINER_NAME" >/dev/null 2>&1 || true
  if (( code != 0 )); then
    for log in prefill decode proxy benchmark correctness; do
      [[ -s "$LOG_DIR/$log.txt" ]] && { echo "+++ $log.txt"; cat "$LOG_DIR/$log.txt"; }
    done
  fi
  echo "✅ Cleanup complete."
}
trap cleanup EXIT

source "$SCRIPT_DIR/../scripts/setup_docker_env.sh"
setup_environment "$IMAGE_NAME" "true"
DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"
echo "Using Docker image: ${DOCKER_IMAGE}"

common_env="-e HF_TOKEN=${HF_TOKEN:-} -e TPU_MULTIHOST_BACKEND=ray -e JAX_PLATFORMS= -e TPU_BACKEND_TYPE=jax -e MODEL_IMPL_TYPE=vllm"
visible_chips="$(seq -s, 0 $((CHIPS_PER_HOST - 1)))"
single_host_env="-e TPU_PROCESS_BOUNDS=1,1,1 -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,${CHIPS_PER_HOST},1 -e TPU_VISIBLE_CHIPS=${visible_chips} -e CLOUD_TPU_TASK_ID=0 -e JAX_PROCESS_ID=0 -e JAX_NUM_PROCESSES=1"
PREFILL_SINGLE_ENV=""
DECODE_SINGLE_ENV=""
[[ "$PREFILL_HOSTS_COUNT" -eq 1 ]] && PREFILL_SINGLE_ENV="$single_host_env"
[[ "$DECODE_HOSTS_COUNT" -eq 1 ]] && DECODE_SINGLE_ENV="$single_host_env"
PREFILL_ENV="$PREFILL_SINGLE_ENV $common_env"
DECODE_ENV="$DECODE_SINGLE_ENV $common_env -e DRAFT_MODEL_IMPL_TYPE=${DRAFT_MODEL_IMPL_TYPE}"

launch_cluster() {
  local head=$1 role=$2 env_args=$3 workers=$4
  if [[ "$head" == "$HEAD_INTERNAL_IP" ]]; then
    bash "$TOP_DIR/scripts/multihost/run_cluster.sh" "$DOCKER_IMAGE" "$head" "$role" "$HOST_HF_HOME" $env_args &
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$head" "mkdir -p ~/tpu-inference/scripts/multihost"
    base64 < "$TOP_DIR/scripts/multihost/run_cluster.sh" | ssh "${SSH_OPTS[@]}" "$SSH_USER@$head" \
      "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$head" \
      "bash ~/tpu-inference/scripts/multihost/run_cluster.sh '$DOCKER_IMAGE' '$head' '$role' '$HOST_HF_HOME' $env_args" &
  fi
  sleep 20
  for worker in $workers; do
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$worker" "mkdir -p ~/tpu-inference/scripts/multihost"
    base64 < "$TOP_DIR/scripts/multihost/run_cluster.sh" | ssh "${SSH_OPTS[@]}" "$SSH_USER@$worker" \
      "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$worker" \
      "bash ~/tpu-inference/scripts/multihost/run_cluster.sh '$DOCKER_IMAGE' '$head' --worker '$HOST_HF_HOME' $env_args" &
    sleep 10
  done
}

wait_ray() {
  local host=$1
  echo "Waiting for Ray head on ${host}:6379..."
  for _ in {1..60}; do nc -z -w2 "$host" 6379 >/dev/null 2>&1 && return 0; sleep 5; done
  echo "Ray head did not start on $host" >&2
  return 1
}

echo "--- 1. Starting Prefill Ray Cluster ---"
launch_cluster "$PREFILL_HEAD_IP" --head "$PREFILL_ENV" "${PREFILL_WORKERS[*]:-}"
wait_ray "$PREFILL_HEAD_IP"

echo "--- 2. Starting Decode Ray Cluster ---"
launch_cluster "$DECODE_HEAD_IP" --head "$DECODE_ENV" "${DECODE_WORKERS[*]:-}"
wait_ray "$DECODE_HEAD_IP"
echo "--- Waiting for Ray clusters to fully form ---"
sleep 90

echo "--- 3. Starting vLLM Prefill and Decode Servers ---"
PREFILL_PORT=8400
DECODE_PORT=9400
echo "Starting vLLM Prefill server on ${PREFILL_HEAD_IP}:${PREFILL_PORT}..."
cat << EOF > /tmp/start_prefill_speculative.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${NODE_CONTAINER_NAME} bash -c "vllm serve ${MODEL} \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size ${PREFILL_TP} \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_producer\"}' \
    --max-model-len 1024 > /root/vllm_serve_prefill.log 2>&1"
set +x
EOF
chmod +x /tmp/start_prefill_speculative.sh
bash /tmp/start_prefill_speculative.sh

echo "Starting vLLM Decode server with speculative decoding on ${DECODE_HEAD_IP}:${DECODE_PORT}..."
cat << EOF > /tmp/start_decode_speculative.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${NODE_CONTAINER_NAME} bash -c "vllm serve ${MODEL} \
    --port ${DECODE_PORT} \
    --tensor-parallel-size ${DECODE_TP} \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --speculative-config '${SPECULATIVE_CONFIG}' \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_consumer\"}' \
    --max-model-len 1024 > /root/vllm_serve_decode.log 2>&1"
set +x
EOF
chmod +x /tmp/start_decode_speculative.sh
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" "mkdir -p ~/tpu-inference/scripts"
cat /tmp/start_decode_speculative.sh | base64 | ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "base64 -d > ~/tpu-inference/scripts/start_decode_speculative.sh"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "chmod +x ~/tpu-inference/scripts/start_decode_speculative.sh && bash ~/tpu-inference/scripts/start_decode_speculative.sh"

wait_server() {
  local host=$1 port=$2
  for _ in {1..720}; do curl -fs "http://$host:$port/health" >/dev/null 2>&1 && return 0; sleep 5; done
  echo "Server did not become healthy: $host:$port" >&2
  return 1
}

print_vllm_log() {
  local host=$1 log_path=$2
  echo "--- vLLM server log: $host:$log_path ---" >&2
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$NODE_CONTAINER_NAME" cat "$log_path" 2>&1 || true
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "docker exec '$NODE_CONTAINER_NAME' cat '$log_path'" 2>&1 || true
  fi
  echo "--- end vLLM server log ---" >&2
}

wait_vllm_server() {
  local health_host=$1 port=$2 node_host=$3 log_path=$4
  local vllm_running
  echo "Waiting for vLLM server on ${health_host}:${port} to become healthy..."
  for attempt in {1..720}; do
    curl -fs "http://$health_host:$port/health" >/dev/null 2>&1 && return 0

    # Allow up to 10 minutes for model loading and initialization. After that,
    # if the vLLM process has exited it cannot become healthy, so fail early
    # and print the server log instead of waiting for the full timeout.
    if (( attempt > 120 )); then
      if [[ "$node_host" == "$HEAD_INTERNAL_IP" ]]; then
        vllm_running="$(docker exec "$NODE_CONTAINER_NAME" pgrep -f '[v]llm serve' 2>/dev/null || true)"
      else
        vllm_running="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$node_host" \
          "docker exec '$NODE_CONTAINER_NAME' pgrep -f '[v]llm serve' 2>/dev/null || true" 2>/dev/null || true)"
      fi
      if [[ -z "$vllm_running" ]]; then
        echo "vLLM process exited before server became healthy: $health_host:$port" >&2
        print_vllm_log "$node_host" "$log_path"
        return 1
      fi
    fi
    sleep 5
  done

  echo "Server did not become healthy before timeout: $health_host:$port" >&2
  print_vllm_log "$node_host" "$log_path"
  return 1
}

echo "--- 4. Checking vLLM health and speculative decoding process ---"
wait_vllm_server 127.0.0.1 "$PREFILL_PORT" "$PREFILL_HEAD_IP" /root/vllm_serve_prefill.log
wait_vllm_server "$DECODE_HEAD_IP" "$DECODE_PORT" "$DECODE_HEAD_IP" /root/vllm_serve_decode.log
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "docker exec '$NODE_CONTAINER_NAME' pgrep -af '[v]llm serve' | grep -F -- '--speculative-config'"

echo "--- 5. Starting Proxy and Benchmark Container ---"
docker run -d --privileged --network host --shm-size 16G --name "$PROXY_CONTAINER_NAME" \
  -e HF_HOME=/root/hf -v "$HOST_HF_HOME:/root/hf" -v "$LOG_DIR:/root/logs" \
  --entrypoint /bin/bash "$DOCKER_IMAGE" -c 'tail -f /dev/null'
docker exec -d "$PROXY_CONTAINER_NAME" /bin/bash -c \
  "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port 8000 --prefiller-hosts 127.0.0.1 --prefiller-ports $PREFILL_PORT --decoder-hosts $DECODE_HEAD_IP --decoder-ports $DECODE_PORT > /root/logs/proxy.txt 2>&1"
echo "Waiting for Toy Proxy Server on 127.0.0.1:8000..."
wait_server 127.0.0.1 8000

if [[ "$TEST_MODE" == 2 || "$TEST_MODE" == 3 ]]; then
  echo "--- Running correctness test ---"
  docker exec "$PROXY_CONTAINER_NAME" /bin/bash -c \
    "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py --baseline_url http://$DECODE_HEAD_IP:$DECODE_PORT/v1/completions --disagg_url http://127.0.0.1:8000/v1/completions --model '$MODEL' --num_requests 20 --input_length 32 --output_length 64 > /root/logs/correctness.txt 2>&1"
  docker exec "$PROXY_CONTAINER_NAME" cat /root/logs/correctness.txt
fi

if [[ "$TEST_MODE" == 1 || "$TEST_MODE" == 3 ]]; then
  echo "--- Running benchmark test ---"
  docker exec "$PROXY_CONTAINER_NAME" /bin/bash -c \
    "vllm bench serve --backend vllm --host 127.0.0.1 --port 8000 --model '$MODEL' --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN --num-prompts $NUM_PROMPTS --request-rate inf --max-concurrency $MAX_CONCURRENCY --trust-remote-code --seed $RANDOM_SEED > /root/logs/benchmark.txt 2>&1"
  docker exec "$PROXY_CONTAINER_NAME" cat /root/logs/benchmark.txt
fi

echo "--- Tests completed successfully ---"
echo "Multi-host P/D speculative test completed: method=$SPECULATIVE_METHOD model=$MODEL"
