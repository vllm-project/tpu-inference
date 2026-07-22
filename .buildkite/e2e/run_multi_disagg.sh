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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# We are running ON the head node.
export SSH_USER="${SSH_USER:-$(whoami)}"

# We need a valid path for run_cluster.sh's HF_HOME bind mount
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"

# Benchmark related defaults
MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
INPUT_LEN=${INPUT_LEN:=128}
OUTPUT_LEN=${OUTPUT_LEN:=20}
NUM_PROMPTS=${NUM_PROMPTS:=100}
RANDOM_SEED=${RANDOM_SEED:=10}
MAX_CONCURRENCY=${MAX_CONCURRENCY:=10}
TEST_MODE=${TEST_MODE:=1} # 1: benchmark, 2: correctness, 3: both


# Log directory setup
LOG_DIR=$HOME/logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/prefill.txt "$LOG_DIR"/decode.txt "$LOG_DIR"/benchmark.txt "$LOG_DIR"/proxy.txt "$LOG_DIR"/correctness.txt

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

# Auto-generate SSH Key if it doesn't exist
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "--- Auto-generating SSH key for passwordless auth..."
    mkdir -p ~/.ssh
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i ~/.ssh/id_rsa)

# Assemble all IP addresses (Head + Workers)
IFS=',' read -r -a ALL_WORKERS_ARRAY <<< "${WORKER_IPS}"
ALL_IPS_ARRAY=("$HEAD_INTERNAL_IP")
for ip in "${ALL_WORKERS_ARRAY[@]}"; do
  if [[ -n "$ip" && "$ip" != "$HEAD_INTERNAL_IP" ]]; then
    ALL_IPS_ARRAY+=("$ip")
  fi
done
NUM_HOSTS=${#ALL_IPS_ARRAY[@]}

echo "Discovered TPU hosts in launch order: ${ALL_IPS_ARRAY[*]}"
echo "Current/local head IP: ${HEAD_INTERNAL_IP}"
echo "Total TPU hosts available: ${NUM_HOSTS}"

# Dynamic TP calculation based on accelerator type or TPU Version
if [[ "${ACCELERATOR_TYPE:-}" == *"4t"* ]] || [[ "${ACCELERATOR_TYPE:-}" == *"-4"* ]]; then
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-4}"
elif [[ "${ACCELERATOR_TYPE:-}" == *"8t"* ]] || [[ "${ACCELERATOR_TYPE:-}" == *"-8"* ]]; then
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-8}"
elif [[ "${TPU_VERSION:-tpu6e}" == "tpu7x" ]]; then
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-4}"
else
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-8}"
fi

if [[ "${TPU_VERSION:-tpu6e}" == "tpu7x" ]]; then
  CORES_PER_CHIP="${CORES_PER_CHIP:-2}"
else
  CORES_PER_CHIP="${CORES_PER_CHIP:-1}"
fi

TOTAL_CHIPS=$(( NUM_HOSTS * CHIPS_PER_HOST ))
echo "Calculated total TPU chips from hosts: ${TOTAL_CHIPS}"
echo "Using TPU cores per chip: ${CORES_PER_CHIP}"

if [[ "${TPU_VERSION:-}" == "tpu7x" && "${ACCELERATOR_TYPE:-}" == *"16"* && "$NUM_HOSTS" -lt 2 ]]; then
  echo "❌ TPU7x-16 should expose multiple host VMs, but discovered ${NUM_HOSTS}: ${ALL_IPS_ARRAY[*]}"
  exit 1
fi

# Specify # of hosts for each instance, or default to splitting hosts equally
PREFILL_HOSTS_COUNT="${PREFILL_HOSTS_COUNT:-}"
DECODE_HOSTS_COUNT="${DECODE_HOSTS_COUNT:-}"

if [[ -z "$PREFILL_HOSTS_COUNT" && -z "$DECODE_HOSTS_COUNT" ]]; then
  # Default to equal split if neither is explicitly provided.
  PREFILL_HOSTS_COUNT=$(( NUM_HOSTS / 2 ))
  DECODE_HOSTS_COUNT=$(( NUM_HOSTS - PREFILL_HOSTS_COUNT ))
  echo "⚠️ PREFILL_HOSTS_COUNT and DECODE_HOSTS_COUNT not specified. Defaulting to equal split: $PREFILL_HOSTS_COUNT hosts for Prefill, $DECODE_HOSTS_COUNT hosts for Decode."
elif [[ -z "$PREFILL_HOSTS_COUNT" ]]; then
  if [[ ! "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ DECODE_HOSTS_COUNT must be a positive integer. Got: $DECODE_HOSTS_COUNT"
    exit 1
  fi
  PREFILL_HOSTS_COUNT=$(( NUM_HOSTS - DECODE_HOSTS_COUNT ))
  echo "⚠️ PREFILL_HOSTS_COUNT not specified. Using remaining hosts for Prefill: $PREFILL_HOSTS_COUNT."
elif [[ -z "$DECODE_HOSTS_COUNT" ]]; then
  if [[ ! "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ PREFILL_HOSTS_COUNT must be a positive integer. Got: $PREFILL_HOSTS_COUNT"
    exit 1
  fi
  DECODE_HOSTS_COUNT=$(( NUM_HOSTS - PREFILL_HOSTS_COUNT ))
  echo "⚠️ DECODE_HOSTS_COUNT not specified. Using remaining hosts for Decode: $DECODE_HOSTS_COUNT."
fi

if [[ ! "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "❌ PREFILL_HOSTS_COUNT must be at least 1. Got: $PREFILL_HOSTS_COUNT"
  exit 1
fi

if [[ ! "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "❌ DECODE_HOSTS_COUNT must be at least 1. Got: $DECODE_HOSTS_COUNT"
  exit 1
fi

TOTAL_HOSTS_USED=$(( PREFILL_HOSTS_COUNT + DECODE_HOSTS_COUNT ))
if (( TOTAL_HOSTS_USED > NUM_HOSTS )); then
  echo "❌ Requested hosts for Prefill ($PREFILL_HOSTS_COUNT) + Decode ($DECODE_HOSTS_COUNT) = $TOTAL_HOSTS_USED exceeds total available hosts ($NUM_HOSTS)."
  exit 1
fi

echo "Partitioning cluster: $PREFILL_HOSTS_COUNT hosts for Prefill, $DECODE_HOSTS_COUNT hosts for Decode"

PREFILL_HOSTS=("${ALL_IPS_ARRAY[@]:0:PREFILL_HOSTS_COUNT}")
DECODE_HOSTS=("${ALL_IPS_ARRAY[@]:PREFILL_HOSTS_COUNT:DECODE_HOSTS_COUNT}")

PREFILL_HEAD_IP="${PREFILL_HOSTS[0]}"
DECODE_HEAD_IP="${DECODE_HOSTS[0]}"

PREFILL_WORKER_IPS=("${PREFILL_HOSTS[@]:1}")
DECODE_WORKER_IPS=("${DECODE_HOSTS[@]:1}")
echo "Prefill hosts: ${PREFILL_HOSTS[*]}"
echo "Decode hosts: ${DECODE_HOSTS[*]}"

PREFILL_TENSOR_PARALLEL_SIZE=$(( PREFILL_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP ))
DECODE_TENSOR_PARALLEL_SIZE=$(( DECODE_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP ))
echo "Calculated PREFILL_TENSOR_PARALLEL_SIZE: $PREFILL_TENSOR_PARALLEL_SIZE"
echo "Calculated DECODE_TENSOR_PARALLEL_SIZE: $DECODE_TENSOR_PARALLEL_SIZE"

TPU_VISIBLE_CHIPS_LOCAL="$(seq -s, 0 $(( CHIPS_PER_HOST - 1 )))"
SINGLE_HOST_TPU_ENV_ARGS=(
  -e TPU_PROCESS_BOUNDS=1,1,1
  -e TPU_CHIPS_PER_PROCESS_BOUNDS="1,${CHIPS_PER_HOST},1"
  -e TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS_LOCAL}"
  -e CLOUD_TPU_TASK_ID=0
  -e JAX_PROCESS_ID=0
  -e JAX_NUM_PROCESSES=1
)
PREFILL_TPU_ENV_ARGS=()
DECODE_TPU_ENV_ARGS=()
if (( PREFILL_HOSTS_COUNT == 1 )); then
  PREFILL_TPU_ENV_ARGS=("${SINGLE_HOST_TPU_ENV_ARGS[@]}")
  echo "Prefill is a single-host Ray cluster; forcing single-process TPU env with TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS_LOCAL}."
fi
if (( DECODE_HOSTS_COUNT == 1 )); then
  DECODE_TPU_ENV_ARGS=("${SINGLE_HOST_TPU_ENV_ARGS[@]}")
  echo "Decode is a single-host Ray cluster; forcing single-process TPU env with TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS_LOCAL}."
fi

PREFILL_LOG_TAIL_PID=""
DECODE_LOG_TAIL_PID=""

stop_vllm_log_streaming() {
  if [[ -n "${PREFILL_LOG_TAIL_PID:-}" ]]; then
    kill "$PREFILL_LOG_TAIL_PID" >/dev/null 2>&1 || true
    wait "$PREFILL_LOG_TAIL_PID" >/dev/null 2>&1 || true
    PREFILL_LOG_TAIL_PID=""
  fi

  if [[ -n "${DECODE_LOG_TAIL_PID:-}" ]]; then
    kill "$DECODE_LOG_TAIL_PID" >/dev/null 2>&1 || true
    wait "$DECODE_LOG_TAIL_PID" >/dev/null 2>&1 || true
    DECODE_LOG_TAIL_PID=""
  fi
}

start_vllm_log_streaming() {
  stop_vllm_log_streaming

  echo "--- Streaming vLLM Prefill and Decode logs while waiting for health..."

  docker exec node bash -c "touch /root/vllm_serve_prefill.log && tail -n +1 -F /root/vllm_serve_prefill.log" \
    > >(sed -u 's/^/[prefill] /') 2>&1 &
  PREFILL_LOG_TAIL_PID=$!

  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" \
    "docker exec node bash -c 'touch /root/vllm_serve_decode.log && tail -n +1 -F /root/vllm_serve_decode.log'" \
    > >(sed -u 's/^/[decode] /') 2>&1 &
  DECODE_LOG_TAIL_PID=$!
}

cleanup() {
  local exit_code=$?
  echo "🧹 Cleaning up containers on all hosts..."

  stop_vllm_log_streaming

  # Capture server logs before removing containers.
  echo "   -> Capturing server logs..."
  docker cp node:/root/vllm_serve_prefill.log "$LOG_DIR/prefill.txt" >/dev/null 2>&1 || true
  if [[ -n "${DECODE_HEAD_IP:-}" ]]; then
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "rm -f /tmp/vllm_serve_decode.log; docker cp node:/root/vllm_serve_decode.log /tmp/vllm_serve_decode.log >/dev/null 2>&1 || true" || true
    scp "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}:/tmp/vllm_serve_decode.log" "$LOG_DIR/decode.txt" >/dev/null 2>&1 || true
  fi

  # Cleanup Prefill workers
  for ip in "${PREFILL_WORKER_IPS[@]}"; do
    echo "   -> Cleaning Prefill worker: $ip"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
  done

  # Cleanup Decode Head
  if [[ -n "${DECODE_HEAD_IP:-}" ]]; then
    echo "   -> Cleaning Decode Head: $DECODE_HEAD_IP"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
  fi

  # Cleanup Decode workers
  for ip in "${DECODE_WORKER_IPS[@]}"; do
    echo "   -> Cleaning Decode worker: $ip"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
  done

  # Cleanup Prefill Head (Local Node)
  echo "   -> Cleaning Prefill Head (Local)..."
  docker stop node >/dev/null 2>&1 || true
  docker rm -f node >/dev/null 2>&1 || true

  # Cleanup Local proxy/benchmark container
  docker stop disagg-proxy-benchmark >/dev/null 2>&1 || true
  docker rm -f disagg-proxy-benchmark >/dev/null 2>&1 || true

  if [ $exit_code -ne 0 ]; then
    echo "--- 🚨 Script failed or timed out (Exit Code: $exit_code). Dumping logs..."
    for log_file in "prefill.txt" "decode.txt" "proxy.txt" "correctness.txt" "benchmark.txt"; do
      if [ -f "$LOG_DIR/$log_file" ] && [ -s "$LOG_DIR/$log_file" ]; then
        echo "+++ 📄 Log: $log_file"
        cat "$LOG_DIR/$log_file"
      fi
    done
  fi

  echo "✅ Cleanup complete."
}
trap cleanup EXIT

wait_for_server_remote() {
  local host=$1
  local port=$2
  local service_name=$3
  local timeout=${4:-7200}

  echo "Waiting for $service_name on ${host}:${port} to become healthy (Timeout: ${timeout}s)..."

  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    if curl -fs "http://${host}:${port}/health" > /dev/null; then
      echo "===== $service_name is healthy on ${host}:${port}. ==="
      return 0
    fi
    sleep 5
  done

  echo "Error: $service_name on ${host}:${port} failed to become healthy within ${timeout}s."
  return 1
}

vllm_server_process_alive() {
  local host=$1
  local port=$2
  local process_check="pgrep -af '[v]llm serve' | grep -q -- '--port ${port}'"

  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec node bash -c "$process_check" >/dev/null 2>&1
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker exec node bash -c \"$process_check\"" >/dev/null 2>&1
  fi
}

dump_vllm_server_log() {
  local host=$1
  local log_path=$2
  local service_name=$3

  echo "+++ 📄 ${service_name} log (${host}:${log_path})"
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec node cat "$log_path" 2>/dev/null || true
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker exec node cat '${log_path}' 2>/dev/null || true" || true
  fi
}

wait_for_vllm_prefill_and_decode() {
  local prefill_port=$1
  local decode_host=$2
  local decode_port=$3
  local timeout=${4:-7200}

  echo "Waiting for vLLM Prefill on localhost:${prefill_port} and vLLM Decode on ${decode_host}:${decode_port} to become healthy (Timeout: ${timeout}s)..."

  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    local prefill_healthy=0
    local decode_healthy=0

    if curl -fs "http://localhost:${prefill_port}/health" > /dev/null; then
      prefill_healthy=1
    fi

    if curl -fs "http://${decode_host}:${decode_port}/health" > /dev/null; then
      decode_healthy=1
    fi

    if (( prefill_healthy == 1 && decode_healthy == 1 )); then
      echo "===== vLLM Prefill and Decode are healthy. ==="
      return 0
    fi

    if ! vllm_server_process_alive "localhost" "$prefill_port"; then
      echo "Error: vLLM Prefill process exited before becoming healthy."
      dump_vllm_server_log "localhost" "/root/vllm_serve_prefill.log" "vLLM Prefill"
      return 1
    fi

    if ! vllm_server_process_alive "$decode_host" "$decode_port"; then
      echo "Error: vLLM Decode process exited before becoming healthy."
      dump_vllm_server_log "$decode_host" "/root/vllm_serve_decode.log" "vLLM Decode"
      return 1
    fi

    sleep 5
  done

  echo "Error: vLLM Prefill and Decode failed to become healthy within ${timeout}s."
  dump_vllm_server_log "localhost" "/root/vllm_serve_prefill.log" "vLLM Prefill"
  dump_vllm_server_log "$decode_host" "/root/vllm_serve_decode.log" "vLLM Decode"
  return 1
}

wait_for_ray_head() {
  local host=$1
  local timeout=300
  echo "Waiting for Ray head on ${host}:6379 to become available..."
  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    if nc -z -w2 "${host}" 6379 &>/dev/null; then
      echo "Ray head is reachable on ${host}:6379"
      return 0
    fi
    sleep 5
  done
  echo "Error: Ray head failed to start within ${timeout}s."
  return 1
}

dump_ray_resources() {
  local host=$1
  local label=$2
  local ray_dump_cmd="import json, ray; ray.init(address='auto', ignore_reinit_error=True); print(json.dumps(ray.nodes(), indent=2, sort_keys=True))"

  echo "--- Ray resources for ${label} cluster (${host})"
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec node python3 -c "$ray_dump_cmd" || true
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker exec node python3 -c \"${ray_dump_cmd}\"" || true
  fi
}


PROJECT="$(gcloud config get-value project)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

# Prune Head Node BEFORE building the new image to ensure we have disk space
echo "--- Pruning Docker on Head Node to clear disk space..."
docker system prune -a --volumes -f >/dev/null 2>&1 || true

# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../scripts/setup_docker_env.sh"
setup_environment "${IMAGE_NAME}" "true"

DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

# Clean up potential leftovers from previous runs
echo "--- Cleaning up previous cluster state..."
cleanup

# -----------------------------------------------------------------
# 1. Start Prefill Ray Cluster
# -----------------------------------------------------------------
echo "--- Starting Prefill Ray Head Node Locally on ${PREFILL_HEAD_IP}"
bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" \
  "${PREFILL_HEAD_IP}" \
  --head \
  "${HOST_HF_HOME}" \
  "${PREFILL_TPU_ENV_ARGS[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}" \
  -e NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-0}" \
  -e MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-}" \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-}" \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE="${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}" \
  -e FORCE_MOE_RANDOM_ROUTING="${FORCE_MOE_RANDOM_ROUTING:-}" &

sleep 30

wait_for_ray_head "${PREFILL_HEAD_IP}"

for worker_ip in "${PREFILL_WORKER_IPS[@]}"; do
    echo "--- Distributing and starting Prefill Ray Worker on ${worker_ip}"
    echo "   -> Pruning Docker on worker to free disk space..."
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f >/dev/null 2>&1" || true

    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
    # shellcheck disable=SC2002
    cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"

    # shellcheck disable=SC2087
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" << EOF &
bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${PREFILL_HEAD_IP}' --worker '${HOST_HF_HOME}' \
  ${PREFILL_TPU_ENV_ARGS[*]} \
  -e HF_TOKEN='${HF_TOKEN:-}' \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM='${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}' \
  -e NEW_MODEL_DESIGN='${NEW_MODEL_DESIGN:-0}' \
  -e MOE_REQUANTIZE_BLOCK_SIZE='${MOE_REQUANTIZE_BLOCK_SIZE:-}' \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE='${MOE_REQUANTIZE_WEIGHT_DTYPE:-}' \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE='${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}' \
  -e FORCE_MOE_RANDOM_ROUTING='${FORCE_MOE_RANDOM_ROUTING:-}'
EOF
    sleep 15
done

# -----------------------------------------------------------------
# 2. Start Decode Ray Cluster
# -----------------------------------------------------------------
echo "--- Starting Decode Ray Head Node on ${DECODE_HEAD_IP}"
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "docker system prune -a --volumes -f >/dev/null 2>&1" || true
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
# shellcheck disable=SC2002
cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" << EOF &
bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${DECODE_HEAD_IP}' --head '${HOST_HF_HOME}' \
  ${DECODE_TPU_ENV_ARGS[*]} \
  -e HF_TOKEN='${HF_TOKEN:-}' \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM='${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}' \
  -e NEW_MODEL_DESIGN='${NEW_MODEL_DESIGN:-0}' \
  -e MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-}" \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-}" \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE="${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}" \
  -e FORCE_MOE_RANDOM_ROUTING="${FORCE_MOE_RANDOM_ROUTING:-}"
EOF

sleep 30

wait_for_ray_head "${DECODE_HEAD_IP}"

for worker_ip in "${DECODE_WORKER_IPS[@]}"; do
    echo "--- Distributing and starting Decode Ray Worker on ${worker_ip}"
    echo "   -> Pruning Docker on worker to free disk space..."
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f >/dev/null 2>&1" || true

    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
    # shellcheck disable=SC2002
    cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"

    # shellcheck disable=SC2087
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" << EOF &
bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${DECODE_HEAD_IP}' --worker '${HOST_HF_HOME}' \
  ${DECODE_TPU_ENV_ARGS[*]} \
  -e HF_TOKEN='${HF_TOKEN:-}' \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM='${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}' \
  -e NEW_MODEL_DESIGN='${NEW_MODEL_DESIGN:-0}' \
  -e MOE_REQUANTIZE_BLOCK_SIZE='${MOE_REQUANTIZE_BLOCK_SIZE:-}' \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE='${MOE_REQUANTIZE_WEIGHT_DTYPE:-}' \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE='${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}' \
  -e FORCE_MOE_RANDOM_ROUTING='${FORCE_MOE_RANDOM_ROUTING:-}'
EOF
    sleep 15
done

echo "--- Waiting for Ray Clusters to fully form..."
sleep 120

dump_ray_resources "$PREFILL_HEAD_IP" "Prefill"
dump_ray_resources "$DECODE_HEAD_IP" "Decode"

# -----------------------------------------------------------------
# 3. Start vLLM Prefill & Decode Servers
# -----------------------------------------------------------------
echo "--- Starting vLLM Prefill server on Head Node locally"
PREFILL_VLLM_PORT="8400"
PREFILL_DOCKER_EXEC_ENV_ARGS="${PREFILL_TPU_ENV_ARGS[*]}"

cat << EOF > /tmp/start_prefill.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${PREFILL_DOCKER_EXEC_ENV_ARGS} \
  node bash -c "vllm serve ${MODEL} \
    --port ${PREFILL_VLLM_PORT} \
    --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_producer\"}' \
    --max-model-len 1024 > /root/vllm_serve_prefill.log 2>&1"
set +x
EOF

chmod +x /tmp/start_prefill.sh
bash /tmp/start_prefill.sh

echo "--- Starting vLLM Decode server on remote Head Node (${DECODE_HEAD_IP})"
DECODE_VLLM_PORT="9400"
DECODE_DOCKER_EXEC_ENV_ARGS="${DECODE_TPU_ENV_ARGS[*]}"

cat << EOF > /tmp/start_decode.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${DECODE_DOCKER_EXEC_ENV_ARGS} \
  node bash -c "vllm serve ${MODEL} \
    --port ${DECODE_VLLM_PORT} \
    --tensor-parallel-size ${DECODE_TENSOR_PARALLEL_SIZE} \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_consumer\"}' \
    --max-model-len 1024 > /root/vllm_serve_decode.log 2>&1"
set +x
EOF

chmod +x /tmp/start_decode.sh

echo "--- Copying start_decode.sh to Decode Head Node (${DECODE_HEAD_IP})..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "mkdir -p ~/tpu-inference/scripts" || true
cat /tmp/start_decode.sh | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "base64 -d > ~/tpu-inference/scripts/start_decode.sh"
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "chmod +x ~/tpu-inference/scripts/start_decode.sh"

echo "--- Executing start_decode.sh on Decode Head Node (${DECODE_HEAD_IP})..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "bash ~/tpu-inference/scripts/start_decode.sh"

# -----------------------------------------------------------------
# 4. Wait for healthiness
# -----------------------------------------------------------------
echo "--- vLLM Prefill and Decode start commands submitted. Checking both health endpoints and server processes..."
start_vllm_log_streaming
wait_for_vllm_prefill_and_decode "$PREFILL_VLLM_PORT" "$DECODE_HEAD_IP" "$DECODE_VLLM_PORT" 7200
stop_vllm_log_streaming

# -----------------------------------------------------------------
# 5. Start Proxy & Run Tests
# -----------------------------------------------------------------
echo "--- Starting Proxy and Benchmark Container locally..."
docker run -d \
    --privileged \
    --network host \
    --shm-size 16G \
    --name "disagg-proxy-benchmark" \
    -e HF_HOME="/root/hf" \
    -v "${HOST_HF_HOME}:/root/hf" \
    -v "$LOG_DIR:/root/logs" \
    --entrypoint /bin/bash \
    "${DOCKER_IMAGE}" -c "tail -f /dev/null"

echo "--- Starting Toy Proxy Server inside container..."
docker exec -d disagg-proxy-benchmark /bin/bash -c "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --prefiller-hosts localhost \
    --prefiller-ports ${PREFILL_VLLM_PORT} \
    --decoder-hosts ${DECODE_HEAD_IP} \
    --decoder-ports ${DECODE_VLLM_PORT} > /root/logs/proxy.txt 2>&1"

wait_for_server_remote "127.0.0.1" 8000 "Toy Proxy Server" 600

if [ "$TEST_MODE" = "1" ] || [ "$TEST_MODE" = "3" ]; then
    echo "--- Running Benchmark Test inside container..."
    timeout "${BENCHMARK_TIMEOUT_SECONDS:-1800}" \
    docker exec disagg-proxy-benchmark /bin/bash -c "vllm bench serve \
        --backend vllm \
        --host 127.0.0.1 \
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

    echo "--- Benchmark Results ---"
    docker exec disagg-proxy-benchmark cat /root/logs/benchmark.txt
fi

if [ "$TEST_MODE" = "2" ] || [ "$TEST_MODE" = "3" ]; then
    echo "--- Running Correctness Test inside container..."
    timeout "${CORRECTNESS_TIMEOUT_SECONDS:-1800}" \
    docker exec disagg-proxy-benchmark /bin/bash -c "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py \
        --baseline_url http://${DECODE_HEAD_IP}:${DECODE_VLLM_PORT}/v1/completions \
        --disagg_url http://127.0.0.1:8000/v1/completions \
        --model ${MODEL} \
        --num_requests ${NUM_PROMPTS} \
        --input_length ${INPUT_LEN} \
        --output_length ${OUTPUT_LEN} > /root/logs/correctness.txt 2>&1"

    echo "--- Correctness Results ---"
    docker exec disagg-proxy-benchmark cat /root/logs/correctness.txt
fi

echo "--- Tests completed successfully ---"