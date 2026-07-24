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
set -x

# We are running ON the head node.
export SSH_USER="${SSH_USER:-$(whoami)}"

# We need a valid path for run_cluster.sh's HF_HOME bind mount
HOST_HF_HOME="${HOST_HF_HOME:-/tmp/hf_home}"

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

# The Ray head must be the VM executing this script, not whichever endpoint
# happens to be listed first by gcloud.
HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_current_internal_ip)}"

# Automatic Worker IP Discovery
if [[ -z "${WORKER_IPS:-}" ]]; then
  echo "⚠️  WORKER_IPS not provided. Attempting to discover via gcloud..."

  # Check if gcloud is available and authorized
  if command -v gcloud &> /dev/null; then
    # Try to grab the zone from metadata if not set
    ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
    TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"

    if [[ -n "$TPU_NAME" && -n "$ZONE" ]]; then
      echo "   -> Found TPU_NAME: $TPU_NAME, ZONE: $ZONE"
      # Get all IPs in the slice
      ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)")

      # Normalize separators to spaces and convert to array
      ALL_IPS="${ALL_IPS//;/ }"
      ALL_IPS="${ALL_IPS//,/ }"

      # shellcheck disable=SC2206
      ALL_IPS_ARRAY=($ALL_IPS)

      # The endpoint order is not a reliable indication of which VM is running
      # this job. Find the local head in the slice and use every other endpoint
      # as a worker.
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
        echo "❌ Current VM IP (${HEAD_INTERNAL_IP}) is not in discovered TPU endpoints: ${ALL_IPS_ARRAY[*]}" >&2
        exit 1
      fi

      # Join with commas
      WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
      echo "   -> Current/local head IP: $HEAD_INTERNAL_IP"
      echo "   -> Discovered Worker IPs: $WORKER_IPS"

      # Detect TPU Version for Docker Build
      if [[ -z "${TPU_VERSION:-}" ]]; then
        ACCELERATOR_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(acceleratorType)" 2>/dev/null || echo "")
        echo "   -> Detected Accelerator Type: $ACCELERATOR_TYPE"
        if [[ "$ACCELERATOR_TYPE" == *"tpu7"* ]]; then
          export TPU_VERSION="tpu7x"
          echo "   -> Setting TPU_VERSION=tpu7x"
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

# Reject a manually supplied worker list that includes this head. Starting a
# second `node` container over SSH on the local VM would replace the Ray head.
IFS=',' read -r -a REQUESTED_WORKER_IPS <<< "${WORKER_IPS:-}"
WORKER_IPS_LIST=()
for worker_ip in "${REQUESTED_WORKER_IPS[@]}"; do
  [[ -z "$worker_ip" ]] && continue
  if [[ "$worker_ip" == "$HEAD_INTERNAL_IP" ]]; then
    echo "ERROR: WORKER_IPS must not include the local head IP (${HEAD_INTERNAL_IP})." >&2
    exit 1
  fi
  WORKER_IPS_LIST+=("$worker_ip")
done
WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")

if [[ -z "${WORKER_IPS:-}" ]]; then
  echo "ERROR: Failed to discover WORKER_IPS. Please provide it manually."
  exit 1
fi

# Enforce TPUv7 requirement
if [[ "${TPU_VERSION:-tpu6e}" != "tpu7x" ]]; then
  echo "❌ This script is strictly for TPUv7 (TPU_VERSION=tpu7x). Exiting."
  exit 0
fi

# Auto-generate SSH Key if it doesn't exist (e.g. in Buildkite CI)
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "--- Auto-generating SSH key for passwordless auth..."
    mkdir -p ~/.ssh
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i ~/.ssh/id_rsa)

get_remote_metadata_value() {
  local host=$1
  local path=$2
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
    "curl -fs -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/${path}' 2>/dev/null || true"
}

validate_tpu_task_id() {
  local host=$1
  local task_id=$2
  if [[ ! "$task_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TPU host ${host} returned an invalid agent-worker-number: '${task_id}'." >&2
    return 1
  fi
}

VLLM_LOG_TAIL_PID=""

stop_vllm_log_streaming() {
  if [[ -n "${VLLM_LOG_TAIL_PID:-}" ]]; then
    kill "$VLLM_LOG_TAIL_PID" >/dev/null 2>&1 || true
    wait "$VLLM_LOG_TAIL_PID" >/dev/null 2>&1 || true
    VLLM_LOG_TAIL_PID=""
  fi
}

start_vllm_log_streaming() {
  local container_name=$1
  local log_path=$2

  stop_vllm_log_streaming
  echo "--- Streaming ${container_name}:${log_path} while waiting for health..."
  docker exec "$container_name" bash -c \
    'touch "$1" && exec tail -n +1 -F "$1"' _ "$log_path" \
    > >(sed -u 's/^/[vllm] /') 2>&1 &
  VLLM_LOG_TAIL_PID=$!
}

dump_container_logs() {
  local host=$1
  local role=$2

  echo "+++ 📄 ${role} host logs (${host})"
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$HEAD_INTERNAL_IP" ]]; then
    echo "--- docker logs node (${role}) ---"
    docker logs node 2>&1 || true
    echo "--- /root/vllm_serve.log (${role}) ---"
    docker exec node cat /root/vllm_serve.log 2>&1 || true
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "echo '--- docker logs node (${role}) ---'; docker logs node 2>&1 || true; \
       echo '--- /root/vllm_serve.log (${role}) ---'; docker exec node cat /root/vllm_serve.log 2>&1 || true" || true
  fi
}

# Cleanup function that runs on exit to tear down the Ray cluster
cleanup() {
  local exit_code=${1:-0}
  echo "🧹 Cleaning up containers on head and workers..."
  IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS:-}"

  stop_vllm_log_streaming

  # Print diagnostics before removing containers. The generic multi-host runner
  # has one vLLM head and Ray workers, so dump the equivalent of the
  # prefill/decode host logs from every participating host.
  if (( exit_code != 0 )); then
    echo "--- 🚨 Script failed (exit code: ${exit_code}). Dumping host logs..."
    dump_container_logs "localhost" "head"
    for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
      [[ -n "$worker_ip" ]] && dump_container_logs "$worker_ip" "worker"
    done
  fi

  echo "   -> Cleaning workers..."
  if [[ ${#WORKER_IPS_ARRAY[@]} -gt 0 && -n "${WORKER_IPS_ARRAY[0]}" ]]; then
    for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
    done
  fi

  echo "   -> Cleaning Head Node..."
  docker cp node:/root/vllm_serve.log /tmp/vllm_serve.log >/dev/null 2>&1 || true
  if [[ -f /tmp/vllm_serve.log ]]; then
    echo "==================== START OF VLLM SERVE LOG ===================="
    cat /tmp/vllm_serve.log || true
    echo "==================== END OF VLLM SERVE LOG ===================="
  fi
  docker stop node >/dev/null 2>&1 || true
  docker rm -f node >/dev/null 2>&1 || true
  rm -f /root/vllm_serve.log || true

  echo "✅ Cleanup complete."
}
trap 'exit_code=$?; cleanup "$exit_code"; exit "$exit_code"' EXIT

wait_for_ray_head() {
  local timeout=${1:-300}
  echo "Waiting for Ray head on ${HEAD_INTERNAL_IP}:6379 to become available..."
  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    if nc -z -w2 "$HEAD_INTERNAL_IP" 6379 &>/dev/null; then
      echo "Ray head is reachable on ${HEAD_INTERNAL_IP}:6379"
      return 0
    fi
    sleep 5
  done
  echo "Error: Ray head failed to start within ${timeout}s." >&2
  return 1
}

check_worker_launchers() {
  local index pid host
  for index in "${!WORKER_LAUNCHER_PIDS[@]}"; do
    pid=${WORKER_LAUNCHER_PIDS[$index]}
    host=${WORKER_LAUNCHER_HOSTS[$index]}
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        echo "Error: worker launcher on ${host} exited before the Ray worker registered." >&2
      else
        echo "Error: worker launcher on ${host} failed before the Ray worker registered." >&2
      fi
      return 1
    fi
  done
}

dump_ray_nodes() {
  local ray_dump_cmd="import json, ray; ray.init(address='auto', ignore_reinit_error=True); print(json.dumps(ray.nodes(), indent=2, sort_keys=True))"
  echo "--- Ray node state (${HEAD_INTERNAL_IP})"
  docker exec node python3 -c "$ray_dump_cmd" || true
}

dump_tpu_process_env() {
  local host=$1
  local role=$2
  local dump_cmd
  dump_cmd='printf "CLOUD_TPU_TASK_ID=%s\nJAX_PROCESS_ID=%s\nTPU_PROCESS_BOUNDS=%s\n" "${CLOUD_TPU_TASK_ID-<unset>}" "${JAX_PROCESS_ID-<unset>}" "${TPU_PROCESS_BOUNDS-<unset>}"'

  echo "--- TPU process environment: ${role} (${host})"
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec node bash -c "$dump_cmd"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "docker exec node bash -c '$dump_cmd'"
  fi
}

wait_for_ray_cluster_members() {
  local expected_nodes=$1
  local timeout=${2:-900}
  local ray_ready_cmd
  ray_ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); alive=sum(node.get('Alive', False) for node in ray.nodes()); raise SystemExit(0 if alive >= ${expected_nodes} else 1)"

  echo "Waiting for Ray cluster to register ${expected_nodes} alive node(s)..."
  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    check_worker_launchers
    if docker exec node python3 -c "$ray_ready_cmd" >/dev/null 2>&1; then
      echo "Ray cluster has registered ${expected_nodes} alive node(s)."
      return 0
    fi
    sleep 5
  done

  echo "Error: Ray cluster did not register ${expected_nodes} alive node(s) within ${timeout}s." >&2
  dump_ray_nodes
  return 1
}

wait_for_server() {
  local port=$1
  local container_name=$2
  local service_name=$3
  local log_path=$4
  local timeout=${5:-7200} # Default 2 hours

  echo "Waiting for $service_name on port $port to become healthy (Timeout: ${timeout}s)..."
  start_vllm_log_streaming "$container_name" "$log_path"

  # 1. Get the PID inside the container
  # We might need to wait a few seconds for the process to actually start
  local pid=""
  for ((i=0; i<10; i++)); do
    pid=$(docker exec "$container_name" pgrep -n -f "$service_name" || true)
    if [[ -n "$pid" ]]; then
      break
    fi
    sleep 1
  done

  if [[ -z "$pid" ]]; then
      echo "Error: Could not find PID for $service_name immediately after start."
      stop_vllm_log_streaming
      docker exec "$container_name" cat "$log_path" || true
      return 1
  fi

  echo "   -> Found PID: $pid"

  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    # 2. Check health
    if curl -fs "localhost:${port}/health" > /dev/null; then
      stop_vllm_log_streaming
      echo "===== $service_name is healthy on port: $port. ==="
      return 0
    fi

    # 3. Check if PID is alive INSIDE the container
    if ! docker exec "$container_name" kill -0 "$pid" 2>/dev/null; then
      echo "Error: $service_name on $port (PID $pid) died inside container."
      echo "Displaying logs from $container_name:$log_path"
      stop_vllm_log_streaming
      docker exec "$container_name" cat "$log_path" || true
      return 1
    fi

    sleep 5
  done

  echo "Error: $service_name on $port failed to become healthy within ${timeout}s."
  echo "Displaying logs from $container_name:$log_path"
  stop_vllm_log_streaming
  docker exec "$container_name" cat "$log_path" || true
  return 1
}

PROJECT="$(gcloud config get-value project)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

# Prune Head Node BEFORE building the new image to ensure we have disk space
echo "--- Pruning Docker on Head Node to clear disk space..."
docker system prune -a --volumes -f || true

# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
# Pass "true" to enable pushing to GCR
setup_environment "${IMAGE_NAME}" "true"

DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

# Clean up potential leftovers from previous runs
echo "--- Cleaning up previous cluster state..."
cleanup 0

# libtpu uses CLOUD_TPU_TASK_ID to identify the local process in a TPU slice.
# Do not rely solely on automatic detection from inside Ray actors: if both
# actors default to task 0, jax.local_devices() reports process_index=0 on both
# hosts and topology_order_id collides. Read the authoritative per-VM worker
# number before starting the containers and pass it through explicitly.
HEAD_TPU_TASK_ID="$(get_metadata_value "instance/attributes/agent-worker-number")"
validate_tpu_task_id "$HEAD_INTERNAL_IP" "$HEAD_TPU_TASK_ID"

IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"
WORKER_TPU_TASK_IDS=()
SEEN_TPU_TASK_IDS=",${HEAD_TPU_TASK_ID},"
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  worker_task_id="$(get_remote_metadata_value "$worker_ip" "instance/attributes/agent-worker-number")"
  validate_tpu_task_id "$worker_ip" "$worker_task_id"
  if [[ "$SEEN_TPU_TASK_IDS" == *",${worker_task_id},"* ]]; then
    echo "ERROR: TPU task ID ${worker_task_id} is reported by more than one host." >&2
    exit 1
  fi
  SEEN_TPU_TASK_IDS+="${worker_task_id},"
  WORKER_TPU_TASK_IDS+=("$worker_task_id")
done

echo "--- TPU process identity mapping"
echo "Head ${HEAD_INTERNAL_IP}: CLOUD_TPU_TASK_ID=${HEAD_TPU_TASK_ID}"
for worker_index in "${!WORKER_IPS_ARRAY[@]}"; do
  echo "Worker ${WORKER_IPS_ARRAY[$worker_index]}: CLOUD_TPU_TASK_ID=${WORKER_TPU_TASK_IDS[$worker_index]}"
done

# 1. Start Ray Head Node locally
echo "--- Starting Ray Head Node Locally"
WORKER_LAUNCHER_PIDS=()
WORKER_LAUNCHER_HOSTS=()

bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" \
  "${HEAD_INTERNAL_IP}" \
  --head \
  "${HOST_HF_HOME}" \
  -e CLOUD_TPU_TASK_ID="${HEAD_TPU_TASK_ID}" \
  -e TPU_WORKER_ID="${HEAD_TPU_TASK_ID}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}" \
  -e RUNAI_STREAMER_LOG_LEVEL=DEBUG \
  -e RUNAI_STREAMER_LOG_TO_STDERR=1 \
  -e NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-0}" \
  -e MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-}" \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-}" \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE="${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}" \
  -e FORCE_MOE_RANDOM_ROUTING="${FORCE_MOE_RANDOM_ROUTING:-}" &

wait_for_ray_head
sleep 60

# 2. Distribute run_cluster.sh to workers and start them
for worker_index in "${!WORKER_IPS_ARRAY[@]}"; do
    worker_ip="${WORKER_IPS_ARRAY[$worker_index]}"
    worker_task_id="${WORKER_TPU_TASK_IDS[$worker_index]}"
    echo "--- Distributing and starting Ray Worker on ${worker_ip}"

    # Prune Worker Node BEFORE it tries to pull the new giant image
    echo "   -> Pruning Docker on worker to free disk space..."
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f" || true

    echo "   -> Disk usage on worker ${worker_ip} after Docker prune:"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "df -h"

    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
    # shellcheck disable=SC2002
    cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"

    # shellcheck disable=SC2087
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" << EOF &
bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${HEAD_INTERNAL_IP}' --worker '${HOST_HF_HOME}' \
  -e CLOUD_TPU_TASK_ID='${worker_task_id}' \
  -e TPU_WORKER_ID='${worker_task_id}' \
  -e HF_TOKEN='${HF_TOKEN:-}' \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e RUNAI_STREAMER_LOG_LEVEL=DEBUG \
  -e RUNAI_STREAMER_LOG_TO_STDERR=1 \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM='${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}' \
  -e NEW_MODEL_DESIGN='${NEW_MODEL_DESIGN:-0}' \
  -e MOE_REQUANTIZE_BLOCK_SIZE='${MOE_REQUANTIZE_BLOCK_SIZE:-}' \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE='${MOE_REQUANTIZE_WEIGHT_DTYPE:-}' \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE='${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}' \
  -e FORCE_MOE_RANDOM_ROUTING='${FORCE_MOE_RANDOM_ROUTING:-}'
EOF
    WORKER_LAUNCHER_PIDS+=("$!")
    WORKER_LAUNCHER_HOSTS+=("$worker_ip")
done


echo "--- Waiting for all worker nodes to connect"
wait_for_ray_cluster_members "$(( ${#WORKER_IPS_ARRAY[@]} + 1 ))" "${RAY_CLUSTER_TIMEOUT:-900}"

echo "--- TPU process environment on all Ray nodes"
dump_tpu_process_env "$HEAD_INTERNAL_IP" "head"
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  dump_tpu_process_env "$worker_ip" "worker"
done

# 3. Start vLLM server on the head node
echo "--- Starting vLLM server on head node"
MODEL="${TEST_MODEL:-gs://tpu-commons-ci/qwen/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
VLLM_PORT="8000"

VLLM_SERVE_CMD="vllm serve ${MODEL} \
  --port ${VLLM_PORT} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-16} \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --kv_cache_dtype=fp8 \
  --async-scheduling \
  --load-format=runai_streamer \
  --max-model-len 1024"

# Override VLLM_SERVE_CMD if provided as the first argument
if [ "$#" -ge 1 ]; then
    VLLM_SERVE_CMD="$1"
    echo "Using provided VLLM_SERVE_CMD: $VLLM_SERVE_CMD"
    # Shift so that remaining args are treated as the benchmark command
    shift
fi

# Launch vllm serve in the background inside the local 'node' container
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  node bash -c "${VLLM_SERVE_CMD} > /root/vllm_serve.log 2>&1"

# 4. Wait for the server to be healthy
wait_for_server "$VLLM_PORT" "node" "vllm serve" "/root/vllm_serve.log"

# 5. Run Benchmarks / Validation
if [ "$#" -gt 0 ]; then
  echo "--- Running provided Benchmark Command on Head Node"
  COMMAND_ARGS=("$@")

  docker exec \
    -e HF_HOME=/root/.cache/huggingface \
    node bash -c "${COMMAND_ARGS[*]}"
else
  # Default: Run the curl test to verify the endpoint
  echo "--- Running default curl test"
  docker exec node bash -c "
  curl http://localhost:8000/v1/completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 50}'
  "
fi

echo "--- Tests completed successfully"
