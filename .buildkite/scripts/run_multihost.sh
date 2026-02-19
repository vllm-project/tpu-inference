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
      
      # The first IP from gcloud is ALWAYS the head node
      if [[ -z "${HEAD_INTERNAL_IP:-}" ]]; then
        HEAD_INTERNAL_IP="${ALL_IPS_ARRAY[0]}"
        echo "   -> Discovered Head IP: $HEAD_INTERNAL_IP"
      fi
      
      # The rest are Worker IPs
      WORKER_IPS_LIST=("${ALL_IPS_ARRAY[@]:1}")
      
      # Join with commas
      WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
      echo "   -> Discovered Worker IPs: $WORKER_IPS"

      # Detect TPU Version for Docker Build
      if [[ -z "${TPU_VERSION:-}" ]]; then
        ACCELERATOR_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(acceleratorType)" 2>/dev/null || echo "")
        echo "   -> Detected Accelerator Type: $ACCELERATOR_TYPE"
        if [[ "$ACCELERATOR_TYPE" == *"tpu7"* ]]; then
          export IS_FOR_V7X=true
          echo "   -> Setting IS_FOR_V7X=true"
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

# Fallback to hostname -I if HEAD_INTERNAL_IP is still strictly unset.
HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(hostname -I | awk '{print $1}')}"

# Enforce TPUv7 requirement
if [[ "${IS_FOR_V7X:-false}" != "true" ]]; then
  echo "❌ This script is strictly for TPUv7 (IS_FOR_V7X=true). Exiting."
  exit 0
fi

# Auto-generate SSH Key if it doesn't exist (e.g. in Buildkite CI)
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "--- Auto-generating SSH key for passwordless auth..."
    mkdir -p ~/.ssh
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_rsa)

# Cleanup function that runs on exit to tear down the Ray cluster
cleanup() {
  echo "🧹 Cleaning up containers on head and workers..."
  IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS:-}"
  
  echo "   -> Cleaning workers..."
  if [[ ${#WORKER_IPS_ARRAY[@]} -gt 0 && -n "${WORKER_IPS_ARRAY[0]}" ]]; then
    for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
    done
  fi

  echo "   -> Cleaning Head Node..."
  docker stop node >/dev/null 2>&1 || true
  docker rm -f node >/dev/null 2>&1 || true
  rm -f /root/vllm_serve.log || true

  echo "✅ Cleanup complete."
}
# trap cleanup EXIT

wait_for_server() {
  local port=$1
  local container_name=$2
  local service_name=$3
  local log_path=$4
  local timeout=${5:-3600} # Default 1 hour

  echo "Waiting for $service_name on port $port to become healthy (Timeout: ${timeout}s)..."

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
      docker exec "$container_name" cat "$log_path" || true
      return 1
  fi

  echo "   -> Found PID: $pid"

  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    # 2. Check health
    if curl -fs "localhost:${port}/health" > /dev/null; then
      echo "===== $service_name is healthy on port: $port. ==="
      return 0
    fi

    # 3. Check if PID is alive INSIDE the container
    if ! docker exec "$container_name" kill -0 "$pid" 2>/dev/null; then
      echo "Error: $service_name on $port (PID $pid) died inside container."
      echo "Displaying logs from $container_name:$log_path"
      docker exec "$container_name" cat "$log_path" || true
      return 1
    fi

    sleep 5
  done

  echo "Error: $service_name on $port failed to become healthy within ${timeout}s."
  echo "Displaying logs from $container_name:$log_path"
  docker exec "$container_name" cat "$log_path" || true
  return 1
}

PROJECT="$(gcloud config get-value project)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
# Pass "true" to enable pushing to GCR
setup_environment "${IMAGE_NAME}" "true"

DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

# RunAI streamer setup 

# Clean up potential leftovers from previous runs
echo "--- Cleaning up previous cluster state..."
cleanup

# 1. Start Ray Head Node locally
echo "--- Starting Ray Head Node Locally"
bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" \
  "${HEAD_INTERNAL_IP}" \
  --head \
  "${HOST_HF_HOME}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 &

sleep 20

# 2. Distribute run_cluster.sh to workers and start them
IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    echo "--- Distributing and starting Ray Worker on ${worker_ip}"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
    # shellcheck disable=SC2002
    cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "cat > ~/tpu-inference/scripts/multihost/run_cluster.sh"
    
    REMOTE_CMD="bash ~/tpu-inference/scripts/multihost/run_cluster.sh \
      '${DOCKER_IMAGE}' \
      '${HEAD_INTERNAL_IP}' \
      --worker \
      '${HOST_HF_HOME}' \
      -e HF_TOKEN='${HF_TOKEN:-}' \
      -e TPU_MULTIHOST_BACKEND=ray \
      -e JAX_PLATFORMS='' \
      -e TPU_BACKEND_TYPE=jax \
      -e MODEL_IMPL_TYPE=vllm \
      -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1"

    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "$REMOTE_CMD" &
done


echo "--- Waiting for all worker nodes to connect"
# Wait a few seconds for all worker nodes to connect
sleep 60

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
  --no-async-scheduling \
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
