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
  if [[ "${CLEANUP_DONE:-}" == "true" ]]; then return; fi
  CLEANUP_DONE="true"
  set +e
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
      echo "==================== Ray Worker logs from worker node ${worker_ip} ===================="
      if [[ -f "/tmp/worker_${worker_ip}.log" ]]; then
        tail -n 50 "/tmp/worker_${worker_ip}.log" || true
        rm -f "/tmp/worker_${worker_ip}.log"
      fi
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true; sudo -n rm -rf /tmp/ray/* /tmp/vllm/* >/dev/null 2>&1 || true" || true
    done
  fi

  echo "   -> Cleaning Head Node..."
  rm -f /tmp/vllm_serve.log
  docker cp node:/root/vllm_serve.log /tmp/vllm_serve.log >/dev/null 2>&1 || true
  if [[ -f /tmp/vllm_serve.log ]]; then
    echo "==================== START OF VLLM SERVE LOG ===================="
    cat /tmp/vllm_serve.log || true
    echo "==================== END OF VLLM SERVE LOG ===================="
  fi
  rm -f "${TEMP_EXPORT_FILE:-}" >/dev/null 2>&1 || true
  docker stop node >/dev/null 2>&1 || true
  docker rm -f node >/dev/null 2>&1 || true
  sudo -n rm -rf /tmp/ray/* /tmp/vllm/* >/dev/null 2>&1 || true

  echo "✅ Cleanup complete."
  set -e
}
trap cleanup EXIT SIGINT SIGTERM

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
  local loop_count=0
  while [[ $SECONDS -lt $end_time ]]; do
    # 2. Check health
    # max-time 10 is crucial to prevent curl from hanging indefinitely if the server
    # accepts the TCP connection but is deadlocked or blocked from sending an HTTP response.
    if curl -fs --max-time 10 "localhost:${port}/health" > /dev/null; then
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

    # 4. Fast-fail if EngineCore crashed but left the API server zombie-ing
    if [[ $((loop_count % 3)) -eq 0 ]]; then
      if docker exec "$container_name" grep -q -E "EngineCore failed to start|ActorDiedError" "$log_path" 2>/dev/null; then
        echo "Error: Fatal vLLM engine crash detected in logs (e.g. EngineCore failed to start). Aborting."
        echo "Displaying logs from $container_name:$log_path"
        docker exec "$container_name" cat "$log_path" || true
        return 1
      fi
    fi

    sleep 5
    loop_count=$((loop_count + 1))
  done

  echo "Error: $service_name on $port failed to become healthy within ${timeout}s."
  echo "Displaying logs from $container_name:$log_path"
  stop_vllm_log_streaming
  docker exec "$container_name" cat "$log_path" || true
  return 1
}

PROJECT="$(gcloud config get-value project)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

# Prune Head Node BEFORE building the new image to ensure we have disk space
echo "--- Pruning Docker on Head Node to clear disk space..."
docker system prune -a --volumes -f || true

# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
# setup_environment traces a long tail of docker tag/build/push plumbing and
# one line per variable it sources from /etc/environment. It echoes its own
# progress, so the trace adds nothing; xtrace is on for this whole script, so
# turn it off across the call and back on after.
{ set +x; } 2>/dev/null
# Determine the Docker image name and registry path.
# Use the local image for multi-host benchmarks; otherwise, default to the remote GCR image.
if [[ "${IS_MULTI_HOST_BENCH:-false}" == "true" ]]; then
  IMAGE_NAME='vllm-tpu'
  setup_environment "$IMAGE_NAME"
  # Use the exported CI cache image path so Worker Nodes can pull it directly
  DOCKER_IMAGE="${EXPORTED_CI_CACHE_IMAGE:-$IMAGE_NAME:latest}"
else
  IMAGE_NAME="${GCR_REPO}/vllm-tpu"
  # Pass "true" to enable pushing to GCR
  setup_environment "${IMAGE_NAME}" "true"
  DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"
fi
set -x

# Resolve server / client commands and environment variables early
MODEL="${TEST_MODEL:-gs://tpu-commons-ci/qwen/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
VLLM_PORT="8000"
VLLM_SERVE_CMD=""
CLIENT_BENCH_CMD=""
DOCKER_ENV_ARGS=()
DOCKER_ENV_STR=""

if [ "$#" -ge 1 ]; then
    if [[ "$1" == *.json ]]; then
        # JSON configuration mode (multihost benchmark)
        CASE_FILE="$1"
        TARGET_CASE_NAME="$2"
        echo "--- Multi-host JSON case configuration detected: $CASE_FILE ($TARGET_CASE_NAME)"

        TEMP_EXPORT_FILE="/tmp/temp_env_exports.sh"
        rm -f "$TEMP_EXPORT_FILE"

        # Run parser_case.py inside the container to resolve server/client commands
        docker run --rm \
          --privileged \
          --net host \
          -v "$(pwd):/workspace" \
          -w /workspace \
          "${DOCKER_IMAGE}" \
          python3 .buildkite/benchmark/scripts/parser_case.py "$CASE_FILE" "$TARGET_CASE_NAME" > "$TEMP_EXPORT_FILE" || {
              echo "Error running parser_case.py inside container."
              if [[ -f "$TEMP_EXPORT_FILE" ]]; then
                  echo "===== Content of $TEMP_EXPORT_FILE ====="
                  cat "$TEMP_EXPORT_FILE"
                  echo "========================================"
              fi
              exit 1
          }

        # shellcheck source=/dev/null
        source "$TEMP_EXPORT_FILE"
        rm -f "$TEMP_EXPORT_FILE"

        # Write GCS upload metadata variables to a temp file for run_job.sh
        METADATA_FILE="/tmp/multihost_run_metadata.sh"
        rm -f "$METADATA_FILE"
        if [[ -n "${SERVER_CMD_ENVS:-}" ]]; then
          {
            echo "export MODEL_NAME='${MODEL_NAME:-}'"
            echo "export INPUT_LEN='${INPUT_LEN:-}'"
            echo "export OUTPUT_LEN='${OUTPUT_LEN:-}'"
            echo "declare -a SERVER_CMD_ENVS=()"
            for env_item in "${SERVER_CMD_ENVS[@]}"; do
                echo "SERVER_CMD_ENVS+=('$(printf '%q' "$env_item")')"
            done
          } >> "$METADATA_FILE"
        fi

        # Convert SERVER_CMD array to a single string VLLM_SERVE_CMD
        for env_item in "${SERVER_CMD_ENVS[@]}"; do
            VLLM_SERVE_CMD+="$(printf '%q ' "$env_item")"
            DOCKER_ENV_ARGS+=("-e" "$env_item")
        done
        for cmd_item in "${SERVER_CMD[@]}"; do
            VLLM_SERVE_CMD+="$(printf '%q ' "$cmd_item")"
        done

        # Set client benchmark command from CLIENT_CMD resolved by parser
        CLIENT_BENCH_CMD=""
        for env_item in "${CLIENT_CMD_ENVS[@]}"; do
            CLIENT_BENCH_CMD+="$(printf '%q ' "$env_item")"
        done
        for cmd_item in "${CLIENT_CMD[@]}"; do
            CLIENT_BENCH_CMD+="$(printf '%q ' "$cmd_item")"
        done

    else
        # Direct CLI configuration mode
        # If the first argument is not a .json file, the script directly assigns
        # $1 as the server command and the remaining arguments ($*) as the client command.
        if [ "$#" -gt 0 ]; then
            VLLM_SERVE_CMD="$1"
            echo "Using provided VLLM_SERVE_CMD: $VLLM_SERVE_CMD"
            # Shift so that remaining args are treated as the benchmark command
            shift
            if [ "$#" -gt 0 ]; then
                CLIENT_BENCH_CMD="${*:-}"
            fi
        else
            VLLM_SERVE_CMD=""
            CLIENT_BENCH_CMD=""
        fi
    fi
fi

# Extract port from VLLM_SERVE_CMD if present (e.g., --port 8080 or --port=8080)
if [[ "${VLLM_SERVE_CMD}" =~ --port[=\ ]+([0-9]+) ]]; then
    VLLM_PORT="${BASH_REMATCH[1]}"
    echo "--- Extracted VLLM_PORT=${VLLM_PORT} from JSON server command"
fi

# Clean up potential leftovers from previous runs
echo "--- Cleaning up previous cluster state..."
cleanup
CLEANUP_DONE="false"

# Safely parse EXTRA_DOCKER_ARGS into an array to prevent word-splitting on spaces
eval "declare -a EXTRA_DOCKER_ARGS_ARRAY=(${EXTRA_DOCKER_ARGS:-})"

# Serialize DOCKER_ENV_ARGS safely for SSH injection to Worker Nodes.
if [ ${#DOCKER_ENV_ARGS[@]} -gt 0 ]; then
    DOCKER_ENV_STR=$(printf '%q ' "${DOCKER_ENV_ARGS[@]}")
else
    DOCKER_ENV_STR=""
fi

# A tpu-v7x-16 slice has two hosts. Each host owns a 2x2x1 chip partition
# (8 TPU devices because v7x exposes two cores per chip), and the two
# processes are arranged across the final topology dimension.
readonly JAX_NUM_PROCESSES_VALUE=2
readonly TPU_PROCESS_BOUNDS_VALUE="1,1,2"
readonly TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE="2,2,1"

# libtpu uses CLOUD_TPU_TASK_ID to identify the local process in a TPU slice.
# Do not rely solely on automatic detection from inside Ray actors: if both
# actors default to task 0, jax.local_devices() reports process_index=0 on both
# hosts and topology_order_id collides. Read the authoritative per-VM worker
# number before starting the containers and pass it through explicitly.
HEAD_TPU_TASK_ID="$(get_metadata_value "instance/attributes/agent-worker-number")"
validate_tpu_task_id "$HEAD_INTERNAL_IP" "$HEAD_TPU_TASK_ID"

IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"
if (( ${#WORKER_IPS_ARRAY[@]} != 1 )); then
  echo "ERROR: tpu-v7x-16 requires exactly one remote host; found ${#WORKER_IPS_ARRAY[@]}." >&2
  exit 1
fi

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

if [[ "$HEAD_TPU_TASK_ID" != "0" ]]; then
  echo "ERROR: The tpu-v7x-16 head must have TPU process ID 0; metadata returned ${HEAD_TPU_TASK_ID}." >&2
  exit 1
fi
if [[ "${WORKER_TPU_TASK_IDS[0]}" != "1" ]]; then
  echo "ERROR: The second tpu-v7x-16 host must have TPU process ID 1; metadata returned ${WORKER_TPU_TASK_IDS[0]}." >&2
  exit 1
fi

echo "--- TPU process identity mapping"
echo "Head ${HEAD_INTERNAL_IP}: process_index=${HEAD_TPU_TASK_ID}"
for worker_index in "${!WORKER_IPS_ARRAY[@]}"; do
  echo "Worker ${WORKER_IPS_ARRAY[$worker_index]}: process_index=${WORKER_TPU_TASK_IDS[$worker_index]}"
done
echo "TPU_PROCESS_BOUNDS=${TPU_PROCESS_BOUNDS_VALUE}"
echo "TPU_CHIPS_PER_PROCESS_BOUNDS=${TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE}"

# 1. Start Ray Head Node locally
echo "--- Starting Ray Head Node Locally"
# This call carries ~a dozen -e flags; its xtrace line is long and adds
# nothing over the "Starting Ray Head Node Locally" banner above. The worker
# start below is already quiet, since xtrace does not echo ssh heredoc bodies.
{ set +x; } 2>/dev/null
bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" \
  "${HEAD_INTERNAL_IP}" \
  --head \
  "${HOST_HF_HOME}" \
  -e CLOUD_TPU_TASK_ID="${HEAD_TPU_TASK_ID}" \
  -e TPU_WORKER_ID="${HEAD_TPU_TASK_ID}" \
  -e JAX_PROCESS_ID="${HEAD_TPU_TASK_ID}" \
  -e JAX_NUM_PROCESSES="${JAX_NUM_PROCESSES_VALUE}" \
  -e TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS_VALUE}" \
  -e TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-vllm}" \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}" \
  -e RUNAI_STREAMER_LOG_LEVEL=DEBUG \
  -e RUNAI_STREAMER_LOG_TO_STDERR=1 \
  -e NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-0}" \
  -e MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-}" \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-}" \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE="${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}" \
  -e FORCE_MOE_RANDOM_ROUTING="${FORCE_MOE_RANDOM_ROUTING:-}" \
  -e IS_MULTI_HOST_BENCH="${IS_MULTI_HOST_BENCH:-}" \
  ${DOCKER_ENV_ARGS[@]:+"${DOCKER_ENV_ARGS[@]}"} \
  ${EXTRA_DOCKER_ARGS_ARRAY[@]:+"${EXTRA_DOCKER_ARGS_ARRAY[@]}"} &
set -x

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

    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
    # shellcheck disable=SC2002
    cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"

    # shellcheck disable=SC2087
    # shellcheck disable=SC2029
    # Redirect output to a temp file so it doesn't flood the Buildkite console with mixed logs.
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" > "/tmp/worker_${worker_ip}.log" 2>&1 << EOF &
IS_MULTI_HOST_BENCH="${IS_MULTI_HOST_BENCH:-false}" bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${HEAD_INTERNAL_IP}' --worker '${HOST_HF_HOME}' \
  -e HF_TOKEN='${HF_TOKEN:-}' \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e RUNAI_STREAMER_LOG_LEVEL=DEBUG \
  -e RUNAI_STREAMER_LOG_TO_STDERR=1 \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE='${MODEL_IMPL_TYPE:-vllm}' \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM='${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}' \
  -e NEW_MODEL_DESIGN='${NEW_MODEL_DESIGN:-0}' \
  -e MOE_REQUANTIZE_BLOCK_SIZE='${MOE_REQUANTIZE_BLOCK_SIZE:-}' \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE='${MOE_REQUANTIZE_WEIGHT_DTYPE:-}' \
  -e MOE_ALL_GATHER_ACTIVATION_DTYPE='${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}' \
  -e FORCE_MOE_RANDOM_ROUTING='${FORCE_MOE_RANDOM_ROUTING:-}' \
  -e IS_MULTI_HOST_BENCH='${IS_MULTI_HOST_BENCH:-}' \
  ${DOCKER_ENV_STR}
EOF
    WORKER_LAUNCHER_PIDS+=("$!")
    WORKER_LAUNCHER_HOSTS+=("$worker_ip")
    set -x
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

if [ -z "${VLLM_SERVE_CMD}" ]; then
    echo "Error: VLLM_SERVE_CMD cannot be empty! Please provide a JSON config or a command string."
    exit 1
fi

# Pre-download generation config from GCS into the container workspace and rewrite the server command
if [[ "${VLLM_SERVE_CMD}" =~ --generation-config[=\ ]+gs://([^ ]+) ]]; then
    gcs_path="gs://${BASH_REMATCH[1]}"
    config_url="${gcs_path%/*}"
    config_dir_name=$(basename "${config_url}")
    config_file_name=$(basename "${gcs_path}")

    echo "--- Detected GCS generation-config: $gcs_path"
    echo "--- Pre-downloading config to workspace inside container..."

    download_cmd="mkdir -p /workspace && gsutil -m cp -r ${config_url} /workspace/ && "

    VLLM_SERVE_CMD=$(echo "${VLLM_SERVE_CMD}" | sed -E "s|--generation-config[=\ ]+gs://[^ ]+|--generation-config /workspace/${config_dir_name}/${config_file_name}|g")
    VLLM_SERVE_CMD="${download_cmd}${VLLM_SERVE_CMD}"
    echo "Rewritten VLLM_SERVE_CMD: $VLLM_SERVE_CMD"
fi

# Launch vllm serve in the background inside the local 'node' container
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  node bash -c "${VLLM_SERVE_CMD} > /root/vllm_serve.log 2>&1"

# 4. Wait for the server to be healthy
SERVER_TIMEOUT=""
if [[ -n "${SERVER_CMD_ENVS:-}" ]]; then
    for env_item in "${SERVER_CMD_ENVS[@]}"; do
        if [[ "$env_item" == VLLM_ENGINE_READY_TIMEOUT_S=* ]]; then
            SERVER_TIMEOUT="${env_item#*=}"
        fi
    done
fi
wait_for_server "$VLLM_PORT" "node" "vllm serve" "/root/vllm_serve.log" "$SERVER_TIMEOUT"

# 5. Run Benchmarks / Validation
if [[ "${CASE_FILE:-}" == *.json ]]; then
  # JSON (.json): Delegates to run_bm.sh for advanced benchmark logic.
  echo "--- Invoking run_bm.sh for advanced benchmark logic on Head Node..."
  docker exec \
    -e HF_HOME=/root/.cache/huggingface \
    -e SERVER_ALREADY_RUNNING="true" \
    -e VLLM_PORT="${VLLM_PORT}" \
    -e GCS_BUCKET="${GCS_BUCKET:-}" \
    -e RECORD_ID="${RECORD_ID:-}" \
    node bash -c "cd /workspace/tpu_inference && chmod +x .buildkite/benchmark/scripts/run_bm.sh && .buildkite/benchmark/scripts/run_bm.sh $CASE_FILE $TARGET_CASE_NAME"
elif [ -n "${CLIENT_BENCH_CMD}" ]; then
  # Legacy string: Executes CLIENT_BENCH_CMD directly as a raw bash command.
  echo "--- Running Benchmark Command on Head Node"
  docker exec \
    -e HF_HOME=/root/.cache/huggingface \
    node bash -c "cd /workspace/tpu_inference && ${CLIENT_BENCH_CMD}"
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

