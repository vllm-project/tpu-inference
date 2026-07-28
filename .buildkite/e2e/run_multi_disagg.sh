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
    echo "WORKER_IPS not provided. Attempting to discover via gcloud..."

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
                echo "Current VM IP (${HEAD_INTERNAL_IP}) is not in discovered TPU endpoints: ${ALL_IPS_ARRAY[*]}"
                exit 1
            fi

            WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
            echo "   -> Discovered Worker IPs: $WORKER_IPS"
        else
            echo "Could not determine TPU_NAME or ZONE from metadata. Please set WORKER_IPS manually."
            exit 1
        fi
    else
        echo "gcloud not found. Please set WORKER_IPS environment variable manually."
        exit 1
    fi
fi

if [[ -z "${WORKER_IPS:-}" ]]; then
    echo "ERROR: Failed to discover WORKER_IPS. Please provide it manually."
    exit 1
fi

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_current_internal_ip)}"

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

# TPU7x always exposes four chips per host, with two TensorCores per chip.
readonly CHIPS_PER_HOST=4
readonly CORES_PER_CHIP=2

TOTAL_CHIPS=$(( NUM_HOSTS * CHIPS_PER_HOST ))
echo "Calculated total TPU chips from hosts: ${TOTAL_CHIPS}"
echo "Using TPU cores per chip: ${CORES_PER_CHIP}"

# Specify # of hosts for each instance, or default to splitting hosts equally
PREFILL_HOSTS_COUNT="${PREFILL_HOSTS_COUNT:-}"
DECODE_HOSTS_COUNT="${DECODE_HOSTS_COUNT:-}"

if [[ -z "$PREFILL_HOSTS_COUNT" && -z "$DECODE_HOSTS_COUNT" ]]; then
  # Default to equal split if neither is explicitly provided.
  PREFILL_HOSTS_COUNT=$(( NUM_HOSTS / 2 ))
  DECODE_HOSTS_COUNT=$(( NUM_HOSTS - PREFILL_HOSTS_COUNT ))
  echo "PREFILL_HOSTS_COUNT and DECODE_HOSTS_COUNT not specified. Defaulting to equal split: $PREFILL_HOSTS_COUNT hosts for Prefill, $DECODE_HOSTS_COUNT hosts for Decode."
elif [[ -z "$PREFILL_HOSTS_COUNT" ]]; then
  if [[ ! "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "DECODE_HOSTS_COUNT must be a positive integer. Got: $DECODE_HOSTS_COUNT"
    exit 1
  fi
  PREFILL_HOSTS_COUNT=$(( NUM_HOSTS - DECODE_HOSTS_COUNT ))
  echo "PREFILL_HOSTS_COUNT not specified. Using remaining hosts for Prefill: $PREFILL_HOSTS_COUNT."
elif [[ -z "$DECODE_HOSTS_COUNT" ]]; then
  if [[ ! "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREFILL_HOSTS_COUNT must be a positive integer. Got: $PREFILL_HOSTS_COUNT"
    exit 1
  fi
  DECODE_HOSTS_COUNT=$(( NUM_HOSTS - PREFILL_HOSTS_COUNT ))
  echo "DECODE_HOSTS_COUNT not specified. Using remaining hosts for Decode: $DECODE_HOSTS_COUNT."
fi

if [[ ! "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFILL_HOSTS_COUNT must be at least 1. Got: $PREFILL_HOSTS_COUNT"
  exit 1
fi

if [[ ! "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "DECODE_HOSTS_COUNT must be at least 1. Got: $DECODE_HOSTS_COUNT"
  exit 1
fi

TOTAL_HOSTS_USED=$(( PREFILL_HOSTS_COUNT + DECODE_HOSTS_COUNT ))
if (( TOTAL_HOSTS_USED > NUM_HOSTS )); then
  echo "Requested hosts for Prefill ($PREFILL_HOSTS_COUNT) + Decode ($DECODE_HOSTS_COUNT) = $TOTAL_HOSTS_USED exceeds total available hosts ($NUM_HOSTS)."
  exit 1
fi

echo "Partitioning cluster: $PREFILL_HOSTS_COUNT hosts for Prefill, $DECODE_HOSTS_COUNT hosts for Decode"

PREFILL_HOSTS=("${ALL_IPS_ARRAY[@]:0:PREFILL_HOSTS_COUNT}")
DECODE_HOSTS=("${ALL_IPS_ARRAY[@]:PREFILL_HOSTS_COUNT:DECODE_HOSTS_COUNT}")
SHARED_CLUSTER_HOSTS=("${PREFILL_HOSTS[@]}" "${DECODE_HOSTS[@]}")

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

# Prefill and decode share one Ray control plane but use disjoint actor sets.
# Each role forms its own role-local JAX mesh over the selected Ray nodes.
readonly TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE="2,2,1"
readonly TPU_PROCESS_PORT_VALUE=8476
readonly ROLE_HEAD_PROCESS_ID=0

# Populates PROCESS_IDENTITY_ENV_ARGS for a role-local JAX process. Keep the
# three equivalent identity variables synchronized at every launch site.
PROCESS_IDENTITY_ENV_ARGS=()
build_process_identity_env_args() {
  local process_id="$1"
  PROCESS_IDENTITY_ENV_ARGS=(
    -e CLOUD_TPU_TASK_ID="${process_id}"
    -e TPU_WORKER_ID="${process_id}"
    -e JAX_PROCESS_ID="${process_id}"
  )
}

SHARED_RAY_HEAD_IP="${PREFILL_HEAD_IP}"
PREFILL_NODE_IPS="$(IFS=,; echo "${PREFILL_HOSTS[*]}")"
DECODE_NODE_IPS="$(IFS=,; echo "${DECODE_HOSTS[*]}")"

PREFILL_PROCESS_MAP=""
PREFILL_PROCESS_ADDRESSES=""
for host_index in "${!PREFILL_HOSTS[@]}"; do
  [[ -z "$PREFILL_PROCESS_MAP" ]] || PREFILL_PROCESS_MAP+=","
  [[ -z "$PREFILL_PROCESS_ADDRESSES" ]] || PREFILL_PROCESS_ADDRESSES+=","
  PREFILL_PROCESS_MAP+="${PREFILL_HOSTS[$host_index]}=${host_index}"
  PREFILL_PROCESS_ADDRESSES+="${PREFILL_HOSTS[$host_index]}:${TPU_PROCESS_PORT_VALUE}"
done

DECODE_PROCESS_MAP=""
DECODE_PROCESS_ADDRESSES=""
for host_index in "${!DECODE_HOSTS[@]}"; do
  [[ -z "$DECODE_PROCESS_MAP" ]] || DECODE_PROCESS_MAP+=","
  [[ -z "$DECODE_PROCESS_ADDRESSES" ]] || DECODE_PROCESS_ADDRESSES+=","
  DECODE_PROCESS_MAP+="${DECODE_HOSTS[$host_index]}=${host_index}"
  DECODE_PROCESS_ADDRESSES+="${DECODE_HOSTS[$host_index]}:${TPU_PROCESS_PORT_VALUE}"
done

COMMON_TPU_ENV_ARGS=(
  -e TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE}"
  -e TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS_LOCAL}"
  -e TPU_PROCESS_PORT="${TPU_PROCESS_PORT_VALUE}"
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
)
PREFILL_TPU_ENV_ARGS=(
  "${COMMON_TPU_ENV_ARGS[@]}"
  -e TPU_PROCESS_BOUNDS="1,1,${PREFILL_HOSTS_COUNT}"
  -e TPU_PROCESS_ADDRESSES="${PREFILL_PROCESS_ADDRESSES}"
  -e JAX_NUM_PROCESSES="${PREFILL_HOSTS_COUNT}"
  -e VLLM_TPU_RAY_NODE_IPS="${PREFILL_NODE_IPS}"
  -e VLLM_TPU_RAY_PROCESS_MAP="${PREFILL_PROCESS_MAP}"
)
DECODE_TPU_ENV_ARGS=(
  "${COMMON_TPU_ENV_ARGS[@]}"
  -e TPU_PROCESS_BOUNDS="1,1,${DECODE_HOSTS_COUNT}"
  -e TPU_PROCESS_ADDRESSES="${DECODE_PROCESS_ADDRESSES}"
  -e JAX_NUM_PROCESSES="${DECODE_HOSTS_COUNT}"
  -e VLLM_TPU_RAY_NODE_IPS="${DECODE_NODE_IPS}"
  -e VLLM_TPU_RAY_PROCESS_MAP="${DECODE_PROCESS_MAP}"
)
build_process_identity_env_args "${ROLE_HEAD_PROCESS_ID}"
ROLE_HEAD_PROCESS_ENV_ARGS=("${PROCESS_IDENTITY_ENV_ARGS[@]}")

echo "Shared Ray head: ${SHARED_RAY_HEAD_IP}"
echo "Prefill actor hosts: ${PREFILL_NODE_IPS}; process map: ${PREFILL_PROCESS_MAP}"
echo "Decode actor hosts: ${DECODE_NODE_IPS}; process map: ${DECODE_PROCESS_MAP}"

run_host_script() {
  local host=$1
  shift
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    bash -s -- "$@"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" bash -s -- "$@"
  fi
}

print_logs() {
  local log_file
  for log_file in prefill.txt decode.txt proxy.txt correctness.txt benchmark.txt; do
    echo "--- ${log_file} ---"
    if [[ -f "${LOG_DIR}/${log_file}" ]]; then
      cat "${LOG_DIR}/${log_file}"
    else
      echo "(not found)"
    fi
  done
}

cleanup_host() {
  local host=$1
  echo "  Cleaning up containers on ${host}..."

  run_host_script "$host" <<'CLEANUP_HOST'
status=0
for container in node disagg-proxy-benchmark; do
  if docker inspect "$container" >/dev/null 2>&1; then
    if ! docker rm -f "$container" >/dev/null 2>&1; then
      echo "Warning: failed to remove container $container" >&2
      status=1
    fi
  fi
done
exit $status
CLEANUP_HOST
}

collect_remote_logs() {
  echo "  Collecting server logs before cleanup..."
  # Prefill server log (on local head node)
  docker exec node cat /root/vllm_serve_prefill.log \
    > "${LOG_DIR}/prefill.txt" 2>/dev/null || true
  # Decode server log (on remote decode head)
  if [[ -n "${DECODE_HEAD_IP:-}" && "${DECODE_HEAD_IP}" != "${HEAD_INTERNAL_IP}" ]]; then
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" \
      "docker exec node cat /root/vllm_serve_decode.log 2>/dev/null" \
      > "${LOG_DIR}/decode.txt" 2>/dev/null || true
  elif [[ -n "${DECODE_HEAD_IP:-}" ]]; then
    docker exec node cat /root/vllm_serve_decode.log \
      > "${LOG_DIR}/decode.txt" 2>/dev/null || true
  fi
}

cleanup() {
  echo "--- Cleaning up multi-host disaggregated serving"

  collect_remote_logs

  for ip in "${SHARED_CLUSTER_HOSTS[@]}"; do
    cleanup_host "$ip" ||
      echo "Warning: failed to remove test containers on ${ip}." >&2
  done

  print_logs
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  cleanup || true
  exit "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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

wait_for_ray_cluster_members() {
  local expected_nodes=$1
  local timeout=${2:-900}
  local ready_cmd
  ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); alive=sum(node.get('Alive', False) for node in ray.nodes()); raise SystemExit(0 if alive == ${expected_nodes} else 1)"

  echo "Waiting for shared Ray cluster to register ${expected_nodes} nodes..."
  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    if docker exec node python3 -c "$ready_cmd" >/dev/null 2>&1; then
      echo "Shared Ray cluster has registered ${expected_nodes} nodes."
      return 0
    fi
    sleep 5
  done

  echo "Error: shared Ray cluster did not register ${expected_nodes} nodes within ${timeout}s." >&2
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

dump_tpu_process_env() {
  local host=$1
  local role=$2
  local dump_cmd
  dump_cmd='printf "CLOUD_TPU_TASK_ID=%s\nTPU_WORKER_ID=%s\nJAX_PROCESS_ID=%s\nJAX_NUM_PROCESSES=%s\nTPU_PROCESS_BOUNDS=%s\nTPU_CHIPS_PER_PROCESS_BOUNDS=%s\nTPU_PROCESS_ADDRESSES=%s\nTPU_VISIBLE_CHIPS=%s\n" "${CLOUD_TPU_TASK_ID-<unset>}" "${TPU_WORKER_ID-<unset>}" "${JAX_PROCESS_ID-<unset>}" "${JAX_NUM_PROCESSES-<unset>}" "${TPU_PROCESS_BOUNDS-<unset>}" "${TPU_CHIPS_PER_PROCESS_BOUNDS-<unset>}" "${TPU_PROCESS_ADDRESSES-<unset>}" "${TPU_VISIBLE_CHIPS-<unset>}"'

  echo "--- TPU process environment: ${role} (${host})"
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec node bash -c "$dump_cmd"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "docker exec node bash -c '$dump_cmd'"
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
cleanup 0

# -----------------------------------------------------------------
# 1. Start one shared Ray cluster for Prefill and Decode actors
# -----------------------------------------------------------------
echo "--- Starting shared Ray Head Node locally on ${SHARED_RAY_HEAD_IP}"
RUN_CLUSTER_CLEANUP_OWNER=parent \
bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" \
  "${SHARED_RAY_HEAD_IP}" \
  --head \
  "${HOST_HF_HOME}" \
  "${PREFILL_TPU_ENV_ARGS[@]}" \
  "${ROLE_HEAD_PROCESS_ENV_ARGS[@]}" \
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

wait_for_ray_head "${SHARED_RAY_HEAD_IP}"

for worker_ip in "${SHARED_CLUSTER_HOSTS[@]:1}"; do
    worker_role=""
    role_process_id=""
    worker_tpu_env_args=()

    for host_index in "${!PREFILL_HOSTS[@]}"; do
      if [[ "${PREFILL_HOSTS[$host_index]}" == "$worker_ip" ]]; then
        worker_role="prefill"
        role_process_id="$host_index"
        worker_tpu_env_args=("${PREFILL_TPU_ENV_ARGS[@]}")
        break
      fi
    done
    if [[ -z "$worker_role" ]]; then
      for host_index in "${!DECODE_HOSTS[@]}"; do
        if [[ "${DECODE_HOSTS[$host_index]}" == "$worker_ip" ]]; then
          worker_role="decode"
          role_process_id="$host_index"
          worker_tpu_env_args=("${DECODE_TPU_ENV_ARGS[@]}")
          break
        fi
      done
    fi
    if [[ -z "$worker_role" || -z "$role_process_id" ]]; then
      echo "Error: unable to assign Ray worker ${worker_ip} to prefill or decode." >&2
      exit 1
    fi

    build_process_identity_env_args "${role_process_id}"
    role_process_env_args=("${PROCESS_IDENTITY_ENV_ARGS[@]}")

    echo "--- Starting shared Ray Worker on ${worker_ip} for ${worker_role} actor process ${role_process_id}"
    echo "   -> Pruning Docker on worker to free disk space..."
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f >/dev/null 2>&1" || true

    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost" || true
    # shellcheck disable=SC2002
    cat "${TOP_DIR}/scripts/multihost/run_cluster.sh" | base64 | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"

    # shellcheck disable=SC2087
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" << EOF &
RUN_CLUSTER_CLEANUP_OWNER=parent \
bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${SHARED_RAY_HEAD_IP}' --worker '${HOST_HF_HOME}' \
  ${worker_tpu_env_args[*]} \
  ${role_process_env_args[*]} \
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

echo "--- Waiting for the shared Ray cluster to fully form..."
wait_for_ray_cluster_members "$TOTAL_HOSTS_USED" "${RAY_CLUSTER_TIMEOUT:-900}"

dump_ray_resources "$SHARED_RAY_HEAD_IP" "Shared Prefill/Decode"

echo "--- TPU process environment on all Prefill and Decode nodes"
for host in "${PREFILL_HOSTS[@]}"; do
  dump_tpu_process_env "$host" "prefill"
done
for host in "${DECODE_HOSTS[@]}"; do
  dump_tpu_process_env "$host" "decode"
done

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
  ${ROLE_HEAD_PROCESS_ENV_ARGS[*]} \
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
  ${ROLE_HEAD_PROCESS_ENV_ARGS[*]} \
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
wait_for_vllm_prefill_and_decode "$PREFILL_VLLM_PORT" "$DECODE_HEAD_IP" "$DECODE_VLLM_PORT" 7200

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
