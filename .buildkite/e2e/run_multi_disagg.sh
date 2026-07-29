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

# We need a valid host path for the containers' Hugging Face cache bind mount.
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

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_current_internal_ip)}"
ZONE="${ZONE:-$(get_metadata_value "instance/zone" | awk -F/ '{print $NF}')}"
TPU_NAME="${TPU_NAME:-$(get_metadata_value "instance/description")}"

# Auto-generate SSH Key if it doesn't exist
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

if [[ -z "$TPU_NAME" || -z "$ZONE" ]]; then
  echo "ERROR: Could not determine the local TPU resource name or zone." >&2
  exit 1
fi

slice_endpoints="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)")"
slice_endpoints="${slice_endpoints//;/ }"
slice_endpoints="${slice_endpoints//,/ }"
# shellcheck disable=SC2206
SLICE_HOSTS=($slice_endpoints)
if (( ${#SLICE_HOSTS[@]} != 2 )); then
  echo "ERROR: This test requires one TPU7x-16 slice with exactly two hosts; found: ${SLICE_HOSTS[*]:-none}" >&2
  exit 1
fi

REMOTE_HOST=""
for host in "${SLICE_HOSTS[@]}"; do
  [[ "$host" != "$HEAD_INTERNAL_IP" ]] && REMOTE_HOST="$host"
done
if [[ -z "$REMOTE_HOST" ]]; then
  echo "ERROR: Local host ${HEAD_INTERNAL_IP} is not part of TPU slice ${TPU_NAME}: ${SLICE_HOSTS[*]}" >&2
  exit 1
fi

LOCAL_PROCESS_ID="$(get_metadata_value "instance/attributes/agent-worker-number")"
REMOTE_PROCESS_ID="$(get_remote_metadata_value "$REMOTE_HOST" "instance/attributes/agent-worker-number")"
if [[ ! "$LOCAL_PROCESS_ID" =~ ^[01]$ || ! "$REMOTE_PROCESS_ID" =~ ^[01]$ || "$LOCAL_PROCESS_ID" == "$REMOTE_PROCESS_ID" ]]; then
  echo "ERROR: Invalid TPU worker identities: local=${LOCAL_PROCESS_ID:-unset}, remote=${REMOTE_PROCESS_ID:-unset}" >&2
  exit 1
fi
ORDERED_SLICE_HOSTS=()
ORDERED_SLICE_HOSTS[$LOCAL_PROCESS_ID]="$HEAD_INTERNAL_IP"
ORDERED_SLICE_HOSTS[$REMOTE_PROCESS_ID]="$REMOTE_HOST"

accelerator_type="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone "$ZONE" --format="value(acceleratorType)")"
if [[ "$accelerator_type" != *"v7x-16"* && "$accelerator_type" != *"tpu7x-16"* ]]; then
  echo "ERROR: Expected TPU7x-16, got accelerator type: ${accelerator_type:-unknown}" >&2
  exit 1
fi

readonly HOSTS_PER_ROLE=1
readonly CHIPS_PER_HOST=4
readonly CORES_PER_CHIP=2
ALL_SLICE_HOSTS=("${ORDERED_SLICE_HOSTS[@]}")
PREFILL_HEAD_IP="$HEAD_INTERNAL_IP"
DECODE_HEAD_IP="$REMOTE_HOST"
readonly PREFILL_CONTAINER_NAME="prefill-node"
readonly DECODE_CONTAINER_NAME="decode-node"
readonly PROXY_CONTAINER_NAME="disagg-proxy-benchmark"
readonly PREFILL_RAY_PORT=6379
readonly DECODE_RAY_PORT=7379

PREFILL_TENSOR_PARALLEL_SIZE=$(( HOSTS_PER_ROLE * CHIPS_PER_HOST * CORES_PER_CHIP ))
DECODE_TENSOR_PARALLEL_SIZE=$PREFILL_TENSOR_PARALLEL_SIZE
echo "Using TPU7x-16 hosts by TPU process ID: ${ORDERED_SLICE_HOSTS[*]}"
echo "Prefill owns all chips on ${PREFILL_HEAD_IP}; Decode owns all chips on ${DECODE_HEAD_IP}."
echo "Tensor parallel size per role: ${PREFILL_TENSOR_PARALLEL_SIZE}"

# TPU7x-16 is one indivisible 2x2x2 topology. Prefill and Decode keep separate
# Ray control planes, but their TPU workers join one two-process PJRT runtime.
# Each role then builds its mesh from the eight devices local to its physical
# VM, without assuming anything about TPU7x device ID numbering.
readonly TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE="2,2,1"
readonly TPU_VISIBLE_CHIPS_VALUE="0,1,2,3"
readonly TPU_PROCESS_PORT_VALUE=8476
TPU_PROCESS_ADDRESSES="${ORDERED_SLICE_HOSTS[0]}:${TPU_PROCESS_PORT_VALUE},${ORDERED_SLICE_HOSTS[1]}:${TPU_PROCESS_PORT_VALUE}"

# Populates PROCESS_IDENTITY_ENV_ARGS with the physical TPU process identity.
# The two roles must use distinct IDs because they share one PJRT runtime.
PROCESS_IDENTITY_ENV_ARGS=()
build_process_identity_env_args() {
  local process_id=$1
  local physical_worker_id=$2
  PROCESS_IDENTITY_ENV_ARGS=(
    -e CLOUD_TPU_TASK_ID="${process_id}"
    -e TPU_WORKER_ID="${physical_worker_id}"
    -e JAX_PROCESS_ID="${process_id}"
  )
}

COMMON_TPU_ENV_ARGS=(
  -e TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE}"
  -e TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS_VALUE}"
  -e TPU_PROCESS_BOUNDS="1,1,2"
  -e TPU_PROCESS_ADDRESSES="${TPU_PROCESS_ADDRESSES}"
  -e TPU_PROCESS_PORT="${TPU_PROCESS_PORT_VALUE}"
  -e JAX_NUM_PROCESSES=2
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
  -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
)
PREFILL_TPU_ENV_ARGS=("${COMMON_TPU_ENV_ARGS[@]}")
DECODE_TPU_ENV_ARGS=("${COMMON_TPU_ENV_ARGS[@]}")

build_process_identity_env_args "$LOCAL_PROCESS_ID" "$LOCAL_PROCESS_ID"
PREFILL_HEAD_PROCESS_ENV_ARGS=("${PROCESS_IDENTITY_ENV_ARGS[@]}")
build_process_identity_env_args "$REMOTE_PROCESS_ID" "$REMOTE_PROCESS_ID"
DECODE_HEAD_PROCESS_ENV_ARGS=("${PROCESS_IDENTITY_ENV_ARGS[@]}")

echo "Prefill Ray head: ${PREFILL_HEAD_IP}:${PREFILL_RAY_PORT}"
echo "Decode Ray head: ${DECODE_HEAD_IP}:${DECODE_RAY_PORT}"
echo "Prefill actor host: ${PREFILL_HEAD_IP}"
echo "Decode actor host: ${DECODE_HEAD_IP}"
echo "Shared PJRT process addresses: ${TPU_PROCESS_ADDRESSES}"
echo "Each role uses its eight local JAX devices."

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
  local -a containers=(
    node
    "$PREFILL_CONTAINER_NAME"
    "$DECODE_CONTAINER_NAME"
    "$PROXY_CONTAINER_NAME"
  )
  echo "  Cleaning up containers on ${host}..."

  run_host_script "$host" "${containers[@]}" <<'CLEANUP_HOST'
status=0
for container in "$@"; do
  if docker inspect "$container" >/dev/null 2>&1; then
    if ! docker rm -f "$container" >/dev/null 2>&1; then
      echo "Warning: failed to remove container $container" >&2
      status=1
    fi
  fi
done
for container in "$@"; do
  if docker inspect "$container" >/dev/null 2>&1; then
    echo "ERROR: container $container still exists after cleanup" >&2
    status=1
  fi
done
exit $status
CLEANUP_HOST
}

cleanup_start_scripts() {
  local status=0

  if ! rm -f /tmp/start_prefill.sh /tmp/start_decode.sh \
    /tmp/prefill_tpu_probe.log /tmp/decode_tpu_probe.log; then
    echo "ERROR: failed to remove local vLLM start scripts." >&2
    status=1
  fi
  if [[ -e /tmp/start_prefill.sh || -e /tmp/start_decode.sh || \
    -e /tmp/prefill_tpu_probe.log || -e /tmp/decode_tpu_probe.log ]]; then
    echo "ERROR: local vLLM start scripts still exist after cleanup." >&2
    status=1
  fi

  if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" \
    'rm -f "$HOME/tpu-inference/scripts/start_decode.sh" && test ! -e "$HOME/tpu-inference/scripts/start_decode.sh"'; then
    echo "ERROR: failed to verify the remote Decode start script cleanup on ${DECODE_HEAD_IP}." >&2
    status=1
  fi

  return "$status"
}

collect_remote_logs() {
  echo "  Collecting server logs before cleanup..."
  # Prefill server log (on local head node)
  docker exec "$PREFILL_CONTAINER_NAME" cat /root/vllm_serve_prefill.log \
    > "${LOG_DIR}/prefill.txt" 2>/dev/null || true
  # Decode server log (on remote decode head)
  if [[ -n "${DECODE_HEAD_IP:-}" && "${DECODE_HEAD_IP}" != "${HEAD_INTERNAL_IP}" ]]; then
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" \
      "docker exec '$DECODE_CONTAINER_NAME' cat /root/vllm_serve_decode.log 2>/dev/null" \
      > "${LOG_DIR}/decode.txt" 2>/dev/null || true
  elif [[ -n "${DECODE_HEAD_IP:-}" ]]; then
    docker exec "$DECODE_CONTAINER_NAME" cat /root/vllm_serve_decode.log \
      > "${LOG_DIR}/decode.txt" 2>/dev/null || true
  fi
}

cleanup() {
  local status=0
  local dump_logs="${1:-true}"
  echo "--- Cleaning up multi-host disaggregated serving"

  if [[ "$dump_logs" == "true" ]]; then
    collect_remote_logs
  fi

  for ip in "${ALL_SLICE_HOSTS[@]}"; do
    if ! cleanup_host "$ip"; then
      echo "ERROR: failed to verify container cleanup on ${ip}." >&2
      status=1
    fi
  done

  if ! cleanup_start_scripts; then
    status=1
  fi

  if [[ "$dump_logs" == "true" ]]; then
    print_logs
  fi
  return "$status"
}

on_exit() {
  local exit_code=$?
  local cleanup_code=0
  trap - EXIT INT TERM
  cleanup || cleanup_code=$?
  if (( exit_code == 0 && cleanup_code != 0 )); then
    exit_code=$cleanup_code
  fi
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
  local container=$2
  local port=$3
  local process_check="pgrep -af '[v]llm serve' | grep -q -- '--port ${port}'"

  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$container" bash -c "$process_check" >/dev/null 2>&1
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker exec '$container' bash -c \"$process_check\"" >/dev/null 2>&1
  fi
}

dump_vllm_server_log() {
  local host=$1
  local container=$2
  local log_path=$3
  local service_name=$4

  echo "+++ 📄 ${service_name} log (${host}:${log_path})"
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$container" cat "$log_path" 2>/dev/null || true
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker exec '$container' cat '${log_path}' 2>/dev/null || true" || true
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

    if ! vllm_server_process_alive "localhost" "$PREFILL_CONTAINER_NAME" "$prefill_port"; then
      echo "Error: vLLM Prefill process exited before becoming healthy."
      dump_vllm_server_log "localhost" "$PREFILL_CONTAINER_NAME" "/root/vllm_serve_prefill.log" "vLLM Prefill"
      return 1
    fi

    if ! vllm_server_process_alive "$decode_host" "$DECODE_CONTAINER_NAME" "$decode_port"; then
      echo "Error: vLLM Decode process exited before becoming healthy."
      dump_vllm_server_log "$decode_host" "$DECODE_CONTAINER_NAME" "/root/vllm_serve_decode.log" "vLLM Decode"
      return 1
    fi

    sleep 5
  done

  echo "Error: vLLM Prefill and Decode failed to become healthy within ${timeout}s."
  dump_vllm_server_log "localhost" "$PREFILL_CONTAINER_NAME" "/root/vllm_serve_prefill.log" "vLLM Prefill"
  dump_vllm_server_log "$decode_host" "$DECODE_CONTAINER_NAME" "/root/vllm_serve_decode.log" "vLLM Decode"
  return 1
}

wait_for_ray_head() {
  local host=$1
  local port=$2
  local timeout=${3:-300}
  echo "Waiting for Ray head on ${host}:${port} to become available..."
  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    if nc -z -w2 "${host}" "$port" &>/dev/null; then
      echo "Ray head is reachable on ${host}:${port}"
      return 0
    fi
    sleep 5
  done
  echo "Error: Ray head failed to start within ${timeout}s."
  return 1
}

wait_for_ray_cluster_members() {
  local host=$1
  local container=$2
  local label=$3
  local expected_nodes=$4
  local timeout=${5:-900}
  local ready_cmd
  ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); alive=sum(node.get('Alive', False) for node in ray.nodes()); raise SystemExit(0 if alive == ${expected_nodes} else 1)"

  echo "Waiting for ${label} Ray cluster to register ${expected_nodes} nodes..."
  local end_time=$((SECONDS + timeout))
  while [[ $SECONDS -lt $end_time ]]; do
    local ready=0
    if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
      docker exec "$container" python3 -c "$ready_cmd" >/dev/null 2>&1 && ready=1
    elif ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "docker exec '$container' python3 -c \"$ready_cmd\"" >/dev/null 2>&1; then
      ready=1
    fi
    if (( ready == 1 )); then
      echo "${label} Ray cluster has registered ${expected_nodes} nodes."
      return 0
    fi
    sleep 5
  done

  echo "Error: ${label} Ray cluster did not register ${expected_nodes} nodes within ${timeout}s." >&2
  return 1
}

dump_ray_resources() {
  local host=$1
  local label=$2
  local container=$3
  local ray_dump_cmd="import json, ray; ray.init(address='auto', ignore_reinit_error=True); print(json.dumps(ray.nodes(), indent=2, sort_keys=True))"

  echo "--- Ray resources for ${label} cluster (${host})"
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$container" python3 -c "$ray_dump_cmd" || true
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "docker exec '$container' python3 -c \"${ray_dump_cmd}\"" || true
  fi
}

dump_tpu_process_env() {
  local host=$1
  local role=$2
  local container=$3
  local dump_cmd
  dump_cmd='printf "CLOUD_TPU_TASK_ID=%s\nTPU_WORKER_ID=%s\nJAX_PROCESS_ID=%s\nJAX_NUM_PROCESSES=%s\nJAX_PLATFORMS=%s\nTPU_PROCESS_BOUNDS=%s\nTPU_CHIPS_PER_PROCESS_BOUNDS=%s\nTPU_CHIPS_PER_HOST_BOUNDS=%s\nTPU_HOST_BOUNDS=%s\nTPU_PROCESS_ADDRESSES=%s\nTPU_VISIBLE_CHIPS=%s\nRAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=%s\n" "${CLOUD_TPU_TASK_ID-<unset>}" "${TPU_WORKER_ID-<unset>}" "${JAX_PROCESS_ID-<unset>}" "${JAX_NUM_PROCESSES-<unset>}" "${JAX_PLATFORMS-<unset>}" "${TPU_PROCESS_BOUNDS-<unset>}" "${TPU_CHIPS_PER_PROCESS_BOUNDS-<unset>}" "${TPU_CHIPS_PER_HOST_BOUNDS-<unset>}" "${TPU_HOST_BOUNDS-<unset>}" "${TPU_PROCESS_ADDRESSES-<unset>}" "${TPU_VISIBLE_CHIPS-<unset>}" "${RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS-<unset>}"'

  echo "--- TPU process environment: ${role} (${host})"
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$container" bash -c "$dump_cmd"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "docker exec '$container' bash -c '$dump_cmd'"
  fi
}

validate_tpu_device_ids() {
  local role=$1
  local value=$2
  local -a device_ids
  local device_id
  IFS=, read -r -a device_ids <<< "$value"
  if (( ${#device_ids[@]} != PREFILL_TENSOR_PARALLEL_SIZE )); then
    echo "ERROR: ${role} probe returned ${#device_ids[@]} device IDs; expected ${PREFILL_TENSOR_PARALLEL_SIZE}: ${value:-none}" >&2
    return 1
  fi
  for device_id in "${device_ids[@]}"; do
    if [[ ! "$device_id" =~ ^[0-9]+$ ]]; then
      echo "ERROR: ${role} probe returned an invalid device ID: ${device_id}" >&2
      return 1
    fi
  done
}

discover_tpu_device_ids() {
  local probe_code
  local remote_probe_cmd
  local prefill_probe_pid decode_probe_pid
  local prefill_status=0 decode_status=0
  local prefill_log=/tmp/prefill_tpu_probe.log
  local decode_log=/tmp/decode_tpu_probe.log
  local -a decode_device_ids
  probe_code='import jax; print("TPU_LOCAL_DEVICE_IDS=" + ",".join(str(device.id) for device in jax.local_devices(backend="tpu")), flush=True)'

  rm -f "$prefill_log" "$decode_log"
  printf -v remote_probe_cmd '%q ' docker exec "$DECODE_CONTAINER_NAME" \
    python3 -c "$probe_code"

  echo "--- Discovering physical JAX device IDs on both hosts"
  docker exec "$PREFILL_CONTAINER_NAME" python3 -c "$probe_code" \
    > "$prefill_log" 2>&1 &
  prefill_probe_pid=$!
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "$remote_probe_cmd" \
    > "$decode_log" 2>&1 &
  decode_probe_pid=$!

  wait "$prefill_probe_pid" || prefill_status=$?
  wait "$decode_probe_pid" || decode_status=$?
  if (( prefill_status != 0 || decode_status != 0 )); then
    echo "ERROR: JAX device probe failed: prefill=${prefill_status}, decode=${decode_status}" >&2
    echo "--- Prefill JAX device probe log" >&2
    cat "$prefill_log" >&2 || true
    echo "--- Decode JAX device probe log" >&2
    cat "$decode_log" >&2 || true
    return 1
  fi

  PREFILL_DEVICE_INDEXES="$(sed -n 's/^TPU_LOCAL_DEVICE_IDS=//p' "$prefill_log" | tail -n 1)"
  DECODE_DEVICE_INDEXES="$(sed -n 's/^TPU_LOCAL_DEVICE_IDS=//p' "$decode_log" | tail -n 1)"
  validate_tpu_device_ids Prefill "$PREFILL_DEVICE_INDEXES"
  validate_tpu_device_ids Decode "$DECODE_DEVICE_INDEXES"

  local combined=",${PREFILL_DEVICE_INDEXES},"
  local decode_device_id
  IFS=, read -r -a decode_device_ids <<< "$DECODE_DEVICE_INDEXES"
  for decode_device_id in "${decode_device_ids[@]}"; do
    if [[ "$combined" == *",${decode_device_id},"* ]]; then
      echo "ERROR: Prefill and Decode device IDs overlap at ${decode_device_id}." >&2
      return 1
    fi
  done

  echo "Prefill local JAX device IDs: ${PREFILL_DEVICE_INDEXES}"
  echo "Decode local JAX device IDs: ${DECODE_DEVICE_INDEXES}"
  rm -f "$prefill_log" "$decode_log"
  sleep "${TPU_PROBE_SETTLE_SECONDS:-5}"
}


PROJECT="$(gcloud config get-value project)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

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
cleanup false

# Each physical host runs one role-local Ray head and one complete TPU runtime.
# Keep the launch details here instead of changing shared cluster behavior.
run_host_script "$REMOTE_HOST" "$DOCKER_IMAGE" <<'PREPARE_REMOTE_HOST'
set -euo pipefail
image=$1
docker system prune -a --volumes -f >/dev/null 2>&1 || true
gcloud auth configure-docker us-east5-docker.pkg.dev
gcloud auth configure-docker us-central1-docker.pkg.dev
docker pull "$image"
PREPARE_REMOTE_HOST

REMOTE_HOME="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${REMOTE_HOST}" 'printf "%s" "$HOME"')"
if [[ ! "$REMOTE_HOME" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  echo "ERROR: Could not determine a safe home directory on ${REMOTE_HOST}: ${REMOTE_HOME:-unset}" >&2
  exit 1
fi

LOCAL_OPTIONAL_MOUNT_ARGS=()
[[ -d "$HOME/.config/gcloud" ]] && \
  LOCAL_OPTIONAL_MOUNT_ARGS+=(-v "$HOME/.config/gcloud:/root/.config/gcloud")
[[ -d /mnt/disks/checkpoint ]] && \
  LOCAL_OPTIONAL_MOUNT_ARGS+=(-v /mnt/disks/checkpoint:/mnt/disks/checkpoint)

REMOTE_OPTIONAL_MOUNT_ARGS=()
if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${REMOTE_HOST}" test -d "$REMOTE_HOME/.config/gcloud"; then
  REMOTE_OPTIONAL_MOUNT_ARGS+=(-v "$REMOTE_HOME/.config/gcloud:/root/.config/gcloud")
fi
if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${REMOTE_HOST}" test -d /mnt/disks/checkpoint; then
  REMOTE_OPTIONAL_MOUNT_ARGS+=(-v /mnt/disks/checkpoint:/mnt/disks/checkpoint)
fi

launch_cluster_node() {
  local role=$1
  local host=$2
  local node_type=$3
  local process_id=$4
  local physical_worker_id
  local head_ip container head_port dashboard_port client_port min_worker_port max_worker_port agent_port
  local ray_start_cmd remote_docker_cmd
  local -a tpu_env_args mount_args docker_args
  local -a runtime_env_args=(
    -e VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
    -e NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-0}"
    -e MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-}"
    -e MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-}"
    -e MOE_ALL_GATHER_ACTIVATION_DTYPE="${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}"
    -e FORCE_MOE_RANDOM_ROUTING="${FORCE_MOE_RANDOM_ROUTING:-}"
  )

  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    physical_worker_id="$LOCAL_PROCESS_ID"
  else
    physical_worker_id="$REMOTE_PROCESS_ID"
  fi
  build_process_identity_env_args "$process_id" "$physical_worker_id"
  local -a identity_env_args=("${PROCESS_IDENTITY_ENV_ARGS[@]}")
  if [[ "$role" == "prefill" ]]; then
    head_ip="$PREFILL_HEAD_IP"
    container="$PREFILL_CONTAINER_NAME"
    head_port="$PREFILL_RAY_PORT"
    dashboard_port=8265
    client_port=10001
    min_worker_port=20000
    max_worker_port=23999
    agent_port=52365
    tpu_env_args=("${PREFILL_TPU_ENV_ARGS[@]}")
  else
    head_ip="$DECODE_HEAD_IP"
    container="$DECODE_CONTAINER_NAME"
    head_port="$DECODE_RAY_PORT"
    dashboard_port=8365
    client_port=11001
    min_worker_port=24000
    max_worker_port=27999
    agent_port=53365
    tpu_env_args=("${DECODE_TPU_ENV_ARGS[@]}")
  fi

  ray_start_cmd="ray start --block --min-worker-port=${min_worker_port} --max-worker-port=${max_worker_port} --dashboard-agent-listen-port=${agent_port}"
  if [[ "$node_type" == "--head" ]]; then
    ray_start_cmd+=" --head --port=${head_port} --dashboard-port=${dashboard_port} --ray-client-server-port=${client_port}"
  else
    ray_start_cmd+=" --address=${head_ip}:${head_port}"
  fi

  mount_args=(-v "$HOST_HF_HOME:/root/.cache/huggingface")
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    mount_args+=("${LOCAL_OPTIONAL_MOUNT_ARGS[@]}")
  else
    mount_args+=("${REMOTE_OPTIONAL_MOUNT_ARGS[@]}")
  fi
  docker_args=(
    docker run --detach
    --privileged
    --entrypoint /bin/bash
    --network host
    --shm-size=16G
    --name "$container"
    "${mount_args[@]}"
    "${tpu_env_args[@]}"
    "${identity_env_args[@]}"
    "${runtime_env_args[@]}"
    -e HF_TOKEN="${HF_TOKEN:-}"
    -e TPU_MULTIHOST_BACKEND=ray
    -e JAX_PLATFORMS=''
    -e TPU_BACKEND_TYPE=jax
    -e MODEL_IMPL_TYPE=vllm
    "$DOCKER_IMAGE" -c "$ray_start_cmd"
  )

  echo "--- Starting ${role} Ray ${node_type#--} on ${host} (process ${process_id})"
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    "${docker_args[@]}"
  else
    printf -v remote_docker_cmd '%q ' "${docker_args[@]}"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$remote_docker_cmd"
  fi
  sleep 15
}

# The two single-node Ray clusters are independent control planes. Their TPU
# actors use different physical process IDs and join the shared PJRT topology.
launch_cluster_node prefill "$PREFILL_HEAD_IP" --head "$LOCAL_PROCESS_ID"
wait_for_ray_head "$PREFILL_HEAD_IP" "$PREFILL_RAY_PORT"
wait_for_ray_cluster_members "$PREFILL_HEAD_IP" "$PREFILL_CONTAINER_NAME" Prefill \
  "$HOSTS_PER_ROLE" "${RAY_CLUSTER_TIMEOUT:-900}"

launch_cluster_node decode "$DECODE_HEAD_IP" --head "$REMOTE_PROCESS_ID"
wait_for_ray_head "$DECODE_HEAD_IP" "$DECODE_RAY_PORT"
wait_for_ray_cluster_members "$DECODE_HEAD_IP" "$DECODE_CONTAINER_NAME" Decode \
  "$HOSTS_PER_ROLE" "${RAY_CLUSTER_TIMEOUT:-900}"

dump_ray_resources "$PREFILL_HEAD_IP" Prefill "$PREFILL_CONTAINER_NAME"
dump_ray_resources "$DECODE_HEAD_IP" Decode "$DECODE_CONTAINER_NAME"

echo "--- TPU process environment on Prefill and Decode nodes"
dump_tpu_process_env "$PREFILL_HEAD_IP" prefill "$PREFILL_CONTAINER_NAME"
dump_tpu_process_env "$DECODE_HEAD_IP" decode "$DECODE_CONTAINER_NAME"

# device_indexes is interpreted as the actual jax.Device.id, not as an offset
# into jax.devices(). Probe both hosts together so the values match this slice.
PREFILL_DEVICE_INDEXES=""
DECODE_DEVICE_INDEXES=""
discover_tpu_device_ids

# -----------------------------------------------------------------
# 3. Start vLLM Prefill & Decode Servers
# -----------------------------------------------------------------
echo "--- Preparing vLLM Prefill server on Head Node locally"
PREFILL_VLLM_PORT="8400"
PREFILL_DOCKER_EXEC_ENV_ARGS="${PREFILL_TPU_ENV_ARGS[*]}"

cat << EOF > /tmp/start_prefill.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${PREFILL_HEAD_PROCESS_ENV_ARGS[*]} \
  ${PREFILL_DOCKER_EXEC_ENV_ARGS} \
  ${PREFILL_CONTAINER_NAME} bash -c "vllm serve ${MODEL} \
    --port ${PREFILL_VLLM_PORT} \
    --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
    --additional-config '{\"sharding\": {\"sharding_strategy\": {\"device_indexes\": [${PREFILL_DEVICE_INDEXES}]}}}' \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_producer\"}' \
    --max-model-len 1024 > /root/vllm_serve_prefill.log 2>&1"
set +x
EOF

chmod +x /tmp/start_prefill.sh

echo "--- Preparing vLLM Decode server on remote Head Node (${DECODE_HEAD_IP})"
DECODE_VLLM_PORT="9400"
DECODE_DOCKER_EXEC_ENV_ARGS="${DECODE_TPU_ENV_ARGS[*]}"

cat << EOF > /tmp/start_decode.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${DECODE_HEAD_PROCESS_ENV_ARGS[*]} \
  ${DECODE_DOCKER_EXEC_ENV_ARGS} \
  ${DECODE_CONTAINER_NAME} bash -c "vllm serve ${MODEL} \
    --port ${DECODE_VLLM_PORT} \
    --tensor-parallel-size ${DECODE_TENSOR_PARALLEL_SIZE} \
    --additional-config '{\"sharding\": {\"sharding_strategy\": {\"device_indexes\": [${DECODE_DEVICE_INDEXES}]}}}' \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_consumer\"}' \
    --max-model-len 1024 > /root/vllm_serve_decode.log 2>&1"
set +x
EOF

chmod +x /tmp/start_decode.sh

echo "--- Starting Prefill and Decode together so both PJRT processes can rendezvous"
bash /tmp/start_prefill.sh &
prefill_submit_pid=$!
base64 < /tmp/start_decode.sh | \
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HEAD_IP}" "base64 -d | bash" &
decode_submit_pid=$!
wait "$prefill_submit_pid"
wait "$decode_submit_pid"

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
    --name "$PROXY_CONTAINER_NAME" \
    -e HF_HOME="/root/hf" \
    -v "${HOST_HF_HOME}:/root/hf" \
    -v "$LOG_DIR:/root/logs" \
    --entrypoint /bin/bash \
    "${DOCKER_IMAGE}" -c "tail -f /dev/null"

echo "--- Starting Toy Proxy Server inside container..."
docker exec -d "$PROXY_CONTAINER_NAME" /bin/bash -c "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py \
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
    docker exec "$PROXY_CONTAINER_NAME" /bin/bash -c "vllm bench serve \
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
    docker exec "$PROXY_CONTAINER_NAME" cat /root/logs/benchmark.txt
fi

if [ "$TEST_MODE" = "2" ] || [ "$TEST_MODE" = "3" ]; then
    echo "--- Running Correctness Test inside container..."
    timeout "${CORRECTNESS_TIMEOUT_SECONDS:-1800}" \
    docker exec "$PROXY_CONTAINER_NAME" /bin/bash -c "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py \
        --baseline_url http://${DECODE_HEAD_IP}:${DECODE_VLLM_PORT}/v1/completions \
        --disagg_url http://127.0.0.1:8000/v1/completions \
        --model ${MODEL} \
        --num_requests ${NUM_PROMPTS} \
        --input_length ${INPUT_LEN} \
        --output_length ${OUTPUT_LEN} > /root/logs/correctness.txt 2>&1"

    echo "--- Correctness Results ---"
    docker exec "$PROXY_CONTAINER_NAME" cat /root/logs/correctness.txt
fi

echo "--- Tests completed successfully ---"
