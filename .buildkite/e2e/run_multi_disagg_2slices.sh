#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deploy disaggregated serving across two independent TPU v7x-16 slices.
#
# The script runs on worker 0 of the Prefill slice. Each slice contains two TPU
# VM workers and owns one independent two-node Ray cluster and one two-process
# PJRT runtime:
#
#   Prefill slice: worker 0 (Ray head + vLLM API), worker 1 (Ray worker)
#   Decode slice:  worker 0 (Ray head + vLLM API), worker 1 (Ray worker)
#
# Each role uses all four dual-core chips on both workers, for tensor parallel
# size 16. Only HTTP traffic, TPUConnector side-channel traffic, and KV cache
# data cross between the two slices.
#
# Required:
#   DECODE_TPU_NAME=<name of the independent Decode v7x-16 slice>
#
# Optional:
#   DECODE_ZONE=<Decode slice zone; defaults to the Prefill zone>
#
# Network policy must permit intra-slice Ray/PJRT traffic, the Decode API port,
# and TPUConnector transfer/side-channel traffic between both slices.

set -euo pipefail

# ==============================================================================
# Fixed TPU topology
# ==============================================================================

readonly HOSTS_PER_SLICE=2
readonly CHIPS_PER_HOST=4
readonly CORES_PER_CHIP=2
readonly TENSOR_PARALLEL_SIZE=$((HOSTS_PER_SLICE * CHIPS_PER_HOST * CORES_PER_CHIP))
readonly TPU_VISIBLE_CHIPS_VALUE="0,1,2,3"
readonly TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE="2,2,1"
readonly TPU_PROCESS_BOUNDS_VALUE="1,1,2"
readonly JAX_NUM_PROCESSES_VALUE=2
readonly TPU_PROCESS_PORT=8476

# ==============================================================================
# User-configurable settings
# ==============================================================================

SSH_USER="${SSH_USER:-$(whoami)}"
SSH_KEY_EXPIRE_AFTER="${SSH_KEY_EXPIRE_AFTER:-6h}"
HOST_HF_HOME="${HOST_HF_HOME:-/tmp/hf_home}"
DECODE_HOST_HF_HOME="${DECODE_HOST_HF_HOME:-${HOST_HF_HOME}}"
LOG_DIR="${LOG_DIR:-${HOME}/logs}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-20}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RANDOM_SEED="${RANDOM_SEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
TEST_MODE="${TEST_MODE:-1}" # 1: benchmark, 2: correctness, 3: both
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
export TPU_VERSION="${TPU_VERSION:-tpu7x}"

PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_RAY_PORT="${PREFILL_RAY_PORT:-6379}"
DECODE_RAY_PORT="${DECODE_RAY_PORT:-7379}"
PREFILL_KV_TRANSFER_PORT="${PREFILL_KV_TRANSFER_PORT:-9100}"
DECODE_KV_TRANSFER_PORT="${DECODE_KV_TRANSFER_PORT:-9200}"
TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-8900}"

CONTAINER_PREFIX="${CONTAINER_PREFIX:-disagg-two-v7x16-slices}"
PREFILL_CONTAINER_NAME="${CONTAINER_PREFIX}-prefill"
DECODE_CONTAINER_NAME="${CONTAINER_PREFIX}-decode"
PROXY_CONTAINER_NAME="${CONTAINER_PREFIX}-proxy-benchmark"

case "${TEST_MODE}" in
  1 | 2 | 3) ;;
  *)
    echo "ERROR: TEST_MODE must be 1 (benchmark), 2 (correctness), or 3 (both)." >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}" "${HOST_HF_HOME}"
rm -f \
  "${LOG_DIR}/prefill.txt" \
  "${LOG_DIR}/decode.txt" \
  "${LOG_DIR}/benchmark.txt" \
  "${LOG_DIR}/proxy.txt" \
  "${LOG_DIR}/correctness.txt"

# ==============================================================================
# Metadata, slice discovery, and SSH
# ==============================================================================

get_metadata_value() {
  local path=$1
  curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/${path}" 2>/dev/null ||
    true
}

get_current_internal_ip() {
  local metadata_ip
  metadata_ip="$(get_metadata_value "instance/network-interfaces/0/ip")"
  if [[ -n "${metadata_ip}" ]]; then
    echo "${metadata_ip}"
    return
  fi
  hostname -I | awk '{print $1}'
}

PREFILL_LOCAL_IP="${PREFILL_LOCAL_IP:-$(get_current_internal_ip)}"
PREFILL_ZONE="${PREFILL_ZONE:-$(get_metadata_value "instance/zone" | awk -F/ '{print $NF}')}"
PREFILL_TPU_NAME="${PREFILL_TPU_NAME:-$(get_metadata_value "instance/description")}"
DECODE_ZONE="${DECODE_ZONE:-${PREFILL_ZONE}}"

if [[ -z "${PREFILL_TPU_NAME}" || -z "${PREFILL_ZONE}" ]]; then
  echo "ERROR: Could not discover the Prefill TPU resource name or zone." >&2
  exit 1
fi
if [[ -z "${DECODE_TPU_NAME:-}" || -z "${DECODE_ZONE}" ]]; then
  echo "ERROR: Set DECODE_TPU_NAME and, when necessary, DECODE_ZONE." >&2
  exit 1
fi
if [[ "${PREFILL_TPU_NAME}" == "${DECODE_TPU_NAME}" &&
      "${PREFILL_ZONE}" == "${DECODE_ZONE}" ]]; then
  echo "ERROR: Prefill and Decode must be different TPU v7x-16 slices." >&2
  exit 1
fi

if [[ ! -f "${HOME}/.ssh/id_rsa" ]]; then
  echo "--- Generating an SSH key for cross-worker orchestration"
  mkdir -p "${HOME}/.ssh"
  ssh-keygen -t rsa -b 4096 -N "" -f "${HOME}/.ssh/id_rsa" -q
fi
SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o IPQoS=none
  -i "${HOME}/.ssh/id_rsa"
)

run_on_host() {
  local host=$1
  shift
  if [[ "${host}" == "${PREFILL_LOCAL_IP}" ]]; then
    "$@"
    return
  fi

  local command
  printf -v command '%q ' "$@"
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "${command}"
}

DISCOVERED_ENDPOINTS=()
discover_slice_endpoints() {
  local role=$1
  local resource=$2
  local zone=$3
  local endpoint_string

  endpoint_string="$(
    gcloud compute tpus tpu-vm describe "${resource}" \
      --zone "${zone}" \
      --format='value(networkEndpoints[].ipAddress)'
  )"
  endpoint_string="${endpoint_string//;/ }"
  endpoint_string="${endpoint_string//,/ }"
  # shellcheck disable=SC2206
  DISCOVERED_ENDPOINTS=(${endpoint_string})
  if (( ${#DISCOVERED_ENDPOINTS[@]} != HOSTS_PER_SLICE )); then
    echo "ERROR: ${role} ${resource} must expose ${HOSTS_PER_SLICE} endpoints; found: ${DISCOVERED_ENDPOINTS[*]:-none}" >&2
    return 1
  fi
  local endpoint
  for endpoint in "${DISCOVERED_ENDPOINTS[@]}"; do
    if [[ ! "${endpoint}" =~ ^[A-Za-z0-9.:-]+$ ]]; then
      echo "ERROR: ${role} endpoint contains unexpected characters: ${endpoint}" >&2
      return 1
    fi
  done
}

validate_v7x16_resource() {
  local role=$1
  local resource=$2
  local zone=$3
  local accelerator_type

  accelerator_type="$(
    gcloud compute tpus tpu-vm describe "${resource}" \
      --zone "${zone}" \
      --format='value(acceleratorType)'
  )"
  if [[ "${accelerator_type}" != *"v7x-16"* &&
        "${accelerator_type}" != *"tpu7x-16"* ]]; then
    echo "ERROR: ${role} must be TPU v7x-16; got '${accelerator_type:-unknown}'." >&2
    return 1
  fi
  echo "${role}: ${resource} (${accelerator_type}, ${zone})"
}

authorize_slice_workers() {
  local role=$1
  local resource=$2
  local zone=$3
  local worker

  echo "--- Authorizing SSH access to ${role} workers"
  for ((worker = 0; worker < HOSTS_PER_SLICE; worker++)); do
    gcloud compute tpus tpu-vm ssh \
      "${SSH_USER}@${resource}" \
      --zone "${zone}" \
      --worker "${worker}" \
      --internal-ip \
      --ssh-key-file "${HOME}/.ssh/id_rsa" \
      --ssh-key-expire-after "${SSH_KEY_EXPIRE_AFTER}" \
      --command true \
      --quiet
  done
}

get_host_worker_number() {
  local host=$1
  run_on_host "${host}" curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number"
}

ORDERED_ENDPOINTS=()
order_slice_endpoints() {
  local role=$1
  shift
  local -a endpoints=("$@")
  local host worker
  ORDERED_ENDPOINTS=("" "")

  for host in "${endpoints[@]}"; do
    worker="$(get_host_worker_number "${host}")"
    if [[ ! "${worker}" =~ ^[01]$ ]]; then
      echo "ERROR: ${role} endpoint ${host} has invalid worker number '${worker:-unset}'." >&2
      return 1
    fi
    if [[ -n "${ORDERED_ENDPOINTS[$worker]}" ]]; then
      echo "ERROR: ${role} has duplicate worker number ${worker}." >&2
      return 1
    fi
    ORDERED_ENDPOINTS[$worker]="${host}"
  done

  if [[ -z "${ORDERED_ENDPOINTS[0]}" || -z "${ORDERED_ENDPOINTS[1]}" ]]; then
    echo "ERROR: ${role} did not resolve both worker 0 and worker 1." >&2
    return 1
  fi
}

validate_v7x16_resource Prefill "${PREFILL_TPU_NAME}" "${PREFILL_ZONE}"
validate_v7x16_resource Decode "${DECODE_TPU_NAME}" "${DECODE_ZONE}"

discover_slice_endpoints Prefill "${PREFILL_TPU_NAME}" "${PREFILL_ZONE}"
PREFILL_UNORDERED_HOSTS=("${DISCOVERED_ENDPOINTS[@]}")
discover_slice_endpoints Decode "${DECODE_TPU_NAME}" "${DECODE_ZONE}"
DECODE_UNORDERED_HOSTS=("${DISCOVERED_ENDPOINTS[@]}")

authorize_slice_workers Prefill "${PREFILL_TPU_NAME}" "${PREFILL_ZONE}"
authorize_slice_workers Decode "${DECODE_TPU_NAME}" "${DECODE_ZONE}"

order_slice_endpoints Prefill "${PREFILL_UNORDERED_HOSTS[@]}"
PREFILL_HOSTS=("${ORDERED_ENDPOINTS[@]}")
order_slice_endpoints Decode "${DECODE_UNORDERED_HOSTS[@]}"
DECODE_HOSTS=("${ORDERED_ENDPOINTS[@]}")

PREFILL_HEAD_IP="${PREFILL_HOSTS[0]}"
DECODE_HEAD_IP="${DECODE_HOSTS[0]}"
ALL_HOSTS=("${PREFILL_HOSTS[@]}" "${DECODE_HOSTS[@]}")

if [[ "${PREFILL_LOCAL_IP}" != "${PREFILL_HEAD_IP}" ]]; then
  echo "ERROR: Run this script on Prefill worker 0 (${PREFILL_HEAD_IP}); current host is ${PREFILL_LOCAL_IP}." >&2
  exit 1
fi

for prefill_host in "${PREFILL_HOSTS[@]}"; do
  for decode_host in "${DECODE_HOSTS[@]}"; do
    if [[ "${prefill_host}" == "${decode_host}" ]]; then
      echo "ERROR: Prefill and Decode slices overlap at ${prefill_host}." >&2
      exit 1
    fi
  done
done

echo "Prefill workers by process ID: ${PREFILL_HOSTS[*]}"
echo "Decode workers by process ID:  ${DECODE_HOSTS[*]}"
echo "Each role uses two PJRT processes and tensor parallel size ${TENSOR_PARALLEL_SIZE}."

# ==============================================================================
# Preflight, logs, and cleanup
# ==============================================================================

preflight() {
  local host
  echo "--- Preflight: validating four TPU VM workers"
  command -v docker >/dev/null
  command -v gcloud >/dev/null
  command -v curl >/dev/null

  for host in "${PREFILL_HOSTS[@]}"; do
    run_on_host "${host}" command -v docker >/dev/null
    run_on_host "${host}" command -v gcloud >/dev/null
    run_on_host "${host}" mkdir -p "${HOST_HF_HOME}"
  done
  for host in "${DECODE_HOSTS[@]}"; do
    run_on_host "${host}" command -v docker >/dev/null
    run_on_host "${host}" command -v gcloud >/dev/null
    run_on_host "${host}" mkdir -p "${DECODE_HOST_HF_HOME}"
  done
}

remote_container_exists() {
  local host=$1
  local container=$2
  run_on_host "${host}" docker inspect "${container}" >/dev/null 2>&1
}

collect_logs() {
  if remote_container_exists "${PREFILL_HEAD_IP}" "${PREFILL_CONTAINER_NAME}"; then
    run_on_host "${PREFILL_HEAD_IP}" docker exec "${PREFILL_CONTAINER_NAME}" \
      cat /root/vllm_serve_prefill.log >"${LOG_DIR}/prefill.txt" 2>/dev/null ||
      true
  fi
  if remote_container_exists "${DECODE_HEAD_IP}" "${DECODE_CONTAINER_NAME}"; then
    run_on_host "${DECODE_HEAD_IP}" docker exec "${DECODE_CONTAINER_NAME}" \
      cat /root/vllm_serve_decode.log >"${LOG_DIR}/decode.txt" 2>/dev/null ||
      true
  fi
}

print_logs() {
  local log_file
  for log_file in prefill.txt decode.txt proxy.txt correctness.txt benchmark.txt; do
    echo "--- ${log_file} ---"
    if [[ -s "${LOG_DIR}/${log_file}" ]]; then
      cat "${LOG_DIR}/${log_file}"
    else
      echo "(not found or empty)"
    fi
  done
}

cleanup() {
  local dump_logs=${1:-true}
  local status=0
  local host

  echo "--- Cleaning up two-slice v7x-16 disaggregated serving"
  if [[ "${dump_logs}" == "true" ]]; then
    collect_logs
  fi

  docker rm -f "${PROXY_CONTAINER_NAME}" >/dev/null 2>&1 || true
  for host in "${PREFILL_HOSTS[@]}"; do
    run_on_host "${host}" docker rm -f \
      "${PREFILL_CONTAINER_NAME}" >/dev/null 2>&1 || true
  done
  for host in "${DECODE_HOSTS[@]}"; do
    run_on_host "${host}" docker rm -f \
      "${DECODE_CONTAINER_NAME}" >/dev/null 2>&1 || true
  done

  docker inspect "${PROXY_CONTAINER_NAME}" >/dev/null 2>&1 && status=1
  for host in "${PREFILL_HOSTS[@]}"; do
    remote_container_exists "${host}" "${PREFILL_CONTAINER_NAME}" && status=1
  done
  for host in "${DECODE_HOSTS[@]}"; do
    remote_container_exists "${host}" "${DECODE_CONTAINER_NAME}" && status=1
  done

  if [[ "${dump_logs}" == "true" ]]; then
    print_logs
  fi
  return "${status}"
}

on_exit() {
  local exit_code=$?
  local cleanup_code=0
  trap - EXIT INT TERM
  cleanup || cleanup_code=$?
  if (( exit_code == 0 && cleanup_code != 0 )); then
    exit_code=${cleanup_code}
  fi
  exit "${exit_code}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ==============================================================================
# Ray and vLLM lifecycle helpers
# ==============================================================================

wait_for_ray_cluster() {
  local head_ip=$1
  local container=$2
  local label=$3
  local timeout=${4:-900}
  local ready_cmd
  ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); alive=sum(node.get('Alive', False) for node in ray.nodes()); raise SystemExit(0 if alive == ${HOSTS_PER_SLICE} else 1)"

  echo "Waiting for ${label} Ray cluster to register ${HOSTS_PER_SLICE} nodes..."
  local end_time=$((SECONDS + timeout))
  while (( SECONDS < end_time )); do
    if run_on_host "${head_ip}" docker exec "${container}" \
      python3 -c "${ready_cmd}" >/dev/null 2>&1; then
      echo "${label} Ray cluster is ready."
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${label} Ray cluster did not register ${HOSTS_PER_SLICE} nodes." >&2
  return 1
}

wait_for_ray_head() {
  local head_ip=$1
  local container=$2
  local label=$3
  local timeout=${4:-300}
  local ready_cmd
  ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); raise SystemExit(0 if any(node.get('Alive', False) for node in ray.nodes()) else 1)"

  echo "Waiting for ${label} Ray head..."
  local end_time=$((SECONDS + timeout))
  while (( SECONDS < end_time )); do
    if run_on_host "${head_ip}" docker exec "${container}" \
      python3 -c "${ready_cmd}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${label} Ray head did not become ready." >&2
  return 1
}

dump_ray_and_tpu_state() {
  local role=$1
  local head_ip=$2
  local container=$3
  shift 3
  local -a hosts=("$@")
  local host
  local ray_cmd
  local env_cmd

  ray_cmd="import json, ray; ray.init(address='auto', ignore_reinit_error=True); print(json.dumps(ray.nodes(), indent=2, sort_keys=True))"
  echo "--- ${role} Ray nodes"
  run_on_host "${head_ip}" docker exec "${container}" \
    python3 -c "${ray_cmd}" || true

  env_cmd='printf "CLOUD_TPU_TASK_ID=%s\nTPU_WORKER_ID=%s\nJAX_PROCESS_ID=%s\nJAX_NUM_PROCESSES=%s\nTPU_PROCESS_BOUNDS=%s\nTPU_CHIPS_PER_PROCESS_BOUNDS=%s\nTPU_PROCESS_ADDRESSES=%s\nTPU_VISIBLE_CHIPS=%s\n" "${CLOUD_TPU_TASK_ID-<unset>}" "${TPU_WORKER_ID-<unset>}" "${JAX_PROCESS_ID-<unset>}" "${JAX_NUM_PROCESSES-<unset>}" "${TPU_PROCESS_BOUNDS-<unset>}" "${TPU_CHIPS_PER_PROCESS_BOUNDS-<unset>}" "${TPU_PROCESS_ADDRESSES-<unset>}" "${TPU_VISIBLE_CHIPS-<unset>}"'
  for host in "${hosts[@]}"; do
    echo "--- ${role} TPU environment on ${host}"
    run_on_host "${host}" docker exec "${container}" \
      bash -c "${env_cmd}" || true
  done
}

vllm_process_alive() {
  local host=$1
  local container=$2
  local port=$3
  local process_check
  process_check="pgrep -af '[v]llm serve' | grep -q -- '--port ${port}'"
  run_on_host "${host}" docker exec "${container}" \
    bash -c "${process_check}" >/dev/null 2>&1
}

dump_vllm_log() {
  local host=$1
  local container=$2
  local log_path=$3
  local label=$4
  echo "+++ ${label} log (${host}:${log_path})" >&2
  run_on_host "${host}" docker exec "${container}" \
    cat "${log_path}" 2>&1 || true
}

wait_for_vllm_server() {
  local health_host=$1
  local node_host=$2
  local port=$3
  local container=$4
  local log_path=$5
  local label=$6
  local timeout=${7:-3600}

  echo "Waiting for ${label} on ${health_host}:${port}..."
  local start_time=$SECONDS
  local end_time=$((SECONDS + timeout))
  while (( SECONDS < end_time )); do
    if curl -fs "http://${health_host}:${port}/health" >/dev/null; then
      echo "${label} is healthy on ${health_host}:${port}."
      return 0
    fi
    if ! vllm_process_alive "${node_host}" "${container}" "${port}"; then
      echo "ERROR: ${label} process exited before becoming healthy." >&2
      dump_vllm_log "${node_host}" "${container}" "${log_path}" "${label}"
      return 1
    fi
    echo "${label} is still starting ($((SECONDS - start_time))s elapsed)."
    sleep 5
  done
  echo "ERROR: ${label} did not become healthy within ${timeout}s." >&2
  dump_vllm_log "${node_host}" "${container}" "${log_path}" "${label}"
  return 1
}

wait_for_proxy() {
  local timeout=${PROXY_STARTUP_TIMEOUT_SECONDS:-600}
  local end_time=$((SECONDS + timeout))
  while (( SECONDS < end_time )); do
    if curl -fs "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null; then
      return 0
    fi
    if ! docker exec "${PROXY_CONTAINER_NAME}" \
      pgrep -f '[t]oy_proxy_server' >/dev/null 2>&1; then
      echo "ERROR: Toy Proxy Server exited before becoming healthy." >&2
      docker exec "${PROXY_CONTAINER_NAME}" \
        cat /root/logs/proxy.txt 2>&1 || true
      return 1
    fi
    sleep 5
  done
  echo "ERROR: Toy Proxy Server did not become healthy." >&2
  return 1
}

smoke_test_disagg_completion() {
  local request_body
  request_body="$(python3 -c '
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "prompt": "San Francisco is a",
    "max_tokens": 1,
    "temperature": 0.0,
}))
' "${MODEL}")"

  echo "--- Running one completion through the full Prefill/Decode path"
  if ! curl -fsS "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    --data "${request_body}" >/dev/null; then
    docker exec "${PROXY_CONTAINER_NAME}" \
      cat /root/logs/proxy.txt 2>&1 || true
    dump_vllm_log "${PREFILL_HEAD_IP}" "${PREFILL_CONTAINER_NAME}" \
      /root/vllm_serve_prefill.log "vLLM Prefill"
    dump_vllm_log "${DECODE_HEAD_IP}" "${DECODE_CONTAINER_NAME}" \
      /root/vllm_serve_decode.log "vLLM Decode"
    return 1
  fi
}

# ==============================================================================
# Image preparation
# ==============================================================================

preflight

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "${DOCKER_IMAGE:-}" ]]; then
  IMAGE_NAME="${IMAGE_NAME:-us-central1-docker.pkg.dev/${PROJECT}/tpu-inference/vllm-tpu}"
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/../scripts/setup_docker_env.sh"
  setup_environment "${IMAGE_NAME}" "true"
  DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"
fi
echo "Using Docker image: ${DOCKER_IMAGE}"

echo "--- Removing containers left by an interrupted prior run"
cleanup false

echo "--- Pulling the image on remote workers"
for host in "${ALL_HOSTS[@]}"; do
  if [[ "${host}" == "${PREFILL_LOCAL_IP}" ]]; then
    continue
  fi
  run_on_host "${host}" gcloud auth configure-docker \
    us-central1-docker.pkg.dev --quiet
  run_on_host "${host}" docker pull "${DOCKER_IMAGE}"
done

# ==============================================================================
# Two independent two-node Ray/PJRT clusters
# ==============================================================================

PREFILL_PROCESS_ADDRESSES="${PREFILL_HOSTS[0]}:${TPU_PROCESS_PORT},${PREFILL_HOSTS[1]}:${TPU_PROCESS_PORT}"
DECODE_PROCESS_ADDRESSES="${DECODE_HOSTS[0]}:${TPU_PROCESS_PORT},${DECODE_HOSTS[1]}:${TPU_PROCESS_PORT}"
PREFILL_NODE_IPS="$(IFS=,; echo "${PREFILL_HOSTS[*]}")"
DECODE_NODE_IPS="$(IFS=,; echo "${DECODE_HOSTS[*]}")"
PREFILL_PROCESS_MAP="${PREFILL_HOSTS[0]}=0,${PREFILL_HOSTS[1]}=1"
DECODE_PROCESS_MAP="${DECODE_HOSTS[0]}=0,${DECODE_HOSTS[1]}=1"

launch_cluster_node() {
  local role=$1
  local host=$2
  local node_type=$3
  local process_id=$4
  local container head_ip head_port kv_port process_addresses node_ips process_map
  local min_worker_port max_worker_port dashboard_port client_port ray_start_cmd hf_home
  local -a docker_args

  if [[ "${role}" == "prefill" ]]; then
    container="${PREFILL_CONTAINER_NAME}"
    head_ip="${PREFILL_HEAD_IP}"
    head_port="${PREFILL_RAY_PORT}"
    kv_port="${PREFILL_KV_TRANSFER_PORT}"
    process_addresses="${PREFILL_PROCESS_ADDRESSES}"
    node_ips="${PREFILL_NODE_IPS}"
    process_map="${PREFILL_PROCESS_MAP}"
    min_worker_port=20000
    max_worker_port=23999
    dashboard_port=8265
    client_port=10001
    hf_home="${HOST_HF_HOME}"
  else
    container="${DECODE_CONTAINER_NAME}"
    head_ip="${DECODE_HEAD_IP}"
    head_port="${DECODE_RAY_PORT}"
    kv_port="${DECODE_KV_TRANSFER_PORT}"
    process_addresses="${DECODE_PROCESS_ADDRESSES}"
    node_ips="${DECODE_NODE_IPS}"
    process_map="${DECODE_PROCESS_MAP}"
    min_worker_port=24000
    max_worker_port=27999
    dashboard_port=8365
    client_port=11001
    hf_home="${DECODE_HOST_HF_HOME}"
  fi

  ray_start_cmd="ray start --block --min-worker-port=${min_worker_port} --max-worker-port=${max_worker_port}"
  if [[ "${node_type}" == "head" ]]; then
    ray_start_cmd+=" --head --port=${head_port} --dashboard-port=${dashboard_port} --ray-client-server-port=${client_port}"
  else
    ray_start_cmd+=" --address=${head_ip}:${head_port}"
  fi

  docker_args=(
    docker run -d
    --privileged
    --network host
    --shm-size 16G
    --name "${container}"
    -e TPU_MULTIHOST_BACKEND=ray
    -e "TPU_NODE_ID=${process_id}"
    -e "CLOUD_TPU_TASK_ID=${process_id}"
    -e "TPU_WORKER_ID=${process_id}"
    -e "JAX_PROCESS_ID=${process_id}"
    -e "JAX_NUM_PROCESSES=${JAX_NUM_PROCESSES_VALUE}"
    -e "TPU_CHIPS_PER_PROCESS_BOUNDS=${TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE}"
    -e "TPU_PROCESS_BOUNDS=${TPU_PROCESS_BOUNDS_VALUE}"
    -e "TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS_VALUE}"
    -e "TPU_PROCESS_PORT=${TPU_PROCESS_PORT}"
    -e "TPU_PROCESS_ADDRESSES=${process_addresses}"
    -e "VLLM_TPU_RAY_NODE_IPS=${node_ips}"
    -e "VLLM_TPU_RAY_PROCESS_MAP=${process_map}"
    -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
    -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
    -e RAY_DEDUP_LOGS=0
    -e "JAX_PLATFORMS="
    -e TPU_BACKEND_TYPE=jax
    -e MODEL_IMPL_TYPE=vllm
    -e "TPU_KV_TRANSFER_PORT=${kv_port}"
    -e "TPU_SIDE_CHANNEL_PORT=${TPU_SIDE_CHANNEL_PORT}"
    -e "SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE}"
    -e HF_HOME=/root/hf
    -e "HF_TOKEN=${HF_TOKEN:-}"
    -e "TPU_VERSION=${TPU_VERSION}"
    -v "${hf_home}:/root/hf"
    --entrypoint /bin/bash
    "${DOCKER_IMAGE}"
    -c "${ray_start_cmd}"
  )

  echo "--- Starting ${role} Ray ${node_type} on ${host} (process ${process_id})"
  run_on_host "${host}" "${docker_args[@]}"
}

launch_cluster_node prefill "${PREFILL_HOSTS[0]}" head 0
wait_for_ray_head "${PREFILL_HEAD_IP}" "${PREFILL_CONTAINER_NAME}" Prefill
launch_cluster_node prefill "${PREFILL_HOSTS[1]}" worker 1
wait_for_ray_cluster "${PREFILL_HEAD_IP}" "${PREFILL_CONTAINER_NAME}" Prefill

launch_cluster_node decode "${DECODE_HOSTS[0]}" head 0
wait_for_ray_head "${DECODE_HEAD_IP}" "${DECODE_CONTAINER_NAME}" Decode
launch_cluster_node decode "${DECODE_HOSTS[1]}" worker 1
wait_for_ray_cluster "${DECODE_HEAD_IP}" "${DECODE_CONTAINER_NAME}" Decode

dump_ray_and_tpu_state Prefill "${PREFILL_HEAD_IP}" \
  "${PREFILL_CONTAINER_NAME}" "${PREFILL_HOSTS[@]}"
dump_ray_and_tpu_state Decode "${DECODE_HEAD_IP}" \
  "${DECODE_CONTAINER_NAME}" "${DECODE_HOSTS[@]}"

# ==============================================================================
# vLLM Prefill and Decode servers
# ==============================================================================

printf -v quoted_model '%q' "${MODEL}"
printf -v quoted_load_format '%q' "${LOAD_FORMAT}"
PREFILL_SERVE_CMD="vllm serve ${quoted_model} \
  --port ${PREFILL_PORT} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --trust-remote-code \
  --load-format ${quoted_load_format} \
  --no-enable-prefix-caching \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}' \
  > /root/vllm_serve_prefill.log 2>&1"
DECODE_SERVE_CMD="vllm serve ${quoted_model} \
  --port ${DECODE_PORT} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --trust-remote-code \
  --load-format ${quoted_load_format} \
  --no-enable-prefix-caching \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}' \
  > /root/vllm_serve_decode.log 2>&1"

echo "--- Starting vLLM Prefill and Decode servers"
run_on_host "${PREFILL_HEAD_IP}" docker exec -d \
  "${PREFILL_CONTAINER_NAME}" bash -c "${PREFILL_SERVE_CMD}"
run_on_host "${DECODE_HEAD_IP}" docker exec -d \
  "${DECODE_CONTAINER_NAME}" bash -c "${DECODE_SERVE_CMD}"

wait_for_vllm_server 127.0.0.1 "${PREFILL_HEAD_IP}" "${PREFILL_PORT}" \
  "${PREFILL_CONTAINER_NAME}" /root/vllm_serve_prefill.log "vLLM Prefill"
wait_for_vllm_server "${DECODE_HEAD_IP}" "${DECODE_HEAD_IP}" "${DECODE_PORT}" \
  "${DECODE_CONTAINER_NAME}" /root/vllm_serve_decode.log "vLLM Decode"

# ==============================================================================
# Proxy, smoke test, benchmark, and correctness
# ==============================================================================

echo "--- Starting Toy Proxy Server locally"
docker run -d \
  --network host \
  --shm-size 16G \
  --name "${PROXY_CONTAINER_NAME}" \
  -e HF_HOME=/root/hf \
  -v "${HOST_HF_HOME}:/root/hf" \
  -v "${LOG_DIR}:/root/logs" \
  --entrypoint /bin/bash \
  "${DOCKER_IMAGE}" -c 'tail -f /dev/null'

docker exec -d "${PROXY_CONTAINER_NAME}" bash -c \
  "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py \
    --host 0.0.0.0 \
    --port ${PROXY_PORT} \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports ${PREFILL_PORT} \
    --decoder-hosts '${DECODE_HEAD_IP}' \
    --decoder-ports ${DECODE_PORT} \
    > /root/logs/proxy.txt 2>&1"
wait_for_proxy
smoke_test_disagg_completion

if [[ "${TEST_MODE}" == "1" || "${TEST_MODE}" == "3" ]]; then
  echo "--- Running disaggregated benchmark"
  timeout "${BENCHMARK_TIMEOUT_SECONDS:-1800}" \
    docker exec "${PROXY_CONTAINER_NAME}" bash -c \
    "vllm bench serve \
      --backend vllm \
      --host 127.0.0.1 \
      --port ${PROXY_PORT} \
      --model ${quoted_model} \
      --dataset-name random \
      --random-input-len ${INPUT_LEN} \
      --random-output-len ${OUTPUT_LEN} \
      --num-prompts ${NUM_PROMPTS} \
      --request-rate inf \
      --max-concurrency ${MAX_CONCURRENCY} \
      --trust-remote-code \
      --seed ${RANDOM_SEED} \
      > /root/logs/benchmark.txt 2>&1"
  docker exec "${PROXY_CONTAINER_NAME}" cat /root/logs/benchmark.txt

  failed_requests="$(
    awk '/Failed requests:/ {print $3}' "${LOG_DIR}/benchmark.txt" | tail -1
  )"
  if [[ -z "${failed_requests}" ||
        ! "${failed_requests}" =~ ^[0-9]+$ ||
        "${failed_requests}" -ne 0 ]]; then
    echo "ERROR: Benchmark reported failed requests: ${failed_requests:-unknown}" >&2
    exit 1
  fi
fi

if [[ "${TEST_MODE}" == "2" || "${TEST_MODE}" == "3" ]]; then
  echo "--- Running deterministic correctness comparison"
  timeout "${CORRECTNESS_TIMEOUT_SECONDS:-1800}" \
    docker exec "${PROXY_CONTAINER_NAME}" bash -c \
    "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py \
      --baseline_url 'http://${DECODE_HEAD_IP}:${DECODE_PORT}/v1/completions' \
      --disagg_url http://127.0.0.1:${PROXY_PORT}/v1/completions \
      --model ${quoted_model} \
      --num_requests ${NUM_PROMPTS} \
      --input_length ${INPUT_LEN} \
      --output_length ${OUTPUT_LEN} \
      > /root/logs/correctness.txt 2>&1"
  docker exec "${PROXY_CONTAINER_NAME}" cat /root/logs/correctness.txt
fi

echo "--- Tests completed successfully"
echo "Two-slice v7x-16 disaggregation passed: Prefill=${PREFILL_TPU_NAME}, Decode=${DECODE_TPU_NAME}, model=${MODEL}"
