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

# Deploy and test disaggregated serving across two independent TPU v7x-8
# instances. The script runs on the Prefill instance and controls the Decode
# instance over SSH. Each instance owns a separate one-node Ray cluster and a
# separate PJRT process; only KV cache data crosses the network.
#
# Select the Decode instance with DECODE_TPU_NAME. DECODE_ZONE may be omitted
# when both instances are in the same zone. The active gcloud identity must be
# able to access Artifact Registry and authorize SSH access to the Decode TPU
# VM.
#
# Network policy must allow the Decode API port, TPUConnector transfer ports,
# and the TPU side-channel port between the two instances.
#
# Speculative decoding is enabled only on Decode. For the default n-gram
# proposer, the script sends a repeated-token probe and verifies Decode metrics
# to prove that speculative draft tokens were actually produced.

# ==============================================================================
# Runtime safety and TPU topology
# ==============================================================================

set -euo pipefail

# A v7x-8 host exposes four dual-core chips to one PJRT process.
readonly CHIPS_PER_HOST=4
readonly CORES_PER_CHIP=2
readonly TENSOR_PARALLEL_SIZE=$((CHIPS_PER_HOST * CORES_PER_CHIP))
readonly TPU_VISIBLE_CHIPS_VALUE="0,1,2,3"

# ==============================================================================
# User-configurable paths, model settings, and test parameters
# ==============================================================================

SSH_USER="${SSH_USER:-$(whoami)}"
SSH_KEY_EXPIRE_AFTER="${SSH_KEY_EXPIRE_AFTER:-6h}"
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"
DECODE_HOST_HF_HOME="${DECODE_HOST_HF_HOME:-${HOST_HF_HOME}}"
LOG_DIR="${LOG_DIR:-${HOME}/logs}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RANDOM_SEED="${RANDOM_SEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
TEST_MODE="${TEST_MODE:-3}" # 1: benchmark, 2: correctness, 3: both
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
DEFAULT_SPECULATIVE_CONFIG='{"method":"ngram","prompt_lookup_max":5,"prompt_lookup_min":3,"num_speculative_tokens":3}'
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-${DEFAULT_SPECULATIVE_CONFIG}}"
DRAFT_MODEL_IMPL_TYPE="${DRAFT_MODEL_IMPL_TYPE:-auto}"
CORRECTNESS_NUM_PROMPTS="${CORRECTNESS_NUM_PROMPTS:-20}"
CORRECTNESS_INPUT_LEN="${CORRECTNESS_INPUT_LEN:-32}"
CORRECTNESS_OUTPUT_LEN="${CORRECTNESS_OUTPUT_LEN:-64}"
export TPU_VERSION="${TPU_VERSION:-tpu7x}"

# Keep the API, Ray control-plane, and KV transfer ports distinct because all
# services use host networking.
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_RAY_PORT="${PREFILL_RAY_PORT:-6379}"
DECODE_RAY_PORT="${DECODE_RAY_PORT:-7379}"
PREFILL_KV_TRANSFER_PORT="${PREFILL_KV_TRANSFER_PORT:-9100}"
DECODE_KV_TRANSFER_PORT="${DECODE_KV_TRANSFER_PORT:-9200}"
TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-8900}"

CONTAINER_PREFIX="${CONTAINER_PREFIX:-disagg-speculative-two-v7x8}"
PREFILL_CONTAINER_NAME="${CONTAINER_PREFIX}-prefill"
DECODE_CONTAINER_NAME="${CONTAINER_PREFIX}-decode"
PROXY_CONTAINER_NAME="${CONTAINER_PREFIX}-proxy-benchmark"

# ==============================================================================
# Input validation and local workspace initialization
# ==============================================================================

if ! python3 -c 'import json, sys; json.loads(sys.argv[1])' "${SPECULATIVE_CONFIG}"; then
  echo "ERROR: SPECULATIVE_CONFIG must be valid JSON: ${SPECULATIVE_CONFIG}" >&2
  exit 2
fi
SPECULATIVE_METHOD="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1]).get("method", ""))' "${SPECULATIVE_CONFIG}")"

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
# TPU metadata and host discovery
# ==============================================================================

# Metadata lookup failures are handled by callers so the script can also use
# explicitly supplied host values.
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

PREFILL_HOST_IP="${PREFILL_HOST_IP:-$(get_current_internal_ip)}"
PREFILL_ZONE="${PREFILL_ZONE:-$(get_metadata_value "instance/zone" | awk -F/ '{print $NF}')}"
PREFILL_TPU_NAME="${PREFILL_TPU_NAME:-$(get_metadata_value "instance/description")}"

# ==============================================================================
# SSH bootstrap and Decode host access
# ==============================================================================

if [[ ! -f "${HOME}/.ssh/id_rsa" ]]; then
  echo "--- Generating an SSH key for Prefill-to-Decode access"
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

discover_decode_host() {
  local endpoints_string
  local -a endpoints=()

  if [[ -z "${DECODE_TPU_NAME:-}" ]]; then
    echo "ERROR: Set DECODE_TPU_NAME for the independent Decode v7x-8 instance." >&2
    return 1
  fi
  DECODE_ZONE="${DECODE_ZONE:-${PREFILL_ZONE}}"
  if [[ -z "${DECODE_ZONE}" ]]; then
    echo "ERROR: Set DECODE_ZONE when the local zone cannot be discovered." >&2
    return 1
  fi

  endpoints_string="$(
    gcloud compute tpus tpu-vm describe "${DECODE_TPU_NAME}" \
      --zone "${DECODE_ZONE}" \
      --format='value(networkEndpoints[].ipAddress)'
  )"
  endpoints_string="${endpoints_string//;/ }"
  endpoints_string="${endpoints_string//,/ }"
  # shellcheck disable=SC2206
  endpoints=(${endpoints_string})
  if (( ${#endpoints[@]} != 1 )); then
    echo "ERROR: Decode resource ${DECODE_TPU_NAME} must have one v7x-8 endpoint; found: ${endpoints[*]:-none}" >&2
    return 1
  fi
  echo "${endpoints[0]}"
}

# Resolve and validate host identities before any remote mutation is attempted.
DECODE_HOST_IP="$(discover_decode_host)"
if [[ -z "${PREFILL_HOST_IP}" || -z "${DECODE_HOST_IP}" ]]; then
  echo "ERROR: Prefill and Decode host IPs must both be non-empty." >&2
  exit 1
fi
if [[ "${PREFILL_HOST_IP}" == "${DECODE_HOST_IP}" ]]; then
  echo "ERROR: Prefill and Decode must be different v7x-8 instances." >&2
  exit 1
fi
for host in "${PREFILL_HOST_IP}" "${DECODE_HOST_IP}"; do
  if [[ ! "${host}" =~ ^[A-Za-z0-9.:-]+$ ]]; then
    echo "ERROR: Host contains unexpected characters: ${host}" >&2
    exit 1
  fi
done

# Serialize an argv-style command into a safely escaped remote shell command.
run_decode_host() {
  local command
  printf -v command '%q ' "$@"
  # Each argument is individually shell-escaped before SSH sends it.
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HOST_IP}" "${command}"
}

# Direct SSH is preferred. When it is unavailable, gcloud registers the local
# public key against the named Decode TPU VM and direct SSH is retried.
authorize_decode_ssh_key() {
  if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HOST_IP}" true \
    >/dev/null 2>&1; then
    echo "--- SSH key is already authorized on Decode"
    return
  fi

  DECODE_ZONE="${DECODE_ZONE:-${PREFILL_ZONE}}"
  if [[ -z "${DECODE_ZONE}" ]]; then
    echo "ERROR: Set DECODE_ZONE so the SSH key can be registered on Decode." >&2
    return 1
  fi

  echo "--- Authorizing SSH key on Decode TPU VM ${DECODE_TPU_NAME}"
  gcloud compute tpus tpu-vm ssh \
    "${SSH_USER}@${DECODE_TPU_NAME}" \
    --zone "${DECODE_ZONE}" \
    --worker 0 \
    --internal-ip \
    --ssh-key-file "${HOME}/.ssh/id_rsa" \
    --ssh-key-expire-after "${SSH_KEY_EXPIRE_AFTER}" \
    --command true \
    --quiet

  if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HOST_IP}" true; then
    echo "ERROR: gcloud registered the key, but direct SSH to Decode still failed." >&2
    return 1
  fi
}

# ==============================================================================
# Environment preflight
# ==============================================================================

get_remote_metadata_value() {
  local path=$1
  run_decode_host curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/${path}" 2>/dev/null ||
    true
}

validate_v7x8_type() {
  local role=$1
  local accelerator_type=$2
  if [[ "${accelerator_type}" != *"v7x-8"* && "${accelerator_type}" != *"tpu7x-8"* ]]; then
    echo "ERROR: ${role} must be TPU v7x-8; metadata reported '${accelerator_type:-unknown}'." >&2
    return 1
  fi
}

preflight() {
  local prefill_type
  local decode_type

  echo "--- Preflight: validating two independent TPU v7x-8 instances"
  command -v docker >/dev/null
  command -v gcloud >/dev/null
  command -v curl >/dev/null
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HOST_IP}" true
  run_decode_host command -v docker >/dev/null
  run_decode_host command -v gcloud >/dev/null
  run_decode_host mkdir -p "${DECODE_HOST_HF_HOME}"

  prefill_type="$(get_metadata_value "instance/attributes/accelerator-type")"
  decode_type="$(get_remote_metadata_value "instance/attributes/accelerator-type")"
  validate_v7x8_type Prefill "${prefill_type}"
  validate_v7x8_type Decode "${decode_type}"

  if [[ "$(get_metadata_value "instance/attributes/agent-worker-number")" != "0" ]]; then
    echo "ERROR: Prefill v7x-8 must expose exactly one TPU worker with worker number 0." >&2
    return 1
  fi
  if [[ "$(get_remote_metadata_value "instance/attributes/agent-worker-number")" != "0" ]]; then
    echo "ERROR: Decode v7x-8 must expose exactly one TPU worker with worker number 0." >&2
    return 1
  fi

  echo "Prefill: ${PREFILL_HOST_IP} (${prefill_type})"
  echo "Decode:  ${DECODE_HOST_IP} (${decode_type})"
  echo "Each role uses one PJRT process and tensor parallel size ${TENSOR_PARALLEL_SIZE}."
}

authorize_decode_ssh_key

# The Decode account may use a different home directory, so expansion of HOME
# must happen on the Decode host rather than in the local shell.
# shellcheck disable=SC2016
DECODE_HOME="$(run_decode_host bash -c 'printf "%s" "$HOME"')"
DECODE_LOG_DIR="${DECODE_LOG_DIR:-${DECODE_HOME}/logs}"
run_decode_host mkdir -p "${DECODE_LOG_DIR}"
run_decode_host rm -f \
  "${DECODE_LOG_DIR}/decode.txt" \
  "${DECODE_LOG_DIR}/vllm_serve_decode.log"

# ==============================================================================
# Log collection and cleanup
# ==============================================================================

# Print the normalized local log set used by Buildkite artifacts and failure
# diagnostics. Missing logs are reported explicitly.
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

# Copy detached vLLM logs out of both containers before cleanup removes them.
collect_logs() {
  if docker inspect "${PREFILL_CONTAINER_NAME}" >/dev/null 2>&1; then
    docker exec "${PREFILL_CONTAINER_NAME}" \
      cat /root/vllm_serve_prefill.log >"${LOG_DIR}/prefill.txt" 2>/dev/null || true
  fi
  if run_decode_host docker inspect "${DECODE_CONTAINER_NAME}" >/dev/null 2>&1; then
    run_decode_host docker exec "${DECODE_CONTAINER_NAME}" \
      cat /root/vllm_serve_decode.log >"${LOG_DIR}/decode.txt" 2>/dev/null || true
  fi
}

cleanup_local_container() {
  local container=$1
  docker rm -f "${container}" >/dev/null 2>&1 || true
}

cleanup_decode_container() {
  local container=$1
  run_decode_host docker rm -f "${container}" >/dev/null 2>&1 || true
}

verify_container_absent() {
  local host=$1
  local container=$2
  if [[ "${host}" == "${PREFILL_HOST_IP}" ]]; then
    if docker inspect "${container}" >/dev/null 2>&1; then
      return 1
    fi
  elif run_decode_host docker inspect "${container}" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

# Cleanup is deliberately idempotent because it runs once before startup and
# again from the exit trap. The initial cleanup skips log collection so stale
# logs cannot be mistaken for output from the current run.
cleanup() {
  local status=0
  local dump_logs=${1:-true}
  echo "--- Cleaning up two-instance v7x-8 disaggregated serving"

  if [[ "${dump_logs}" == "true" ]]; then
    collect_logs
  fi

  cleanup_local_container "${PROXY_CONTAINER_NAME}"
  cleanup_local_container "${PREFILL_CONTAINER_NAME}"
  cleanup_decode_container "${DECODE_CONTAINER_NAME}"

  verify_container_absent "${PREFILL_HOST_IP}" "${PROXY_CONTAINER_NAME}" || status=1
  verify_container_absent "${PREFILL_HOST_IP}" "${PREFILL_CONTAINER_NAME}" || status=1
  verify_container_absent "${DECODE_HOST_IP}" "${DECODE_CONTAINER_NAME}" || status=1

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
# Service lifecycle and health checks
# ==============================================================================

# Each role must register exactly one live node in its independent Ray cluster.
wait_for_ray_cluster() {
  local host=$1
  local container=$2
  local label=$3
  local timeout=${4:-600}
  local ready_cmd
  ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); alive=sum(node.get('Alive', False) for node in ray.nodes()); raise SystemExit(0 if alive == 1 else 1)"

  echo "Waiting for ${label} Ray cluster on ${host}..."
  local end_time=$((SECONDS + timeout))
  while (( SECONDS < end_time )); do
    if [[ "${host}" == "${PREFILL_HOST_IP}" ]]; then
      docker exec "${container}" python3 -c "${ready_cmd}" >/dev/null 2>&1 && return 0
    elif run_decode_host docker exec "${container}" \
      python3 -c "${ready_cmd}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${label} Ray cluster did not register one node within ${timeout}s." >&2
  return 1
}

# Check the process inside the owning container instead of relying only on the
# HTTP endpoint, which distinguishes a slow startup from an early process exit.
vllm_process_alive() {
  local host=$1
  local container=$2
  local port=$3
  local process_check="pgrep -af '[v]llm serve' | grep -q -- '--port ${port}'"

  if [[ "${host}" == "${PREFILL_HOST_IP}" ]]; then
    docker exec "${container}" bash -c "${process_check}" >/dev/null 2>&1
  else
    run_decode_host docker exec "${container}" \
      bash -c "${process_check}" >/dev/null 2>&1
  fi
}

# Print the complete role log when startup fails. Routine progress messages stay
# compact so Buildkite output does not repeatedly duplicate a growing log.
dump_vllm_log() {
  local host=$1
  local container=$2
  local log_path=$3
  local label=$4
  echo "+++ ${label} log (${host}:${log_path})" >&2
  if [[ "${host}" == "${PREFILL_HOST_IP}" ]]; then
    docker exec "${container}" cat "${log_path}" 2>&1 || true
  else
    run_decode_host docker exec "${container}" cat "${log_path}" 2>&1 || true
  fi
}

# health_host is the address used by curl; node_host selects whether container
# inspection runs locally or through SSH. These differ for local Prefill.
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
    echo "${label} is still starting on ${health_host}:${port}; process is alive ($((SECONDS - start_time))s elapsed)."
    sleep 5
  done
  echo "ERROR: ${label} did not become healthy within ${timeout}s." >&2
  dump_vllm_log "${node_host}" "${container}" "${log_path}" "${label}"
  return 1
}

# The proxy is local to Prefill and routes each request through the remote
# Prefill/Decode pair.
wait_for_proxy() {
  local timeout=${PROXY_STARTUP_TIMEOUT_SECONDS:-600}
  local end_time=$((SECONDS + timeout))
  echo "Waiting for Toy Proxy Server on 127.0.0.1:${PROXY_PORT}..."
  while (( SECONDS < end_time )); do
    if curl -fs "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null; then
      return 0
    fi
    if ! docker exec "${PROXY_CONTAINER_NAME}" \
      pgrep -f '[t]oy_proxy_server' >/dev/null 2>&1; then
      echo "ERROR: Toy Proxy Server exited before becoming healthy." >&2
      docker exec "${PROXY_CONTAINER_NAME}" cat /root/logs/proxy.txt 2>&1 || true
      return 1
    fi
    sleep 5
  done
  echo "ERROR: Toy Proxy Server did not become healthy within ${timeout}s." >&2
  return 1
}

# Exercise the complete proxy -> Prefill -> Decode path before longer tests.
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
    echo "ERROR: Disaggregated completion smoke test failed." >&2
    docker exec "${PROXY_CONTAINER_NAME}" cat /root/logs/proxy.txt 2>&1 || true
    dump_vllm_log "${PREFILL_HOST_IP}" "${PREFILL_CONTAINER_NAME}" \
      /root/vllm_serve_prefill.log "vLLM Prefill"
    dump_vllm_log "${DECODE_HOST_IP}" "${DECODE_CONTAINER_NAME}" \
      /root/vllm_serve_decode.log "vLLM Decode"
    return 1
  fi
}

# Verify that the default n-gram proposer actually produces draft tokens.
assert_ngram_draft_tokens() {
  [[ "${SPECULATIVE_METHOD}" == "ngram" ]] || return 0

  local metrics
  metrics="$(run_decode_host curl -fsS "http://127.0.0.1:${DECODE_PORT}/metrics")"
  echo "--- Decode speculative-decoding metrics ---"
  printf '%s\n' "${metrics}" |
    grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens(_total)?(\{| )' ||
    true

  if ! printf '%s\n' "${metrics}" | awk '
    $1 ~ /^vllm:spec_decode_num_draft_tokens(_total)?(\{|$)/ { total += $NF }
    END { exit !(total > 0) }
  '; then
    echo "ERROR: n-gram speculative decoding produced no draft tokens." >&2
    return 1
  fi
}

# Send a repeated-token request through the complete disaggregated path so the
# metrics assertion is evidence for the composed Prefill/Decode service.
run_speculative_probe() {
  [[ "${SPECULATIVE_METHOD}" == "ngram" ]] || return 0

  local request_body
  request_body="$(python3 -c '
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "prompt": "Keep repeating: " + "a " * 20,
    "max_tokens": 32,
    "temperature": 0.0,
    "ignore_eos": True,
}))
' "${MODEL}")"

  echo "--- Running n-gram speculative-decoding probe"
  curl -fsS "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    --data "${request_body}" >/dev/null
  assert_ngram_draft_tokens
}

# ==============================================================================
# Main execution: preflight, image preparation, and stale-state cleanup
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

echo "--- Preparing Decode instance image and cache directory"
run_decode_host gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
run_decode_host docker pull "${DOCKER_IMAGE}"

# ==============================================================================
# Independent TPU runtime and Ray control planes
# ==============================================================================

# Both roles use the same single-host PJRT layout. Role-specific connector ports
# are added to the corresponding container below.
COMMON_TPU_ENV=(
  -e TPU_MULTIHOST_BACKEND=ray
  -e TPU_NODE_ID=0
  -e RAY_DEDUP_LOGS=0
  -e "JAX_PLATFORMS="
  -e TPU_BACKEND_TYPE=jax
  -e MODEL_IMPL_TYPE=vllm
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
  -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
  -e "TPU_CHIPS_PER_PROCESS_BOUNDS=1,${CHIPS_PER_HOST},1"
  -e "TPU_PROCESS_BOUNDS=1,1,1"
  -e "TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS_VALUE}"
  -e CLOUD_TPU_TASK_ID=0
  -e TPU_WORKER_ID=0
  -e JAX_PROCESS_ID=0
  -e JAX_NUM_PROCESSES=1
  -e "SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE}"
  -e HF_HOME=/root/hf
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -e "TPU_VERSION=${TPU_VERSION}"
)

echo "--- Starting independent Prefill Ray head on ${PREFILL_HOST_IP}"
docker run -d \
  --privileged \
  --network host \
  --shm-size 16G \
  --name "${PREFILL_CONTAINER_NAME}" \
  "${COMMON_TPU_ENV[@]}" \
  -e "TPU_KV_TRANSFER_PORT=${PREFILL_KV_TRANSFER_PORT}" \
  -e "TPU_SIDE_CHANNEL_PORT=${TPU_SIDE_CHANNEL_PORT}" \
  -v "${HOST_HF_HOME}:/root/hf" \
  -v "${LOG_DIR}:/root/logs" \
  --entrypoint /bin/bash \
  "${DOCKER_IMAGE}" \
  -c "ray start --block --head --port=${PREFILL_RAY_PORT}"

echo "--- Starting independent Decode Ray head on ${DECODE_HOST_IP}"
run_decode_host docker run -d \
  --privileged \
  --network host \
  --shm-size 16G \
  --name "${DECODE_CONTAINER_NAME}" \
  "${COMMON_TPU_ENV[@]}" \
  -e "TPU_KV_TRANSFER_PORT=${DECODE_KV_TRANSFER_PORT}" \
  -e "TPU_SIDE_CHANNEL_PORT=${TPU_SIDE_CHANNEL_PORT}" \
  -e "DRAFT_MODEL_IMPL_TYPE=${DRAFT_MODEL_IMPL_TYPE}" \
  -v "${DECODE_HOST_HF_HOME}:/root/hf" \
  -v "${DECODE_LOG_DIR}:/root/logs" \
  --entrypoint /bin/bash \
  "${DOCKER_IMAGE}" \
  -c "ray start --block --head --port=${DECODE_RAY_PORT}"

wait_for_ray_cluster "${PREFILL_HOST_IP}" "${PREFILL_CONTAINER_NAME}" Prefill
wait_for_ray_cluster "${DECODE_HOST_IP}" "${DECODE_CONTAINER_NAME}" Decode

# ==============================================================================
# vLLM Prefill and Decode servers
# ==============================================================================

# Shell-escape user-supplied values before embedding them in bash -c commands.
printf -v quoted_model '%q' "${MODEL}"
printf -v quoted_load_format '%q' "${LOAD_FORMAT}"
printf -v quoted_speculative_config '%q' "${SPECULATIVE_CONFIG}"
# Prefill produces KV cache entries; Decode consumes transferred entries and
# generates output tokens.
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
  --speculative-config ${quoted_speculative_config} \
  --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}' \
  > /root/vllm_serve_decode.log 2>&1"

echo "--- Starting vLLM Prefill and speculative Decode servers"
docker exec -d "${PREFILL_CONTAINER_NAME}" bash -c "${PREFILL_SERVE_CMD}"
run_decode_host docker exec -d "${DECODE_CONTAINER_NAME}" \
  bash -c "${DECODE_SERVE_CMD}"

wait_for_vllm_server 127.0.0.1 "${PREFILL_HOST_IP}" "${PREFILL_PORT}" \
  "${PREFILL_CONTAINER_NAME}" /root/vllm_serve_prefill.log "vLLM Prefill"
wait_for_vllm_server "${DECODE_HOST_IP}" "${DECODE_HOST_IP}" "${DECODE_PORT}" \
  "${DECODE_CONTAINER_NAME}" /root/vllm_serve_decode.log "vLLM Decode"

# ==============================================================================
# Request proxy and end-to-end smoke test
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
    --decoder-hosts '${DECODE_HOST_IP}' \
    --decoder-ports ${DECODE_PORT} \
    > /root/logs/proxy.txt 2>&1"
wait_for_proxy
smoke_test_disagg_completion
run_speculative_probe

# ==============================================================================
# Optional benchmark
# ==============================================================================

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
  if [[ -z "${failed_requests}" || ! "${failed_requests}" =~ ^[0-9]+$ || "${failed_requests}" -ne 0 ]]; then
    echo "ERROR: Benchmark reported failed requests: ${failed_requests:-unknown}" >&2
    exit 1
  fi
fi

# ==============================================================================
# Optional deterministic correctness test
# ==============================================================================

if [[ "${TEST_MODE}" == "2" || "${TEST_MODE}" == "3" ]]; then
  echo "--- Running deterministic correctness comparison"
  timeout "${CORRECTNESS_TIMEOUT_SECONDS:-1800}" \
    docker exec "${PROXY_CONTAINER_NAME}" bash -c \
    "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py \
      --baseline_url 'http://${DECODE_HOST_IP}:${DECODE_PORT}/v1/completions' \
      --disagg_url http://127.0.0.1:${PROXY_PORT}/v1/completions \
      --model ${quoted_model} \
      --num_requests ${CORRECTNESS_NUM_PROMPTS} \
      --input_length ${CORRECTNESS_INPUT_LEN} \
      --output_length ${CORRECTNESS_OUTPUT_LEN} \
      --prompt-mode repeated-ngram \
      > /root/logs/correctness.txt 2>&1"
  docker exec "${PROXY_CONTAINER_NAME}" cat /root/logs/correctness.txt
fi

# ==============================================================================
# Successful completion
# ==============================================================================

echo "--- Tests completed successfully"
echo "Two-instance v7x-8 speculative disaggregation passed: Prefill=${PREFILL_HOST_IP}, Decode=${DECODE_HOST_IP}, model=${MODEL}, config=${SPECULATIVE_CONFIG}"
