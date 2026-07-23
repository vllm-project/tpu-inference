#!/bin/bash
# Copyright 2026 Google LLC
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

set -euo pipefail

CONTAINER_PREFIX="${CONTAINER_PREFIX:-disagg-node}"
PREFILL_CONTAINER="${CONTAINER_PREFIX}-prefill"
DECODE_CONTAINER="${CONTAINER_PREFIX}-decode"
PROXY_CONTAINER="${CONTAINER_PREFIX}-proxy-benchmark"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-20}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RANDOM_SEED="${RANDOM_SEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
TEST_MODE="${TEST_MODE:-1}" # 1: benchmark, 2: correctness, 3: both
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"
LOG_DIR="${HOME}/logs"
SSH_USER="${SSH_USER:-$(whoami)}"

mkdir -p "${LOG_DIR}"
rm -f \
  "${LOG_DIR}/prefill.txt" \
  "${LOG_DIR}/decode.txt" \
  "${LOG_DIR}/benchmark.txt" \
  "${LOG_DIR}/proxy.txt" \
  "${LOG_DIR}/correctness.txt"

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
  else
    hostname -I | awk '{print $1}'
  fi
}

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_current_internal_ip)}"

if [[ ! -f "${HOME}/.ssh/id_rsa" ]]; then
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
  local zone
  local tpu_name
  local all_ips
  local ip
  local -a endpoints=()
  local -a remote_endpoints=()

  if [[ -n "${DECODE_HOST_IP:-}" ]]; then
    echo "${DECODE_HOST_IP}"
    return
  fi

  if [[ -n "${WORKER_IPS:-}" ]]; then
    IFS=',' read -r -a endpoints <<< "${WORKER_IPS}"
  else
    zone="${ZONE:-$(get_metadata_value "instance/zone" | awk -F/ '{print $NF}')}"
    tpu_name="${TPU_NAME:-$(get_metadata_value "instance/description")}"
    if [[ -z "${zone}" || -z "${tpu_name}" ]]; then
      echo "Unable to determine ZONE or TPU_NAME; set DECODE_HOST_IP." >&2
      return 1
    fi
    all_ips="$(
      gcloud compute tpus tpu-vm describe "${tpu_name}" \
        --zone "${zone}" \
        --format="value(networkEndpoints[].ipAddress)"
    )"
    all_ips="${all_ips//;/ }"
    all_ips="${all_ips//,/ }"
    # shellcheck disable=SC2206
    endpoints=(${all_ips})
  fi

  for ip in "${endpoints[@]}"; do
    [[ -n "${ip}" && "${ip}" != "${HEAD_INTERNAL_IP}" ]] || continue
    remote_endpoints+=("${ip}")
  done
  if (( ${#remote_endpoints[@]} != 1 )); then
    echo "Expected exactly one Decode host for tpu7x-16; found: ${remote_endpoints[*]:-none}" >&2
    return 1
  fi
  echo "${remote_endpoints[0]}"
}

DECODE_HOST_IP="$(discover_decode_host)"
if [[ "${DECODE_HOST_IP}" == "${HEAD_INTERNAL_IP}" ]]; then
  echo "Decode host must be different from the Prefill host." >&2
  exit 1
fi
if [[ ! "${DECODE_HOST_IP}" =~ ^[A-Za-z0-9.:-]+$ ]]; then
  echo "Decode host contains unexpected characters: ${DECODE_HOST_IP}" >&2
  exit 1
fi

run_decode_host() {
  local command
  printf -v command '%q ' "$@"
  # The command is individually shell-escaped above before SSH sends it.
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${DECODE_HOST_IP}" "${command}"
}

DECODE_HOME="$(
  # Expansion of HOME must happen on the Decode host.
  # shellcheck disable=SC2016
  run_decode_host bash -c 'printf "%s" "$HOME"'
)"
DECODE_LOG_DIR="${DECODE_HOME}/logs"
run_decode_host mkdir -p "${DECODE_LOG_DIR}"
run_decode_host rm -f \
  "${DECODE_LOG_DIR}/decode.txt" \
  "${DECODE_LOG_DIR}/vllm_serve_decode.log"

print_logs() {
  local log_file
  for log_file in prefill.txt decode.txt proxy.txt correctness.txt benchmark.txt; do
    echo "--- ${log_file} ---"
    if [[ -f "${LOG_DIR}/${log_file}" ]]; then
      cat "${LOG_DIR}/${log_file}"
    else
      echo "File not found."
    fi
  done
}

stop_local_container() {
  local container=$1
  if ! docker inspect "${container}" >/dev/null 2>&1; then
    return
  fi
  docker exec "${container}" bash -c \
    "pkill -TERM -f '[v]llm serve|[A]PIServer|[E]ngineCore|[t]oy_proxy_server' || true" \
    >/dev/null 2>&1 || true
  sleep 2
  docker exec "${container}" ray stop --force >/dev/null 2>&1 || true
  docker rm -f "${container}" >/dev/null 2>&1
}

stop_decode_container() {
  if ! run_decode_host docker inspect "${DECODE_CONTAINER}" >/dev/null 2>&1; then
    return
  fi
  run_decode_host docker exec "${DECODE_CONTAINER}" bash -c \
    "pkill -TERM -f '[v]llm serve|[A]PIServer|[E]ngineCore' || true" \
    >/dev/null 2>&1 || true
  sleep 2
  run_decode_host docker exec "${DECODE_CONTAINER}" ray stop --force \
    >/dev/null 2>&1 || true
  run_decode_host docker rm -f "${DECODE_CONTAINER}" >/dev/null
}

verify_tpu_released() {
  local host=$1
  local check_script
  # This script is intentionally expanded only by bash on the target host.
  # shellcheck disable=SC2016
  check_script='
if ! command -v fuser >/dev/null 2>&1; then
  echo "fuser is required to verify TPU cleanup on host $1." >&2
  exit 1
fi
fuser_cmd=(fuser)
rm_cmd=(rm -f)
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  fuser_cmd=(sudo -n fuser)
  rm_cmd=(sudo -n rm -f)
fi
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  busy=0
  for device in /dev/accel* /dev/vfio/[0-9]*; do
    [[ -e "$device" ]] || continue
    "${fuser_cmd[@]}" "$device" >/dev/null 2>&1 && busy=1
  done
  if (( busy == 0 )); then
    if ! "${rm_cmd[@]}" /tmp/libtpu_lockfile; then
      echo "Failed to remove /tmp/libtpu_lockfile on host $1." >&2
      exit 1
    fi
    exit 0
  fi
  sleep 2
done
echo "TPU devices are still busy on host $1." >&2
exit 1
'
  if [[ "${host}" == "${HEAD_INTERNAL_IP}" ]]; then
    bash -c "${check_script}" _ "${host}"
  else
    run_decode_host bash -c "${check_script}" _ "${host}"
  fi
}

cleanup() {
  local exit_code=$1
  local cleanup_failed=0

  echo "--- Cleaning up two-host disaggregated serving"
  if run_decode_host docker exec "${DECODE_CONTAINER}" \
    cat /root/logs/decode.txt >"${LOG_DIR}/decode.txt" 2>/dev/null; then
    :
  fi

  stop_local_container "${PROXY_CONTAINER}" || cleanup_failed=1
  stop_local_container "${PREFILL_CONTAINER}" || cleanup_failed=1
  stop_decode_container || cleanup_failed=1
  verify_tpu_released "${HEAD_INTERNAL_IP}" || cleanup_failed=1
  verify_tpu_released "${DECODE_HOST_IP}" || cleanup_failed=1
  print_logs

  if (( cleanup_failed != 0 )); then
    echo "Cleanup did not fully release both TPU hosts." >&2
    return 1
  fi
  return "${exit_code}"
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  cleanup "${exit_code}" || exit 1
  exit "${exit_code}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PROJECT="$(gcloud config get-value project)"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference/vllm-tpu"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../scripts/setup_docker_env.sh"
setup_environment "${IMAGE_NAME}" "true"
DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

stop_local_container "${PROXY_CONTAINER}"
stop_local_container "${PREFILL_CONTAINER}"
stop_decode_container
verify_tpu_released "${HEAD_INTERNAL_IP}"
verify_tpu_released "${DECODE_HOST_IP}"

run_decode_host gcloud auth configure-docker \
  us-central1-docker.pkg.dev --quiet
run_decode_host docker pull "${DOCKER_IMAGE}"

COMMON_ENV=(
  -e TPU_MULTIHOST_BACKEND=ray
  -e TPU_NODE_ID=0
  -e RAY_DEDUP_LOGS=0
  -e SKIP_JAX_PRECOMPILE=1
  -e "JAX_PLATFORMS="
  -e TPU_BACKEND_TYPE=jax
  -e MODEL_IMPL_TYPE=vllm
  -e "TPU_CHIPS_PER_PROCESS_BOUNDS=1,4,1"
  -e "TPU_PROCESS_BOUNDS=1,1,1"
  -e "TPU_VISIBLE_CHIPS=0,1,2,3"
  -e CLOUD_TPU_TASK_ID=0
  -e JAX_PROCESS_ID=0
  -e JAX_NUM_PROCESSES=1
  -e HF_HOME=/root/hf
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -e TPU_VERSION=tpu7x
  -e "VLLM_DISABLE_SHARED_EXPERTS_STREAM=${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
  -e "NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN:-0}"
  -e "MOE_REQUANTIZE_BLOCK_SIZE=${MOE_REQUANTIZE_BLOCK_SIZE:-}"
  -e "MOE_REQUANTIZE_WEIGHT_DTYPE=${MOE_REQUANTIZE_WEIGHT_DTYPE:-}"
  -e "MOE_ALL_GATHER_ACTIVATION_DTYPE=${MOE_ALL_GATHER_ACTIVATION_DTYPE:-}"
  -e "FORCE_MOE_RANDOM_ROUTING=${FORCE_MOE_RANDOM_ROUTING:-}"
)

echo "--- Starting Prefill Ray head on ${HEAD_INTERNAL_IP}"
docker run -d \
  --privileged \
  --network host \
  --shm-size 16G \
  --name "${PREFILL_CONTAINER}" \
  "${COMMON_ENV[@]}" \
  -e TPU_KV_TRANSFER_PORT=8200 \
  -e TPU_SIDE_CHANNEL_PORT=8900 \
  -v "${HOST_HF_HOME}:/root/hf" \
  -v "${LOG_DIR}:/root/logs" \
  --entrypoint /bin/bash \
  "${DOCKER_IMAGE}" -c "ray start --block --head --port=8100"

echo "--- Starting Decode Ray head on ${DECODE_HOST_IP}"
run_decode_host docker run -d \
  --privileged \
  --network host \
  --shm-size 16G \
  --name "${DECODE_CONTAINER}" \
  "${COMMON_ENV[@]}" \
  -e TPU_KV_TRANSFER_PORT=9200 \
  -e TPU_SIDE_CHANNEL_PORT=8900 \
  -v "${HOST_HF_HOME}:/root/hf" \
  -v "${DECODE_LOG_DIR}:/root/logs" \
  --entrypoint /bin/bash \
  "${DOCKER_IMAGE}" -c "ray start --block --head --port=9100"

wait_for_ray() {
  local host=$1
  local container=$2
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if [[ "${host}" == "${HEAD_INTERNAL_IP}" ]]; then
      docker exec "${container}" ray status >/dev/null 2>&1 && return
    else
      run_decode_host docker exec "${container}" ray status \
        >/dev/null 2>&1 && return
    fi
    sleep 2
  done
  echo "Ray did not become ready on ${host}." >&2
  return 1
}

wait_for_ray "${HEAD_INTERNAL_IP}" "${PREFILL_CONTAINER}"
wait_for_ray "${DECODE_HOST_IP}" "${DECODE_CONTAINER}"

printf -v quoted_model '%q' "${MODEL}"
PREFILL_SERVE_CMD="vllm serve ${quoted_model} \
  --port 8400 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}' \
  > /root/logs/prefill.txt 2>&1"
DECODE_SERVE_CMD="vllm serve ${quoted_model} \
  --port 9400 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_consumer\"}' \
  > /root/logs/decode.txt 2>&1"

docker exec -d "${PREFILL_CONTAINER}" bash -c "${PREFILL_SERVE_CMD}"
run_decode_host docker exec -d "${DECODE_CONTAINER}" \
  bash -c "${DECODE_SERVE_CMD}"

wait_for_server() {
  local host=$1
  local port=$2
  local container=$3
  local log_path=$4
  local process_pattern=$5
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -fs "http://${host}:${port}/health" >/dev/null; then
      echo "Server is healthy on ${host}:${port}."
      return
    fi
    if [[ "${host}" == "${HEAD_INTERNAL_IP}" ]]; then
      docker exec "${container}" pgrep -f "${process_pattern}" >/dev/null 2>&1 ||
        break
    else
      run_decode_host docker exec "${container}" \
        pgrep -f "${process_pattern}" >/dev/null 2>&1 || break
    fi
    sleep 2
  done
  echo "Server failed to become healthy on ${host}:${port}." >&2
  if [[ "${host}" == "${HEAD_INTERNAL_IP}" ]]; then
    docker exec "${container}" cat "${log_path}" || true
  else
    run_decode_host docker exec "${container}" cat "${log_path}" || true
  fi
  return 1
}

wait_for_server "${HEAD_INTERNAL_IP}" 8400 \
  "${PREFILL_CONTAINER}" /root/logs/prefill.txt "[v]llm serve"
wait_for_server "${DECODE_HOST_IP}" 9400 \
  "${DECODE_CONTAINER}" /root/logs/decode.txt "[v]llm serve"

docker run -d \
  --network host \
  --shm-size 16G \
  --name "${PROXY_CONTAINER}" \
  -e HF_HOME=/root/hf \
  -v "${HOST_HF_HOME}:/root/hf" \
  -v "${LOG_DIR}:/root/logs" \
  --entrypoint /bin/bash \
  "${DOCKER_IMAGE}" -c "tail -f /dev/null"

docker exec -d "${PROXY_CONTAINER}" bash -c \
  "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports 8400 \
    --decoder-hosts '${DECODE_HOST_IP}' \
    --decoder-ports 9400 \
    > /root/logs/proxy.txt 2>&1"
wait_for_server "${HEAD_INTERNAL_IP}" 8000 \
  "${PROXY_CONTAINER}" /root/logs/proxy.txt "[t]oy_proxy_server"

if [[ "${TEST_MODE}" == "1" || "${TEST_MODE}" == "3" ]]; then
  timeout "${BENCHMARK_TIMEOUT_SECONDS:-1800}" \
    docker exec "${PROXY_CONTAINER}" bash -c \
    "vllm bench serve \
      --backend vllm \
      --host 127.0.0.1 \
      --port 8000 \
      --model ${quoted_model} \
      --dataset-name random \
      --random-input-len '${INPUT_LEN}' \
      --random-output-len '${OUTPUT_LEN}' \
      --num-prompts '${NUM_PROMPTS}' \
      --request-rate inf \
      --max-concurrency '${MAX_CONCURRENCY}' \
      --trust-remote-code \
      --seed '${RANDOM_SEED}' \
      > /root/logs/benchmark.txt 2>&1"

  failed_requests="$(
    awk '/Failed requests:/ {print $3}' "${LOG_DIR}/benchmark.txt" | tail -1
  )"
  if [[ -z "${failed_requests}" || "${failed_requests}" -gt 0 ]]; then
    echo "Benchmark reported failed requests: ${failed_requests:-unknown}" >&2
    exit 1
  fi
fi

if [[ "${TEST_MODE}" == "2" || "${TEST_MODE}" == "3" ]]; then
  timeout "${CORRECTNESS_TIMEOUT_SECONDS:-1800}" \
    docker exec "${PROXY_CONTAINER}" bash -c \
    "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py \
      --baseline_url 'http://${DECODE_HOST_IP}:9400/v1/completions' \
      --disagg_url http://127.0.0.1:8000/v1/completions \
      --model ${quoted_model} \
      --num_requests '${NUM_PROMPTS}' \
      --input_length '${INPUT_LEN}' \
      --output_length '${OUTPUT_LEN}' \
      > /root/logs/correctness.txt 2>&1"
fi
