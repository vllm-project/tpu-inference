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

# E2E coverage for n-gram speculative decoding on one multi-host TPU v7x-16
# slice. Both TPU hosts join one Ray cluster; a single vLLM server on the Ray
# head owns all 16 TPU cores. There is intentionally no Prefill/Decode split,
# KV transfer connector, or request proxy in this test.
set -euo pipefail

readonly CHIPS_PER_HOST=4
readonly CORES_PER_CHIP=2
readonly HOSTS_PER_V7X16=2
readonly TENSOR_PARALLEL_SIZE=$((CHIPS_PER_HOST * CORES_PER_CHIP * HOSTS_PER_V7X16))

SSH_USER="${SSH_USER:-$(whoami)}"
HOST_HF_HOME="${HOST_HF_HOME:-/tmp/hf_home}"
LOG_DIR="${LOG_DIR:-${HOME}/logs}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RANDOM_SEED="${RANDOM_SEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-node}"
DEFAULT_SPECULATIVE_CONFIG='{"method":"ngram","prompt_lookup_max":5,"prompt_lookup_min":3,"num_speculative_tokens":3}'
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-${DEFAULT_SPECULATIVE_CONFIG}}"
export TPU_VERSION="${TPU_VERSION:-tpu7x}"

if ! python3 -c 'import json, sys; json.loads(sys.argv[1])' "${SPECULATIVE_CONFIG}"; then
  echo "ERROR: SPECULATIVE_CONFIG must be valid JSON: ${SPECULATIVE_CONFIG}" >&2
  exit 2
fi
SPECULATIVE_METHOD="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1]).get("method", ""))' "${SPECULATIVE_CONFIG}")"
if [[ "${SPECULATIVE_METHOD}" != "ngram" ]]; then
  echo "ERROR: This test requires SPECULATIVE_CONFIG.method=ngram." >&2
  exit 2
fi
if [[ "${TPU_VERSION}" != "tpu7x" ]]; then
  echo "ERROR: This test requires TPU_VERSION=tpu7x." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TOP_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
CLUSTER_SCRIPT="${TOP_DIR}/scripts/multihost/run_cluster.sh"
mkdir -p "${LOG_DIR}" "${HOST_HF_HOME}"
rm -f "${LOG_DIR}/vllm_serve.log" "${LOG_DIR}/benchmark.txt"

metadata() {
  curl -fs -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null || true
}

discover_cluster() {
  if [[ -n "${WORKER_IPS:-}" ]]; then
    HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(metadata instance/network-interfaces/0/ip)}"
  else
    local zone tpu_name endpoints
    zone="${ZONE:-$(metadata instance/zone | awk -F/ '{print $NF}')}"
    tpu_name="${TPU_NAME:-$(metadata instance/description)}"
    [[ -n "${zone}" && -n "${tpu_name}" ]] || {
      echo "ERROR: Could not discover TPU_NAME or ZONE; set HEAD_INTERNAL_IP and WORKER_IPS." >&2
      exit 1
    }
    endpoints="$(gcloud compute tpus tpu-vm describe "${tpu_name}" --zone "${zone}" \
      --format='value(networkEndpoints[].ipAddress)')"
    endpoints="${endpoints//;/ }"
    endpoints="${endpoints//,/ }"
    # shellcheck disable=SC2206
    local ips=(${endpoints})
    if (( ${#ips[@]} != HOSTS_PER_V7X16 )); then
      echo "ERROR: Expected exactly ${HOSTS_PER_V7X16} hosts for a TPU v7x-16 slice; found ${#ips[@]}." >&2
      exit 1
    fi
    HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-${ips[0]}}"
    WORKER_IPS="${ips[1]}"
  fi

  [[ -n "${HEAD_INTERNAL_IP:-}" && -n "${WORKER_IPS:-}" ]] || {
    echo "ERROR: HEAD_INTERNAL_IP and one WORKER_IPS entry are required." >&2
    exit 1
  }
  IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
  if (( ${#worker_ips[@]} != HOSTS_PER_V7X16 - 1 )) || [[ -z "${worker_ips[0]}" ]]; then
    echo "ERROR: TPU v7x-16 requires exactly one Ray worker host; got WORKER_IPS=${WORKER_IPS}." >&2
    exit 1
  fi
}

validate_tpu_slice() {
  local zone tpu_name accelerator_type
  zone="${ZONE:-$(metadata instance/zone | awk -F/ '{print $NF}')}"
  tpu_name="${TPU_NAME:-$(metadata instance/description)}"
  [[ -n "${zone}" && -n "${tpu_name}" ]] || return 0
  accelerator_type="$(gcloud compute tpus tpu-vm describe "${tpu_name}" --zone "${zone}" \
    --format='value(acceleratorType)')"
  if [[ "${accelerator_type}" != *"v7x-16"* && "${accelerator_type}" != *"tpu7x-16"* ]]; then
    echo "ERROR: Expected a TPU v7x-16 slice; got '${accelerator_type:-unknown}'." >&2
    exit 1
  fi
}

discover_cluster
validate_tpu_slice

if [[ ! -f "${HOME}/.ssh/id_rsa" ]]; then
  mkdir -p "${HOME}/.ssh"
  ssh-keygen -t rsa -b 4096 -N '' -f "${HOME}/.ssh/id_rsa" -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i "${HOME}/.ssh/id_rsa")
CLUSTER_LAUNCH_PIDS=()

start_launcher() {
  setsid "$@" &
  CLUSTER_LAUNCH_PIDS+=("$!")
}

stop_launcher() {
  local pid=$1
  kill -0 "${pid}" 2>/dev/null || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || true
  for _ in {1..10}; do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
    sleep 1
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local worker_ip pid
  docker cp "${CONTAINER_NAME}:/root/vllm_serve.log" "${LOG_DIR}/vllm_serve.log" 2>/dev/null || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
  for worker_ip in "${worker_ips[@]}"; do
    [[ -n "${worker_ip}" ]] || continue
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true" || true
  done
  for pid in "${CLUSTER_LAUNCH_PIDS[@]}"; do
    stop_launcher "${pid}"
  done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local end_time=$((SECONDS + ${SERVER_STARTUP_TIMEOUT_SECONDS:-3600}))
  while (( SECONDS < end_time )); do
    if curl -fs "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null; then return 0; fi
    if ! docker exec "${CONTAINER_NAME}" pgrep -f '[v]llm serve' >/dev/null 2>&1; then
      docker exec "${CONTAINER_NAME}" cat /root/vllm_serve.log 2>&1 || true
      return 1
    fi
    sleep 5
  done
  docker exec "${CONTAINER_NAME}" cat /root/vllm_serve.log 2>&1 || true
  return 1
}

assert_ngram_draft_tokens() {
  local metrics
  metrics="$(curl -fsS "http://127.0.0.1:${VLLM_PORT}/metrics")"
  echo "--- N-gram speculative-decoding metrics ---"
  printf '%s\n' "${metrics}" | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens(_total)?(\{| )' || true
  if ! printf '%s\n' "${metrics}" | awk '
    $1 ~ /^vllm:spec_decode_num_draft_tokens(_total)?(\{|$)/ { total += $NF }
    END { exit !(total > 0) }
  '; then
    echo "ERROR: n-gram speculative decoding produced no draft tokens." >&2
    return 1
  fi
}

run_speculative_probe() {
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
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
    -H 'Content-Type: application/json' --data "${request_body}" >/dev/null
  assert_ngram_draft_tokens
}

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "${DOCKER_IMAGE:-}" ]]; then
  IMAGE_NAME="${IMAGE_NAME:-us-central1-docker.pkg.dev/${PROJECT}/tpu-inference/vllm-tpu}"
  # shellcheck disable=SC1091
  source "${TOP_DIR}/.buildkite/scripts/setup_docker_env.sh"
  setup_environment "${IMAGE_NAME}" true
  DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"
fi
echo "Using Docker image: ${DOCKER_IMAGE}"

COMMON_ENV=(
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e TPU_MULTIHOST_BACKEND=ray
  -e JAX_PLATFORMS=''
  -e TPU_BACKEND_TYPE=jax
  -e MODEL_IMPL_TYPE=vllm
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
  -e "SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE}"
  -e "TPU_VERSION=${TPU_VERSION}"
  -e RUN_CLUSTER_CLEANUP_OWNER=parent
)

cleanup
echo "--- Starting one multi-host Ray cluster: head=${HEAD_INTERNAL_IP}, worker=${WORKER_IPS}"
start_launcher bash "${CLUSTER_SCRIPT}" "${DOCKER_IMAGE}" "${HEAD_INTERNAL_IP}" --head "${HOST_HF_HOME}" "${COMMON_ENV[@]}"
sleep "${HEAD_STARTUP_WAIT_SECONDS:-60}"

IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
for worker_ip in "${worker_ips[@]}"; do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "mkdir -p ~/tpu-inference/scripts/multihost $(printf '%q' "${HOST_HF_HOME}")"
  base64 < "${CLUSTER_SCRIPT}" | ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
  printf -v remote_common_env '%q ' "${COMMON_ENV[@]}"
  start_launcher ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "bash ~/tpu-inference/scripts/multihost/run_cluster.sh $(printf '%q ' "${DOCKER_IMAGE}" "${HEAD_INTERNAL_IP}" --worker "${HOST_HF_HOME}")${remote_common_env}"
done
sleep "${WORKER_STARTUP_WAIT_SECONDS:-120}"

printf -v quoted_model '%q' "${MODEL}"
printf -v quoted_load_format '%q' "${LOAD_FORMAT}"
printf -v quoted_speculative_config '%q' "${SPECULATIVE_CONFIG}"
SERVE_CMD="vllm serve ${quoted_model} --port ${VLLM_PORT} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --trust-remote-code --load-format ${quoted_load_format} --no-enable-prefix-caching --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-model-len ${MAX_MODEL_LEN} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --max-num-seqs ${MAX_NUM_SEQS} --speculative-config ${quoted_speculative_config}"

echo "--- Starting one multi-host speculative vLLM server"
docker exec -d -e HF_HOME=/root/.cache/huggingface "${CONTAINER_NAME}" \
  bash -c "${SERVE_CMD} > /root/vllm_serve.log 2>&1"
wait_for_server
run_speculative_probe

echo "--- Running multi-host speculative benchmark"
timeout "${BENCHMARK_TIMEOUT_SECONDS:-1800}" docker exec "${CONTAINER_NAME}" \
  vllm bench serve --backend vllm --host 127.0.0.1 --port "${VLLM_PORT}" --model "${MODEL}" \
  --dataset-name random --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" --request-rate inf --max-concurrency "${MAX_CONCURRENCY}" \
  --trust-remote-code --seed "${RANDOM_SEED}" >"${LOG_DIR}/benchmark.txt" 2>&1
cat "${LOG_DIR}/benchmark.txt"
failed_requests="$(awk '/Failed requests:/ {print $3}' "${LOG_DIR}/benchmark.txt" | tail -1)"
[[ "${failed_requests}" =~ ^[0-9]+$ && "${failed_requests}" -eq 0 ]] || {
  echo "ERROR: Benchmark reported failed requests: ${failed_requests:-unknown}" >&2
  exit 1
}

echo "--- Tests completed successfully"
echo "Multi-host TPU v7x-16 n-gram speculative decoding passed: head=${HEAD_INTERNAL_IP}, worker=${WORKER_IPS}, model=${MODEL}, config=${SPECULATIVE_CONFIG}"
