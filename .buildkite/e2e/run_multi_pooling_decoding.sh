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

# Standalone multi-host E2E correctness coverage for step pooling and
# structured decoding.
set -euo pipefail
set -x

export SSH_USER="${SSH_USER:-$(whoami)}"
HOST_HF_HOME="${HOST_HF_HOME:-/tmp/hf_home}"
POOLING_MODEL="${POOLING_MODEL:-Qwen/Qwen3-Embedding-8B}"
GENERATION_MODEL="${GENERATION_MODEL:-gs://tpu-commons-ci/qwen/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-16}"
VLLM_PORT=8000

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BUILDKITE_DIR=$(dirname "$SCRIPT_DIR")
TOP_DIR=$(dirname "$BUILDKITE_DIR")

discover_workers() {
  if [[ -n "${WORKER_IPS:-}" ]]; then
    return
  fi
  command -v gcloud &>/dev/null || {
    echo "gcloud is required when WORKER_IPS is not set."
    exit 1
  }

  local zone="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
  local tpu_name="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description")}"
  [[ -n "$zone" && -n "$tpu_name" ]] || {
    echo "Unable to discover TPU_NAME or ZONE; set WORKER_IPS explicitly."
    exit 1
  }

  local all_ips
  all_ips=$(gcloud compute tpus tpu-vm describe "$tpu_name" --zone "$zone" --format="value(networkEndpoints[].ipAddress)")
  all_ips="${all_ips//;/ }"
  all_ips="${all_ips//,/ }"
  # shellcheck disable=SC2206
  local ips=($all_ips)
  [[ ${#ips[@]} -gt 1 ]] || {
    echo "Expected a multi-host TPU slice, but discovered ${#ips[@]} host(s)."
    exit 1
  }
  HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-${ips[0]}}"
  WORKER_IPS=$(IFS=,; echo "${ips[*]:1}")
}

discover_workers
HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(hostname -I | awk '{print $1}')}"
[[ -n "${WORKER_IPS:-}" ]] || {
  echo "WORKER_IPS must contain at least one worker."
  exit 1
}
if [[ "${TPU_VERSION:-tpu7x}" != "tpu7x" ]]; then
  echo "This test requires TPU_VERSION=tpu7x."
  exit 1
fi

if [[ ! -f ~/.ssh/id_rsa ]]; then
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i ~/.ssh/id_rsa)
CLUSTER_LAUNCH_PIDS=()

# run_cluster.sh blocks for the lifetime of its Ray node.  Start each launcher
# in its own session so cleanup can stop its whole process group (including an
# SSH child) instead of leaving a detached remote command behind.
start_cluster_launcher() {
  setsid "$@" &
  CLUSTER_LAUNCH_PIDS+=("$!")
}

stop_cluster_launcher() {
  local pid=$1

  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true

  # A remote SSH command can keep its parent alive after SIGTERM.  Escalate
  # after a short grace period so CI cleanup cannot consume the step timeout.
  for _ in {1..10}; do
    kill -0 "$pid" 2>/dev/null || {
      wait "$pid" 2>/dev/null || true
      return 0
    }
    sleep 1
  done

  echo "Launcher process group ${pid} did not stop after SIGTERM; sending SIGKILL."
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local worker_ip
  IFS=',' read -r -a worker_ips <<< "${WORKER_IPS:-}"
  for worker_ip in "${worker_ips[@]}"; do
    [[ -n "$worker_ip" ]] || continue
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
  done

  docker cp node:/root/vllm_serve.log /tmp/vllm_serve.log >/dev/null 2>&1 || true
  if [[ -f /tmp/vllm_serve.log ]]; then
    echo "==================== START OF VLLM SERVE LOG ===================="
    cat /tmp/vllm_serve.log || true
    echo "==================== END OF VLLM SERVE LOG ===================="
  fi
  docker stop node >/dev/null 2>&1 || true
  docker rm -f node >/dev/null 2>&1 || true

  # Do this last: stopping the containers normally lets the launchers return,
  # while the bounded process-group cleanup covers a stuck SSH/log stream.
  local pid
  for pid in "${CLUSTER_LAUNCH_PIDS[@]}"; do
    stop_cluster_launcher "$pid"
  done
}
trap cleanup EXIT

wait_for_server() {
  local pid=""
  for _ in {1..10}; do
    pid=$(docker exec node pgrep -n -f "vllm serve" || true)
    [[ -n "$pid" ]] && break
    sleep 1
  done
  if [[ -z "$pid" ]]; then
    docker exec node cat /root/vllm_serve.log || true
    return 1
  fi

  local end_time=$((SECONDS + 7200))
  while [[ $SECONDS -lt $end_time ]]; do
    if curl -fs "localhost:${VLLM_PORT}/health" >/dev/null; then
      return 0
    fi
    if ! docker exec node kill -0 "$pid" 2>/dev/null; then
      docker exec node cat /root/vllm_serve.log || true
      return 1
    fi
    sleep 5
  done
  docker exec node cat /root/vllm_serve.log || true
  return 1
}

stop_server() {
  docker exec node pkill -f "vllm serve" || true
  for _ in {1..30}; do
    docker exec node pgrep -f "vllm serve" >/dev/null 2>&1 || return
    sleep 1
  done
  echo "vLLM server did not stop cleanly."
  return 1
}

run_case() {
  local name=$1
  local serve_command=$2
  local validation_command=$3

  echo "--- Starting ${name} server"
  docker exec -d -e HF_HOME=/root/.cache/huggingface node \
    bash -c "${serve_command} > /root/vllm_serve.log 2>&1"
  wait_for_server

  echo "--- Validating ${name}"
  docker exec -e HF_HOME=/root/.cache/huggingface node bash -c "$validation_command"
  stop_server
}

PROJECT=$(gcloud config get-value project)
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"

docker system prune -a --volumes -f || true
# shellcheck disable=SC1091
source "${BUILDKITE_DIR}/scripts/setup_docker_env.sh"
setup_environment "${IMAGE_NAME}" true
DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

cleanup
start_cluster_launcher bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" "${HEAD_INTERNAL_IP}" --head "${HOST_HF_HOME}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TPU_MULTIHOST_BACKEND=ray \
  -e JAX_PLATFORMS='' \
  -e TPU_BACKEND_TYPE=jax \
  -e MODEL_IMPL_TYPE=vllm
sleep 60

IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
for worker_ip in "${worker_ips[@]}"; do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f" || true
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "mkdir -p ~/tpu-inference/scripts/multihost"
  base64 < "${TOP_DIR}/scripts/multihost/run_cluster.sh" | \
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
  # shellcheck disable=SC2029
  start_cluster_launcher ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "bash ~/tpu-inference/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${HEAD_INTERNAL_IP}' --worker '${HOST_HF_HOME}' -e HF_TOKEN='${HF_TOKEN:-}' -e TPU_MULTIHOST_BACKEND=ray -e JAX_PLATFORMS='' -e TPU_BACKEND_TYPE=jax -e MODEL_IMPL_TYPE=vllm"
done
sleep 120

# This input is larger than the chunked server's max-num-batched-tokens. The
# resulting embedding is compared to an unchunked reference generated with
# the same model and input below.
POOLING_INPUT=$(python3 -c 'print("TPU step pooling validates chunked prefill. " * 180)')
POOLING_REQUEST_BODY=$(POOLING_MODEL="$POOLING_MODEL" POOLING_INPUT="$POOLING_INPUT" \
  python3 -c 'import json, os; print(json.dumps({"model": os.environ["POOLING_MODEL"], "input": os.environ["POOLING_INPUT"]}))')
POOLING_CHUNKED_VALIDATION="curl --fail --silent --show-error http://localhost:${VLLM_PORT}/v1/embeddings -H 'Content-Type: application/json' -d '${POOLING_REQUEST_BODY}' > /tmp/pooling_chunked.json; python3 -c 'import json; payload = json.load(open(\"/tmp/pooling_chunked.json\")); embedding = payload[\"data\"][0][\"embedding\"]; assert embedding and all(isinstance(value, (int, float)) for value in embedding), payload'"
POOLING_REFERENCE_VALIDATION="response=\$(curl --fail --silent --show-error http://localhost:${VLLM_PORT}/v1/embeddings -H 'Content-Type: application/json' -d '${POOLING_REQUEST_BODY}'); POOLING_REFERENCE=\"\$response\" python3 -c 'import json, math, os; chunked = json.load(open(\"/tmp/pooling_chunked.json\"))[\"data\"][0][\"embedding\"]; reference = json.loads(os.environ[\"POOLING_REFERENCE\"])[\"data\"][0][\"embedding\"]; assert len(chunked) == len(reference) and chunked, (len(chunked), len(reference)); assert all(math.isclose(actual, expected, rel_tol=1e-3, abs_tol=1e-3) for actual, expected in zip(chunked, reference)), \"chunked and unchunked embeddings differ\"'"

STRUCTURED_VALIDATION="response=\$(curl --fail --silent --show-error http://localhost:${VLLM_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"${GENERATION_MODEL}\",\"prompt\":\"Classify the sentiment as positive or negative. Return only the requested label.\",\"max_tokens\":16,\"temperature\":0,\"structured_outputs\":{\"choice\":[\"<TPU-CI-ALPHA>\",\"<TPU-CI-BETA>\"]}}'); python3 -c 'import json, sys; payload = json.load(sys.stdin); text = payload[\"choices\"][0][\"text\"].strip(); assert text in {\"<TPU-CI-ALPHA>\", \"<TPU-CI-BETA>\"}, payload' <<< \"\$response\""

run_case "chunked step pooling" \
  "vllm serve ${POOLING_MODEL} --port ${VLLM_PORT} --runner pooling --tensor-parallel-size ${TP_SIZE} --trust-remote-code --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 256 --no-enable-prefix-caching --load-format=auto" \
  "$POOLING_CHUNKED_VALIDATION"
run_case "unchunked pooling reference" \
  "vllm serve ${POOLING_MODEL} --port ${VLLM_PORT} --runner pooling --tensor-parallel-size ${TP_SIZE} --trust-remote-code --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching --load-format=auto" \
  "$POOLING_REFERENCE_VALIDATION"
run_case "structured decoding" \
  "vllm serve ${GENERATION_MODEL} --port ${VLLM_PORT} --tensor-parallel-size ${TP_SIZE} --trust-remote-code --max-model-len 1024 --no-async-scheduling --load-format=runai_streamer --no-enable-prefix-caching" \
  "$STRUCTURED_VALIDATION"
