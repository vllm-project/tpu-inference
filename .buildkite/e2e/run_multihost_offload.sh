#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# E2E coverage for one vLLM instance distributed across a multi-host TPU slice
# with host-side KV-cache offload enabled. Unlike run_multi_disagg_offload.sh,
# this script has no Prefill/Decode split or proxy. It verifies both actual
# D2H/H2D transfers and deterministic output equality against a no-offload
# baseline before running the optional serving benchmark.
set -euo pipefail

readonly CHIPS_PER_HOST=4
readonly CORES_PER_CHIP=2

SSH_USER="${SSH_USER:-$(whoami)}"
# Use a writable local cache by default. CI jobs with a persistent model disk
# can still set HOST_HF_HOME explicitly.
HOST_HF_HOME="${HOST_HF_HOME:-/tmp/hf_home}"
LOG_DIR="${LOG_DIR:-${HOME}/logs}"
# Match .buildkite/scripts/run_multihost.sh: Runai Streamer reads this
# checkpoint directly from GCS instead of first materializing it in HF_HOME.
MODEL="${MODEL:-gs://tpu-commons-ci/qwen/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-20}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RANDOM_SEED="${RANDOM_SEED:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
TEST_MODE="${TEST_MODE:-3}" # 1: benchmark, 2: correctness, 3: both
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}"
SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
TPU_ENABLE_D2H_TRANSFER="${TPU_ENABLE_D2H_TRANSFER:-true}"
TPU_MAX_HOST_KV_BUFFER_SIZE="${TPU_MAX_HOST_KV_BUFFER_SIZE:-128}"
TPU_OFFLOAD_NUM_CPU_CHUNKS="${TPU_OFFLOAD_NUM_CPU_CHUNKS:-64}"
TPU_OFFLOAD_NUM_STAGING_BLOCKS="${TPU_OFFLOAD_NUM_STAGING_BLOCKS:-16}"
TPU_OFFLOAD_METRICS_LOG_INTERVAL="${TPU_OFFLOAD_METRICS_LOG_INTERVAL:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
# scripts/multihost/run_cluster.sh names its Ray container "node".
CONTAINER_NAME="node"
export TPU_VERSION="${TPU_VERSION:-tpu7x}"

case "${TEST_MODE}" in
  1 | 2 | 3) ;;
  *) echo "ERROR: TEST_MODE must be 1 (benchmark), 2 (correctness), or 3 (both)." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TOP_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
CLUSTER_SCRIPT="${TOP_DIR}/scripts/multihost/run_cluster.sh"
mkdir -p "${LOG_DIR}" "${HOST_HF_HOME}"
rm -f "${LOG_DIR}/vllm_serve.log" "${LOG_DIR}/benchmark.txt" \
  "${LOG_DIR}/correctness.txt" "${LOG_DIR}/offload_metrics.txt" \
  "${LOG_DIR}/baseline.json" "${LOG_DIR}/offload.json"

metadata() {
  curl -fs -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null || true
}

discover_cluster() {
  if [[ -n "${WORKER_IPS:-}" ]]; then
    HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(metadata instance/network-interfaces/0/ip)}"
    return
  fi
  local zone tpu_name all_ips
  zone="${ZONE:-$(metadata instance/zone | awk -F/ '{print $NF}')}"
  tpu_name="${TPU_NAME:-$(metadata instance/description)}"
  [[ -n "${zone}" && -n "${tpu_name}" ]] || {
    echo "ERROR: Could not discover TPU_NAME or ZONE; set WORKER_IPS and HEAD_INTERNAL_IP." >&2
    exit 1
  }
  all_ips="$(gcloud compute tpus tpu-vm describe "${tpu_name}" --zone "${zone}" \
    --format='value(networkEndpoints[].ipAddress)')"
  all_ips="${all_ips//;/ }"
  all_ips="${all_ips//,/ }"
  # shellcheck disable=SC2206
  local ips=(${all_ips})
  (( ${#ips[@]} > 1 )) || {
    echo "ERROR: Expected a multi-host TPU slice, found ${#ips[@]} host(s)." >&2
    exit 1
  }
  HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-${ips[0]}}"
  WORKER_IPS="$(IFS=,; echo "${ips[*]:1}")"
}

discover_cluster
[[ -n "${HEAD_INTERNAL_IP:-}" && -n "${WORKER_IPS:-}" ]] || {
  echo "ERROR: HEAD_INTERNAL_IP and at least one WORKER_IPS entry are required." >&2
  exit 1
}
IFS=',' read -r -a discovered_worker_ips <<< "${WORKER_IPS}"
if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
  TENSOR_PARALLEL_SIZE=$(( (1 + ${#discovered_worker_ips[@]}) * CHIPS_PER_HOST * CORES_PER_CHIP ))
fi
[[ "${TPU_VERSION}" == "tpu7x" ]] || {
  echo "ERROR: This test requires TPU_VERSION=tpu7x." >&2
  exit 1
}

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
  local worker_ip pid remote_check_status
  docker cp "${CONTAINER_NAME}:/root/vllm_serve.log" "${LOG_DIR}/vllm_serve.log" 2>/dev/null || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "WARNING: ${CONTAINER_NAME} still exists on the head host after cleanup." >&2
  fi
  IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
  for worker_ip in "${worker_ips[@]}"; do
    [[ -n "${worker_ip}" ]] || continue
    if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true"; then
      echo "WARNING: Could not request cleanup of ${CONTAINER_NAME} on worker ${worker_ip}." >&2
    fi
    if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "if docker inspect '${CONTAINER_NAME}' >/dev/null 2>&1; then exit 10; fi"; then
      :
    else
      remote_check_status=$?
      if (( remote_check_status == 10 )); then
        echo "WARNING: ${CONTAINER_NAME} still exists on worker ${worker_ip} after cleanup." >&2
      else
        echo "WARNING: Could not verify cleanup of ${CONTAINER_NAME} on worker ${worker_ip}." >&2
      fi
    fi
  done
  for pid in "${CLUSTER_LAUNCH_PIDS[@]}"; do stop_launcher "${pid}"; done
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

stream_ray_vllm_logs() {
  # vLLM workers run as Ray processes, so their stdout/stderr is written under
  # the Ray session directory rather than to `docker logs`. Track each file's
  # line offset to stream only newly emitted output to Buildkite.
  docker exec "${CONTAINER_NAME}" bash -c '
    set -u
    log_dir=/tmp/ray/session_latest/logs
    declare -A emitted_lines=()
    while true; do
      while IFS= read -r -d "" log_file; do
        line_count=$(wc -l < "${log_file}")
        start_line=${emitted_lines["${log_file}"]:-1}
        if (( line_count >= start_line )); then
          echo "--- Ray/vLLM log: ${log_file} ---"
          sed -n "${start_line},${line_count}p" "${log_file}"
        fi
        emitted_lines["${log_file}"]=$((line_count + 1))
      done < <(find "${log_dir}" -maxdepth 1 -type f \
        \( -name "*worker*.out" -o -name "*worker*.err" -o -name "*driver*.log" \) -print0 2>/dev/null)
      sleep 1
    done
  '
}

# The verification helper first generates with no connector, then uses the
# TPUOffloadConnector twice with a prefix-cache reset in between. It compares
# generated text bit-for-bit. The reset makes the second
# connector run load saved KV blocks from host memory rather than relying on
# HBM-resident prefix cache.
run_offload_correctness() {
  local verification_pid correctness_tail_pid ray_log_pid
  local verification_status=0 elapsed_seconds=0

  echo "--- Verifying deterministic output with actual TPU KV-cache offload"
  : >"${LOG_DIR}/correctness.txt"
  timeout "${CORRECTNESS_TIMEOUT_SECONDS:-3600}" \
    docker exec "${CONTAINER_NAME}" bash -c \
    "python3 /workspace/tpu_inference/.buildkite/e2e/verify_multihost_offload.py \\
      --model $(printf '%q' "${MODEL}") \\
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \\
      --load-format ${LOAD_FORMAT} \\
      --max-model-len ${MAX_MODEL_LEN} \\
      --max-tokens ${OUTPUT_LEN} \\
      --seed ${RANDOM_SEED} \\
      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
      --baseline-output /tmp/multihost_offload_verification/baseline.json \\
      --offload-output /tmp/multihost_offload_verification/offload.json" \
    >"${LOG_DIR}/correctness.txt" 2>&1 &
  verification_pid=$!

  tail -n 0 -F "${LOG_DIR}/correctness.txt" &
  correctness_tail_pid=$!
  stream_ray_vllm_logs &
  ray_log_pid=$!

  while kill -0 "${verification_pid}" 2>/dev/null; do
    sleep 5
    elapsed_seconds=$((elapsed_seconds + 5))
    echo "--- Offload correctness is still running (${elapsed_seconds}s elapsed)"
    docker exec "${CONTAINER_NAME}" ray status --address=auto 2>&1 | sed -n '1,12p' || true
  done

  if wait "${verification_pid}"; then
    :
  else
    verification_status=$?
  fi
  kill "${correctness_tail_pid}" "${ray_log_pid}" 2>/dev/null || true
  wait "${correctness_tail_pid}" 2>/dev/null || true
  wait "${ray_log_pid}" 2>/dev/null || true

  if (( verification_status != 0 )); then
    echo "ERROR: Offload correctness verification failed; showing ${LOG_DIR}/correctness.txt" >&2
    cat "${LOG_DIR}/correctness.txt" >&2 || true
    return "${verification_status}"
  fi
  docker cp "${CONTAINER_NAME}:/tmp/multihost_offload_verification/baseline.json" \
    "${LOG_DIR}/baseline.json" 2>/dev/null || true
  docker cp "${CONTAINER_NAME}:/tmp/multihost_offload_verification/offload.json" \
    "${LOG_DIR}/offload.json" 2>/dev/null || true
  cat "${LOG_DIR}/correctness.txt"
}

collect_offload_metrics() {
  local worker_ip
  : > "${LOG_DIR}/offload_metrics.txt"
  docker exec "${CONTAINER_NAME}" bash -c \
    'grep -R -h "Offload Metrics Snapshot:" /tmp/ray/session_latest/logs 2>/dev/null || true' \
    >>"${LOG_DIR}/offload_metrics.txt"
  IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
  for worker_ip in "${worker_ips[@]}"; do
    [[ -n "${worker_ip}" ]] || continue
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "docker exec '${CONTAINER_NAME}' bash -c 'grep -R -h \"Offload Metrics Snapshot:\" /tmp/ray/session_latest/logs 2>/dev/null || true'" \
      >>"${LOG_DIR}/offload_metrics.txt" || true
  done
  cat "${LOG_DIR}/offload_metrics.txt"
}

assert_actual_offload() {
  # The stats logger clears interval counters, so D2H and H2D can appear in
  # separate snapshots. Require a positive operation count for each.
  sleep "$((TPU_OFFLOAD_METRICS_LOG_INTERVAL + 1))"
  collect_offload_metrics
  grep -Eq 'Offload Metrics Snapshot:.*d2h=[1-9][0-9]*' "${LOG_DIR}/offload_metrics.txt" || {
    echo "ERROR: No D2H offload operation was observed." >&2
    return 1
  }
  grep -Eq 'Offload Metrics Snapshot:.*h2d=[1-9][0-9]*' "${LOG_DIR}/offload_metrics.txt" || {
    echo "ERROR: No H2D reload operation was observed after the cache reset." >&2
    return 1
  }
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
  -e "SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE}"
  -e "TPU_ENABLE_D2H_TRANSFER=${TPU_ENABLE_D2H_TRANSFER}"
  -e "TPU_MAX_HOST_KV_BUFFER_SIZE=${TPU_MAX_HOST_KV_BUFFER_SIZE}"
  -e "TPU_OFFLOAD_NUM_CPU_CHUNKS=${TPU_OFFLOAD_NUM_CPU_CHUNKS}"
  -e "TPU_OFFLOAD_NUM_STAGING_BLOCKS=${TPU_OFFLOAD_NUM_STAGING_BLOCKS}"
  -e "TPU_OFFLOAD_METRICS_LOG_INTERVAL=${TPU_OFFLOAD_METRICS_LOG_INTERVAL}"
)

cleanup
echo "--- Starting multi-host Ray cluster: head=${HEAD_INTERNAL_IP}, workers=${WORKER_IPS}"
start_launcher bash "${CLUSTER_SCRIPT}" "${DOCKER_IMAGE}" "${HEAD_INTERNAL_IP}" --head "${HOST_HF_HOME}" "${COMMON_ENV[@]}"
sleep "${HEAD_STARTUP_WAIT_SECONDS:-60}"

IFS=',' read -r -a worker_ips <<< "${WORKER_IPS}"
for worker_ip in "${worker_ips[@]}"; do
  [[ -n "${worker_ip}" ]] || continue
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "mkdir -p ~/tpu-inference/scripts/multihost $(printf '%q' "${HOST_HF_HOME}")"
  base64 < "${CLUSTER_SCRIPT}" | \
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
  printf -v remote_common_env '%q ' "${COMMON_ENV[@]}"
  start_launcher ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "bash ~/tpu-inference/scripts/multihost/run_cluster.sh $(printf '%q ' "${DOCKER_IMAGE}" "${HEAD_INTERNAL_IP}" --worker "${HOST_HF_HOME}")${remote_common_env}"
done
sleep "${WORKER_STARTUP_WAIT_SECONDS:-120}"

run_offload_correctness
assert_actual_offload

printf -v quoted_model '%q' "${MODEL}"
printf -v quoted_load_format '%q' "${LOAD_FORMAT}"
KV_TRANSFER_CONFIG='{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector","kv_role":"kv_both"}'
SERVE_CMD="vllm serve ${quoted_model} --port ${VLLM_PORT} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --trust-remote-code --load-format ${quoted_load_format} --enable-prefix-caching --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-model-len ${MAX_MODEL_LEN} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --max-num-seqs ${MAX_NUM_SEQS} --kv-transfer-config '${KV_TRANSFER_CONFIG}'"
echo "--- Starting one multi-host vLLM server with host KV-cache offload enabled"
docker exec -d -e HF_HOME=/root/.cache/huggingface "${CONTAINER_NAME}" \
  bash -c "${SERVE_CMD} > /root/vllm_serve.log 2>&1"
wait_for_server

if [[ "${TEST_MODE}" == 1 || "${TEST_MODE}" == 3 ]]; then
  echo "--- Running multi-host host-KV-offload benchmark"
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
fi

echo "--- Tests completed successfully"
echo "Multi-host vLLM with host KV-cache offload passed: head=${HEAD_INTERNAL_IP}, workers=${WORKER_IPS}, model=${MODEL}"
