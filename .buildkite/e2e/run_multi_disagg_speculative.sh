#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Multi-host P/D disaggregation plus speculative decoding.
# The default model is the larger MoE model used by repository multi-host CI.
# Speculative decoding is intentionally enabled only on the Decode server.

set -euo pipefail

export SSH_USER="${SSH_USER:-$(whoami)}"
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"
RANDOM_SEED="${RANDOM_SEED:-10}"
TEST_MODE="${TEST_MODE:-3}" # 1: benchmark, 2: correctness, 3: both
PREFILL_HOSTS_COUNT="${PREFILL_HOSTS_COUNT:-1}"
DECODE_HOSTS_COUNT="${DECODE_HOSTS_COUNT:-1}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-ngram}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"ngram\",\"prompt_lookup_max\":5,\"prompt_lookup_min\":3,\"num_speculative_tokens\":3}}"
DRAFT_MODEL_IMPL_TYPE="${DRAFT_MODEL_IMPL_TYPE:-auto}"

LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/{prefill,decode,proxy,benchmark,correctness}.txt

metadata_value() {
  curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null || true
}

get_local_ip() {
  local ip
  ip="$(metadata_value instance/network-interfaces/0/ip)"
  [[ -n "$ip" ]] && echo "$ip" || hostname -I | awk '{print $1}'
}

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_local_ip)}"
if [[ -z "${WORKER_IPS:-}" ]]; then
  command -v gcloud >/dev/null 2>&1 || { echo "WORKER_IPS is required" >&2; exit 1; }
  ZONE="${ZONE:-$(metadata_value instance/zone | awk -F/ '{print $NF}')}"
  TPU_NAME="${TPU_NAME:-$(metadata_value instance/description)}"
  [[ -n "$ZONE" && -n "$TPU_NAME" ]] || { echo "Cannot discover TPU" >&2; exit 1; }
  endpoints="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
    --format='value(networkEndpoints[].ipAddress)')"
  endpoints="${endpoints//;/ }"
  endpoints="${endpoints//,/ }"
  read -r -a endpoint_array <<< "$endpoints"
  workers=()
  found_head=0
  for ip in "${endpoint_array[@]}"; do
    if [[ "$ip" == "$HEAD_INTERNAL_IP" ]]; then
      found_head=1
    elif [[ -n "$ip" ]]; then
      workers+=("$ip")
    fi
  done
  (( found_head == 1 )) || { echo "Local IP is not a TPU endpoint" >&2; exit 1; }
  WORKER_IPS="$(IFS=,; echo "${workers[*]}")"
  ACCELERATOR_TYPE="$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
    --format='value(acceleratorType)' 2>/dev/null || true)"
fi

[[ -n "${WORKER_IPS:-}" ]] || { echo "WORKER_IPS is empty" >&2; exit 1; }
TPU_VERSION="${TPU_VERSION:-}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-}"
if [[ -z "$TPU_VERSION" ]]; then
  if [[ "$ACCELERATOR_TYPE" == *tpu7* ]]; then TPU_VERSION=tpu7x
  elif [[ "$ACCELERATOR_TYPE" == *6e* || "$ACCELERATOR_TYPE" == *tpu6* ]]; then TPU_VERSION=tpu6e
  else echo "TPU_VERSION is required" >&2; exit 1
  fi
fi
export TPU_VERSION

IFS=',' read -r -a worker_array <<< "$WORKER_IPS"
ALL_IPS_ARRAY=("$HEAD_INTERNAL_IP")
for ip in "${worker_array[@]}"; do
  [[ -n "$ip" && "$ip" != "$HEAD_INTERNAL_IP" ]] && ALL_IPS_ARRAY+=("$ip")
done
NUM_HOSTS=${#ALL_IPS_ARRAY[@]}
[[ "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
(( PREFILL_HOSTS_COUNT + DECODE_HOSTS_COUNT <= NUM_HOSTS )) || exit 1

PREFILL_HOSTS=("${ALL_IPS_ARRAY[@]:0:PREFILL_HOSTS_COUNT}")
DECODE_HOSTS=("${ALL_IPS_ARRAY[@]:PREFILL_HOSTS_COUNT:DECODE_HOSTS_COUNT}")
PREFILL_HEAD_IP="${PREFILL_HOSTS[0]}"
DECODE_HEAD_IP="${DECODE_HOSTS[0]}"
PREFILL_WORKERS=("${PREFILL_HOSTS[@]:1}")
DECODE_WORKERS=("${DECODE_HOSTS[@]:1}")

if [[ "$ACCELERATOR_TYPE" == *4t* || "$ACCELERATOR_TYPE" == *-4* || "$TPU_VERSION" == tpu7x ]]; then
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-4}"
else
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-8}"
fi
if [[ "$TPU_VERSION" == tpu7x ]]; then CORES_PER_CHIP="${CORES_PER_CHIP:-2}"; else CORES_PER_CHIP="${CORES_PER_CHIP:-1}"; fi
PREFILL_TP=$((PREFILL_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP))
DECODE_TP=$((DECODE_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP))

if [[ ! -f "$HOME/.ssh/id_rsa" ]]; then
  mkdir -p "$HOME/.ssh"
  ssh-keygen -t rsa -b 4096 -N "" -f "$HOME/.ssh/id_rsa" -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i "$HOME/.ssh/id_rsa")

PROJECT="$(gcloud config get-value project 2>/dev/null)"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference/vllm-tpu"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$SCRIPT_DIR/../scripts/setup_docker_env.sh"
setup_environment "$IMAGE_NAME" "true"
DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

cleanup() {
  local code=$?
  for ip in "${ALL_IPS_ARRAY[@]}"; do
    [[ "$ip" == "$HEAD_INTERNAL_IP" ]] || ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
      "docker rm -f node >/dev/null 2>&1 || true" || true
  done
  docker rm -f node disagg-proxy-benchmark >/dev/null 2>&1 || true
  if (( code != 0 )); then
    for log in prefill decode proxy benchmark correctness; do
      [[ -s "$LOG_DIR/$log.txt" ]] && { echo "+++ $log.txt"; cat "$LOG_DIR/$log.txt"; }
    done
  fi
}
trap cleanup EXIT

common_env="-e HF_TOKEN='${HF_TOKEN:-}' -e TPU_MULTIHOST_BACKEND=ray -e JAX_PLATFORMS='' -e TPU_BACKEND_TYPE=jax -e MODEL_IMPL_TYPE=vllm"
visible_chips="$(seq -s, 0 $((CHIPS_PER_HOST - 1)))"
single_host_env="-e TPU_PROCESS_BOUNDS=1,1,1 -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,${CHIPS_PER_HOST},1 -e TPU_VISIBLE_CHIPS=${visible_chips} -e CLOUD_TPU_TASK_ID=0 -e JAX_PROCESS_ID=0 -e JAX_NUM_PROCESSES=1"
PREFILL_SINGLE_ENV=""
DECODE_SINGLE_ENV=""
[[ "$PREFILL_HOSTS_COUNT" -eq 1 ]] && PREFILL_SINGLE_ENV="$single_host_env"
[[ "$DECODE_HOSTS_COUNT" -eq 1 ]] && DECODE_SINGLE_ENV="$single_host_env"
PREFILL_ENV="$PREFILL_SINGLE_ENV $common_env"
DECODE_ENV="$DECODE_SINGLE_ENV $common_env -e DRAFT_MODEL_IMPL_TYPE='$DRAFT_MODEL_IMPL_TYPE'"

launch_cluster() {
  local head=$1 role=$2 env_args=$3 workers=$4
  if [[ "$head" == "$HEAD_INTERNAL_IP" ]]; then
    bash "$TOP_DIR/scripts/multihost/run_cluster.sh" "$DOCKER_IMAGE" "$head" "$role" "$HOST_HF_HOME" $env_args &
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$head" "mkdir -p ~/tpu-inference/scripts/multihost"
    base64 < "$TOP_DIR/scripts/multihost/run_cluster.sh" | ssh "${SSH_OPTS[@]}" "$SSH_USER@$head" \
      "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$head" \
      "bash ~/tpu-inference/scripts/multihost/run_cluster.sh '$DOCKER_IMAGE' '$head' '$role' '$HOST_HF_HOME' $env_args" &
  fi
  sleep 20
  for worker in $workers; do
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$worker" "mkdir -p ~/tpu-inference/scripts/multihost"
    base64 < "$TOP_DIR/scripts/multihost/run_cluster.sh" | ssh "${SSH_OPTS[@]}" "$SSH_USER@$worker" \
      "base64 -d > ~/tpu-inference/scripts/multihost/run_cluster.sh"
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$worker" \
      "bash ~/tpu-inference/scripts/multihost/run_cluster.sh '$DOCKER_IMAGE' '$head' --worker '$HOST_HF_HOME' $env_args" &
    sleep 10
  done
}

wait_ray() {
  local host=$1
  for _ in {1..60}; do nc -z -w2 "$host" 6379 >/dev/null 2>&1 && return 0; sleep 5; done
  echo "Ray head did not start on $host" >&2
  return 1
}

launch_cluster "$PREFILL_HEAD_IP" --head "$PREFILL_ENV" "${PREFILL_WORKERS[*]:-}"
wait_ray "$PREFILL_HEAD_IP"
launch_cluster "$DECODE_HEAD_IP" --head "$DECODE_ENV" "${DECODE_WORKERS[*]:-}"
wait_ray "$DECODE_HEAD_IP"
sleep 90

PREFILL_PORT=8400
DECODE_PORT=9400
docker exec -d node bash -c "vllm serve '$MODEL' --port $PREFILL_PORT --tensor-parallel-size $PREFILL_TP --trust-remote-code --no-enable-prefix-caching --kv-transfer-config '{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"kv_producer\"}' --max-model-len 1024 > /root/vllm_serve_prefill.log 2>&1"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "docker exec -d node bash -c \"vllm serve '$MODEL' --port $DECODE_PORT --tensor-parallel-size $DECODE_TP --trust-remote-code --no-enable-prefix-caching --speculative-config '$SPECULATIVE_CONFIG' --kv-transfer-config '{\\\"kv_connector\\\":\\\"TPUConnector\\\",\\\"kv_connector_module_path\\\":\\\"tpu_inference.distributed.tpu_connector\\\",\\\"kv_role\\\":\\\"kv_consumer\\\"}' --max-model-len 1024 > /root/vllm_serve_decode.log 2>&1\""

wait_server() {
  local host=$1 port=$2
  for _ in {1..720}; do curl -fs "http://$host:$port/health" >/dev/null 2>&1 && return 0; sleep 5; done
  echo "Server did not become healthy: $host:$port" >&2
  return 1
}
wait_server 127.0.0.1 "$PREFILL_PORT"
wait_server "$DECODE_HEAD_IP" "$DECODE_PORT"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "docker exec node pgrep -af '[v]llm serve' | grep -F -- '--speculative-config'"

docker run -d --privileged --network host --shm-size 16G --name disagg-proxy-benchmark \
  -e HF_HOME=/root/hf -v "$HOST_HF_HOME:/root/hf" -v "$LOG_DIR:/root/logs" \
  --entrypoint /bin/bash "$DOCKER_IMAGE" -c 'tail -f /dev/null'
docker exec -d disagg-proxy-benchmark /bin/bash -c \
  "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port 8000 --prefiller-hosts 127.0.0.1 --prefiller-ports $PREFILL_PORT --decoder-hosts $DECODE_HEAD_IP --decoder-ports $DECODE_PORT > /root/logs/proxy.txt 2>&1"
wait_server 127.0.0.1 8000

if [[ "$TEST_MODE" == 2 || "$TEST_MODE" == 3 ]]; then
  docker exec disagg-proxy-benchmark /bin/bash -c \
    "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py --baseline_url http://$DECODE_HEAD_IP:$DECODE_PORT/v1/completions --disagg_url http://127.0.0.1:8000/v1/completions --model '$MODEL' --num_requests 20 --input_length 32 --output_length 64 > /root/logs/correctness.txt 2>&1"
  docker exec disagg-proxy-benchmark cat /root/logs/correctness.txt
fi

if [[ "$TEST_MODE" == 1 || "$TEST_MODE" == 3 ]]; then
  docker exec disagg-proxy-benchmark /bin/bash -c \
    "vllm bench serve --backend vllm --host 127.0.0.1 --port 8000 --model '$MODEL' --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN --num-prompts $NUM_PROMPTS --request-rate inf --max-concurrency $MAX_CONCURRENCY --trust-remote-code --seed $RANDOM_SEED > /root/logs/benchmark.txt 2>&1"
  docker exec disagg-proxy-benchmark cat /root/logs/benchmark.txt
fi

echo "Multi-host P/D speculative test completed: method=$SPECULATIVE_METHOD model=$MODEL"
