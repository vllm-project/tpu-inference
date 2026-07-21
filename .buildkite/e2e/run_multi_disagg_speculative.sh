#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Multi-host P/D disaggregation plus speculative decoding.
# Keep the default model aligned with run_multi_disagg.sh so this correctness
# test fits a tpu7x-8 P/D split. Speculative decoding is enabled only on Decode.

set -euo pipefail

export SSH_USER="${SSH_USER:-$(whoami)}"
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"

# Test inputs and load defaults. These are environment-overridable so the same
# script can serve both the CI workload and targeted debugging runs.
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"
RANDOM_SEED="${RANDOM_SEED:-10}"
TEST_MODE="${TEST_MODE:-3}" # 1: benchmark, 2: correctness, 3: both
# A tpu7x-8 P/D run assigns one host to each independent engine. Reserve HBM
# for compilation/runtime buffers instead of letting the KV cache consume the
# default share, which can make KV-cache initialization fail.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
NODE_CONTAINER_NAME="node"
PROXY_CONTAINER_NAME="disagg-proxy-benchmark"
PREFILL_HOSTS_COUNT="${PREFILL_HOSTS_COUNT:-1}"
DECODE_HOSTS_COUNT="${DECODE_HOSTS_COUNT:-1}"
DEFAULT_SPECULATIVE_CONFIG='{"method":"ngram","prompt_lookup_max":5,"prompt_lookup_min":3,"num_speculative_tokens":3}'
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-$DEFAULT_SPECULATIVE_CONFIG}"
DRAFT_MODEL_IMPL_TYPE="${DRAFT_MODEL_IMPL_TYPE:-auto}"

# Reject malformed configuration before creating containers or contacting peers.
if ! python3 -c 'import json, sys; json.loads(sys.argv[1])' "$SPECULATIVE_CONFIG"; then
  echo "ERROR: SPECULATIVE_CONFIG must be valid JSON: $SPECULATIVE_CONFIG" >&2
  exit 1
fi
# Keep this value out of generated shell source.  JSON strings may legally
# contain single quotes, which would otherwise break the remote launcher.
SPECULATIVE_CONFIG_B64="$(printf '%s' "$SPECULATIVE_CONFIG" | base64 | tr -d '\n')"
SPECULATIVE_METHOD="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["method"])' "$SPECULATIVE_CONFIG")"

# Log directory setup
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/{prefill,decode,proxy,benchmark,correctness}.txt

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

# Prefer an explicit worker list, but discover the full TPU slice when CI has
# not provided one. The current VM is kept separate as the local head host.
if [[ -z "${WORKER_IPS:-}" ]]; then
    echo "⚠️  WORKER_IPS not provided. Attempting to discover via gcloud..."

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
                echo "❌ Current VM IP (${HEAD_INTERNAL_IP}) is not in discovered TPU endpoints: ${ALL_IPS_ARRAY[*]}"
                exit 1
            fi

            WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
            echo "   -> Discovered Worker IPs: $WORKER_IPS"

            ACCELERATOR_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(acceleratorType)" 2>/dev/null || echo "")
            echo "   -> Detected Accelerator Type: $ACCELERATOR_TYPE"
            if [[ -z "${TPU_VERSION:-}" ]]; then
                if [[ "$ACCELERATOR_TYPE" == *"tpu7"* ]]; then
                    export TPU_VERSION="tpu7x"
                    echo "   -> Setting TPU_VERSION=tpu7x"
                elif [[ "$ACCELERATOR_TYPE" == *"6e"* ]] || [[ "$ACCELERATOR_TYPE" == *"tpu6"* ]]; then
                    export TPU_VERSION="tpu6e"
                    echo "   -> Setting TPU_VERSION=tpu6e"
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

if [[ -z "${WORKER_IPS:-}" ]]; then
    echo "ERROR: Failed to discover WORKER_IPS. Please provide it manually."
    exit 1
fi

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(get_current_internal_ip)}"

# Accelerator type determines the number of addressable TPU chips per host.
if [[ -z "${ACCELERATOR_TYPE:-}" ]] && command -v gcloud &> /dev/null && command -v curl &> /dev/null; then
    ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}' || echo "")}"
    TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"

    if [[ -n "$TPU_NAME" && -n "$ZONE" ]]; then
        ACCELERATOR_TYPE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(acceleratorType)" 2>/dev/null || echo "")
        echo "   -> Detected Accelerator Type: $ACCELERATOR_TYPE"
    fi
fi

# Auto-discover TPU_VERSION if not specified and ACCELERATOR_TYPE is present
if [[ -z "${TPU_VERSION:-}" && -n "${ACCELERATOR_TYPE:-}" ]]; then
    if [[ "$ACCELERATOR_TYPE" == *"tpu7"* ]]; then
        export TPU_VERSION="tpu7x"
        echo "   -> Setting TPU_VERSION=tpu7x"
    elif [[ "$ACCELERATOR_TYPE" == *"6e"* ]] || [[ "$ACCELERATOR_TYPE" == *"tpu6"* ]]; then
        export TPU_VERSION="tpu6e"
        echo "   -> Setting TPU_VERSION=tpu6e"
    fi
fi

echo "Running on TPU_VERSION: ${TPU_VERSION}"

IFS=',' read -r -a worker_array <<< "$WORKER_IPS"
# Host order defines the partition: prefill consumes the first N hosts and
# decode consumes the following N hosts. This keeps the two Ray clusters
# disjoint while allowing the local head to run prefill.
ALL_IPS_ARRAY=("$HEAD_INTERNAL_IP")
for ip in "${worker_array[@]}"; do
  [[ -n "$ip" && "$ip" != "$HEAD_INTERNAL_IP" ]] && ALL_IPS_ARRAY+=("$ip")
done
NUM_HOSTS=${#ALL_IPS_ARRAY[@]}
echo "Discovered TPU hosts in launch order: ${ALL_IPS_ARRAY[*]}"
echo "Current/local head IP: ${HEAD_INTERNAL_IP}"
echo "Total TPU hosts available: ${NUM_HOSTS}"
[[ "$PREFILL_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ "$DECODE_HOSTS_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
(( PREFILL_HOSTS_COUNT + DECODE_HOSTS_COUNT <= NUM_HOSTS )) || exit 1

PREFILL_HOSTS=("${ALL_IPS_ARRAY[@]:0:PREFILL_HOSTS_COUNT}")
DECODE_HOSTS=("${ALL_IPS_ARRAY[@]:PREFILL_HOSTS_COUNT:DECODE_HOSTS_COUNT}")
PREFILL_HEAD_IP="${PREFILL_HOSTS[0]}"
DECODE_HEAD_IP="${DECODE_HOSTS[0]}"
PREFILL_WORKERS=("${PREFILL_HOSTS[@]:1}")
DECODE_WORKERS=("${DECODE_HOSTS[@]:1}")
echo "Prefill hosts: ${PREFILL_HOSTS[*]}"
echo "Decode hosts: ${DECODE_HOSTS[*]}"

# Tensor parallel size is the total number of TPU cores assigned to each
# service, rather than simply its VM count.
if [[ "$ACCELERATOR_TYPE" == *4t* || "$ACCELERATOR_TYPE" == *-4* || "$TPU_VERSION" == tpu7x ]]; then
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-4}"
else
  CHIPS_PER_HOST="${CHIPS_PER_HOST:-8}"
fi
if [[ "$TPU_VERSION" == tpu7x ]]; then CORES_PER_CHIP="${CORES_PER_CHIP:-2}"; else CORES_PER_CHIP="${CORES_PER_CHIP:-1}"; fi
PREFILL_TP=$((PREFILL_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP))
DECODE_TP=$((DECODE_HOSTS_COUNT * CHIPS_PER_HOST * CORES_PER_CHIP))
echo "Calculated PREFILL_TENSOR_PARALLEL_SIZE: ${PREFILL_TP}"
echo "Calculated DECODE_TENSOR_PARALLEL_SIZE: ${DECODE_TP}"

if [[ ! -f "$HOME/.ssh/id_rsa" ]]; then
  mkdir -p "$HOME/.ssh"
  ssh-keygen -t rsa -b 4096 -N "" -f "$HOME/.ssh/id_rsa" -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i "$HOME/.ssh/id_rsa")

PROJECT="$(gcloud config get-value project 2>/dev/null)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

echo "--- Building and pushing speculative decoding Docker image ---"
PREFILL_LOG_TAIL_PID=""
DECODE_LOG_TAIL_PID=""

stop_vllm_log_streaming() {
  for pid_var in PREFILL_LOG_TAIL_PID DECODE_LOG_TAIL_PID; do
    local pid="${!pid_var:-}"
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      printf -v "$pid_var" ''
    fi
  done
}

start_vllm_log_streaming() {
  # Stream both detached server logs into Buildkite while health checks wait;
  # otherwise a slow model-load failure is difficult to diagnose.
  stop_vllm_log_streaming
  echo "--- Streaming vLLM Prefill and Decode logs while waiting for health..."

  docker exec "$NODE_CONTAINER_NAME" bash -c \
    "touch /root/vllm_serve_prefill.log && tail -n +1 -F /root/vllm_serve_prefill.log" \
    > >(sed -u 's/^/[prefill] /') 2>&1 &
  PREFILL_LOG_TAIL_PID=$!

  ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
    "docker exec '$NODE_CONTAINER_NAME' bash -c 'touch /root/vllm_serve_decode.log && tail -n +1 -F /root/vllm_serve_decode.log'" \
    > >(sed -u 's/^/[decode] /') 2>&1 &
  DECODE_LOG_TAIL_PID=$!
}

cleanup() {
  local code=$?
  local cleanup_failed=0
  echo "🧹 Cleaning up speculative disaggregation containers..."
  stop_vllm_log_streaming

  if ! docker cp "$NODE_CONTAINER_NAME:/root/vllm_serve_prefill.log" "$LOG_DIR/prefill.txt" >/dev/null 2>&1; then
    echo "WARNING: Failed to capture the Prefill server log." >&2
    cleanup_failed=1
  fi
  if [[ -n "${DECODE_HEAD_IP:-}" ]]; then
    local capture_status
    if ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
      "rm -f /tmp/vllm_serve_decode.log || exit 1; if docker container inspect '$NODE_CONTAINER_NAME' >/dev/null 2>&1 && docker exec '$NODE_CONTAINER_NAME' test -f /root/vllm_serve_decode.log; then docker cp '$NODE_CONTAINER_NAME:/root/vllm_serve_decode.log' /tmp/vllm_serve_decode.log >/dev/null; else exit 2; fi"; then
      capture_status=0
    else
      capture_status=$?
    fi
    if (( capture_status == 0 )); then
      if ! scp "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP:/tmp/vllm_serve_decode.log" "$LOG_DIR/decode.txt" >/dev/null 2>&1; then
        echo "WARNING: Failed to copy the Decode server log from ${DECODE_HEAD_IP}." >&2
        cleanup_failed=1
      fi
    elif (( capture_status == 2 )); then
      echo "INFO: Decode server log was not available from ${DECODE_HEAD_IP}." >&2
    else
      echo "WARNING: Failed to capture the Decode server log from ${DECODE_HEAD_IP}." >&2
      cleanup_failed=1
    fi
    if ! ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" "rm -f /tmp/vllm_serve_decode.log"; then
      echo "WARNING: Failed to remove /tmp/vllm_serve_decode.log on ${DECODE_HEAD_IP}." >&2
      cleanup_failed=1
    fi
  fi

  # Every remote cluster member owns a node container. Remove it even after a
  # failed launch so later jobs reusing this TPU slice do not inherit Ray or
  # vLLM processes. Cleanup continues after individual-host failures.
  for ip in "${ALL_IPS_ARRAY[@]}"; do
    if [[ "$ip" != "$HEAD_INTERNAL_IP" ]]; then
      echo "   -> Cleaning remote host: $ip"
      local remote_files="~/tpu-inference/scripts/multihost/run_cluster.sh"
      [[ "$ip" == "$DECODE_HEAD_IP" ]] && remote_files="~/tpu-inference/scripts/start_decode.sh $remote_files"
      ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
        "docker stop '$NODE_CONTAINER_NAME' >/dev/null 2>&1 || true
         docker rm -f '$NODE_CONTAINER_NAME' >/dev/null 2>&1 || true
         rm -f $remote_files
         rmdir ~/tpu-inference/scripts/multihost ~/tpu-inference/scripts 2>/dev/null || true" || true
    fi
  done
  if ! docker info >/dev/null 2>&1; then
    echo "WARNING: Local Docker daemon is unavailable during cleanup." >&2
    cleanup_failed=1
  else
    for container in "$NODE_CONTAINER_NAME" "$PROXY_CONTAINER_NAME"; do
      if docker container inspect "$container" >/dev/null 2>&1 && ! docker rm -f "$container" >/dev/null 2>&1; then
        echo "WARNING: Failed to remove local container ${container}." >&2
        cleanup_failed=1
      fi
    done
  fi
  if ! rm -f /tmp/start_prefill.sh /tmp/start_decode.sh; then
    echo "WARNING: Failed to remove local temporary launchers." >&2
    cleanup_failed=1
  fi
  if (( code != 0 || cleanup_failed )); then
    echo "--- Diagnostic logs (job or cleanup failure) ---" >&2
    for log in prefill decode proxy benchmark correctness; do
      [[ -s "$LOG_DIR/$log.txt" ]] && { echo "+++ $log.txt"; cat "$LOG_DIR/$log.txt"; }
    done
  fi
  if (( cleanup_failed )); then
    echo "WARNING: Cleanup completed with failures." >&2
    (( code == 0 )) && exit 1
  fi
  echo "✅ Cleanup complete."
  return "$code"
}
# From this point onward, any error must collect logs and remove containers.
trap cleanup EXIT

source "$SCRIPT_DIR/../scripts/setup_docker_env.sh"
setup_environment "$IMAGE_NAME" "true"
DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"
echo "Using Docker image: ${DOCKER_IMAGE}"

common_env="-e HF_TOKEN=${HF_TOKEN:-} -e TPU_MULTIHOST_BACKEND=ray -e JAX_PLATFORMS= -e TPU_BACKEND_TYPE=jax -e MODEL_IMPL_TYPE=vllm"
visible_chips="$(seq -s, 0 $((CHIPS_PER_HOST - 1)))"
# Single-host Ray clusters need explicit JAX process bounds; multi-host
# clusters obtain their process topology from the Ray launch helper.
single_host_env="-e TPU_PROCESS_BOUNDS=1,1,1 -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,${CHIPS_PER_HOST},1 -e TPU_VISIBLE_CHIPS=${visible_chips} -e CLOUD_TPU_TASK_ID=0 -e JAX_PROCESS_ID=0 -e JAX_NUM_PROCESSES=1"
PREFILL_SINGLE_ENV=""
DECODE_SINGLE_ENV=""
[[ "$PREFILL_HOSTS_COUNT" -eq 1 ]] && PREFILL_SINGLE_ENV="$single_host_env"
[[ "$DECODE_HOSTS_COUNT" -eq 1 ]] && DECODE_SINGLE_ENV="$single_host_env"
PREFILL_ENV="$PREFILL_SINGLE_ENV $common_env"
DECODE_ENV="$DECODE_SINGLE_ENV $common_env -e DRAFT_MODEL_IMPL_TYPE=${DRAFT_MODEL_IMPL_TYPE}"

launch_cluster() {
  local head=$1 role=$2 env_args=$3 workers=$4
  # Remote VMs may not have this checkout at the same path. Transfer the
  # launcher itself and start it asynchronously so all Ray members can join.
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

dump_ray_resources() {
  local host=$1 label=$2
  local ray_dump_cmd="import json, ray; ray.init(address='auto', ignore_reinit_error=True); print(json.dumps(ray.nodes(), indent=2, sort_keys=True))"
  echo "--- Ray node resources for ${label} cluster (${host}) ---" >&2
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$NODE_CONTAINER_NAME" python3 -c "$ray_dump_cmd" >&2 || true
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "docker exec '$NODE_CONTAINER_NAME' python3 -c \"${ray_dump_cmd}\"" >&2 || true
  fi
}

wait_for_ray_cluster_members() {
  local host=$1 expected_nodes=$2 label=$3 timeout=${4:-600}
  local ray_ready_cmd="import ray; ray.init(address='auto', ignore_reinit_error=True); alive=sum(node.get('Alive', False) for node in ray.nodes()); raise SystemExit(0 if alive >= ${expected_nodes} else 1)"
  # A reachable head is insufficient: vLLM must see every expected Ray node
  # before tensor-parallel initialization can succeed.
  local end_time=$((SECONDS + timeout))
  echo "Waiting for ${label} Ray cluster to register ${expected_nodes} alive node(s)..."
  while (( SECONDS < end_time )); do
    if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
      if docker exec "$NODE_CONTAINER_NAME" python3 -c "$ray_ready_cmd" >/dev/null 2>&1; then
        echo "${label} Ray cluster has registered ${expected_nodes} alive node(s)."
        return 0
      fi
    elif ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "docker exec '$NODE_CONTAINER_NAME' python3 -c \"${ray_ready_cmd}\"" >/dev/null 2>&1; then
      echo "${label} Ray cluster has registered ${expected_nodes} alive node(s)."
      return 0
    fi
    sleep 5
  done
  echo "ERROR: ${label} Ray cluster did not register ${expected_nodes} alive node(s) within ${timeout}s." >&2
  dump_ray_resources "$host" "$label"
  return 1
}

echo "--- 1. Starting Prefill Ray Cluster ---"
launch_cluster "$PREFILL_HEAD_IP" --head "$PREFILL_ENV" "${PREFILL_WORKERS[*]:-}"
wait_for_ray_cluster_members "$PREFILL_HEAD_IP" "$PREFILL_HOSTS_COUNT" "Prefill"

echo "--- 2. Starting Decode Ray Cluster ---"
launch_cluster "$DECODE_HEAD_IP" --head "$DECODE_ENV" "${DECODE_WORKERS[*]:-}"
wait_for_ray_cluster_members "$DECODE_HEAD_IP" "$DECODE_HOSTS_COUNT" "Decode"

echo "--- 3. Starting vLLM Prefill and Decode Servers ---"
PREFILL_PORT=8400
DECODE_PORT=9400
echo "Starting vLLM Prefill server on ${PREFILL_HEAD_IP}:${PREFILL_PORT}..."
# Keep the vLLM process detached from the CI shell, but redirect its output to
# a container file that can be streamed during startup and captured on failure.
cat << EOF > /tmp/start_prefill.sh
#!/bin/bash
set -x
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  ${NODE_CONTAINER_NAME} bash -c "vllm serve ${MODEL} \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size ${PREFILL_TP} \
    --trust-remote-code \
    --load-format ${LOAD_FORMAT} \
    --no-enable-prefix-caching \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --kv-transfer-config '{\"kv_connector\": \"TPUConnector\", \"kv_connector_module_path\": \"tpu_inference.distributed.tpu_connector\", \"kv_role\": \"kv_producer\"}' \
    --max-model-len ${MAX_MODEL_LEN} > /root/vllm_serve_prefill.log 2>&1"
set +x
EOF
chmod +x /tmp/start_prefill.sh
bash /tmp/start_prefill.sh

echo "Starting vLLM Decode server with speculative decoding on ${DECODE_HEAD_IP}:${DECODE_PORT}..."
# The launcher is copied to the decode host because it is not necessarily the
# local VM. Base64 makes SPECULATIVE_CONFIG safe to embed in generated source;
# it is then passed as bash's positional argument, not parsed as shell syntax.
cat << EOF > /tmp/start_decode.sh
#!/bin/bash
set -x
SPECULATIVE_CONFIG="\$(python3 -c 'import base64, sys; sys.stdout.buffer.write(base64.b64decode(sys.argv[1]))' '${SPECULATIVE_CONFIG_B64}')"
docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  -e 'KV_TRANSFER_CONFIG={"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}' \
  ${NODE_CONTAINER_NAME} bash -c 'vllm serve ${MODEL} \
    --port ${DECODE_PORT} \
    --tensor-parallel-size ${DECODE_TP} \
    --trust-remote-code \
    --load-format ${LOAD_FORMAT} \
    --no-enable-prefix-caching \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --speculative-config "\$1" \
    --kv-transfer-config "\$KV_TRANSFER_CONFIG" \
    --max-model-len ${MAX_MODEL_LEN} > /root/vllm_serve_decode.log 2>&1' _ "\$SPECULATIVE_CONFIG"
set +x
EOF
chmod +x /tmp/start_decode.sh
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" "mkdir -p ~/tpu-inference/scripts"
base64 < /tmp/start_decode.sh | ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "base64 -d > ~/tpu-inference/scripts/start_decode.sh"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "chmod +x ~/tpu-inference/scripts/start_decode.sh && bash ~/tpu-inference/scripts/start_decode.sh"

wait_server() {
  local host=$1 port=$2
  for _ in {1..720}; do curl -fs "http://$host:$port/health" >/dev/null 2>&1 && return 0; sleep 5; done
  echo "Server did not become healthy: $host:$port" >&2
  return 1
}

print_vllm_log() {
  local host=$1 log_path=$2
  echo "--- vLLM server log: $host:$log_path ---" >&2
  if [[ "$host" == "$HEAD_INTERNAL_IP" ]]; then
    docker exec "$NODE_CONTAINER_NAME" cat "$log_path" 2>&1 || true
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "docker exec '$NODE_CONTAINER_NAME' cat '$log_path'" 2>&1 || true
  fi
  echo "--- end vLLM server log ---" >&2
}

wait_vllm_server() {
  local health_host=$1 port=$2 node_host=$3 log_path=$4
  local vllm_running
  echo "Waiting for vLLM server on ${health_host}:${port} to become healthy..."
  for attempt in {1..720}; do
    curl -fs "http://$health_host:$port/health" >/dev/null 2>&1 && return 0

    # Allow up to 10 minutes for model loading and initialization. After that,
    # if the vLLM process has exited it cannot become healthy, so fail early
    # and print the server log instead of waiting for the full timeout.
    if (( attempt > 120 )); then
      if [[ "$node_host" == "$HEAD_INTERNAL_IP" ]]; then
        vllm_running="$(docker exec "$NODE_CONTAINER_NAME" pgrep -f '[v]llm serve' 2>/dev/null || true)"
      else
        vllm_running="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$node_host" \
          "docker exec '$NODE_CONTAINER_NAME' pgrep -f '[v]llm serve' 2>/dev/null || true" 2>/dev/null || true)"
      fi
      if [[ -z "$vllm_running" ]]; then
        echo "vLLM process exited before server became healthy: $health_host:$port" >&2
        print_vllm_log "$node_host" "$log_path"
        return 1
      fi
    fi
    sleep 5
  done

  echo "Server did not become healthy before timeout: $health_host:$port" >&2
  print_vllm_log "$node_host" "$log_path"
  return 1
}

assert_ngram_draft_tokens() {
  # Starting a server with --speculative-config only checks configuration. For
  # n-gram decoding, prove the workload actually reached the speculative path.
  [[ "$SPECULATIVE_METHOD" == "ngram" ]] || return 0

  local metrics
  metrics="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
    "curl -fsS http://127.0.0.1:${DECODE_PORT}/metrics")"
  echo "--- Decode speculative-decoding metrics ---"
  printf '%s\n' "$metrics" | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens(_total)?(\{| )' || true

  if ! printf '%s\n' "$metrics" | awk '
    $1 ~ /^vllm:spec_decode_num_draft_tokens(_total)?(\{|$)/ { total += $NF }
    END { exit !(total > 0) }
  '; then
    echo "ERROR: n-gram speculative decoding produced no draft tokens." >&2
    return 1
  fi
}

run_ngram_disagg_probe() {
  # Send a request through the P/D proxy before the comparison workload. This
  # makes the metrics assertion below evidence for the composed path, rather
  # than merely for a direct request to the decode server.
  [[ "$SPECULATIVE_METHOD" == "ngram" ]] || return 0

  local request_body
  request_body="$(python3 -c '
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    # Match the n-gram E2E workload: the generated suffix is likely to
    # continue this repeated token sequence, which lets the proposer find a
    # suffix match instead of merely seeing repetition elsewhere in the prompt.
    "prompt": "Keep repeating: " + "a " * 20,
    "max_tokens": 32,
    "temperature": 0.0,
    "ignore_eos": True,
}))
' "$MODEL")"
  curl -fsS http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    --data "$request_body" >/dev/null
  assert_ngram_draft_tokens
}

echo "--- 4. Checking vLLM health and speculative decoding process ---"
start_vllm_log_streaming
wait_vllm_server 127.0.0.1 "$PREFILL_PORT" "$PREFILL_HEAD_IP" /root/vllm_serve_prefill.log
wait_vllm_server "$DECODE_HEAD_IP" "$DECODE_PORT" "$DECODE_HEAD_IP" /root/vllm_serve_decode.log
ssh "${SSH_OPTS[@]}" "$SSH_USER@$DECODE_HEAD_IP" \
  "docker exec '$NODE_CONTAINER_NAME' pgrep -af '[v]llm serve' | grep -F -- '--speculative-config'"
stop_vllm_log_streaming

echo "--- 5. Starting Proxy and Benchmark Container ---"
# The proxy is the disaggregated path under test. Correctness compares it with
# requests sent directly to decode; benchmark traffic is sent only through it.
docker run -d --privileged --network host --shm-size 16G --name "$PROXY_CONTAINER_NAME" \
  -e HF_HOME=/root/hf -v "$HOST_HF_HOME:/root/hf" -v "$LOG_DIR:/root/logs" \
  --entrypoint /bin/bash "$DOCKER_IMAGE" -c 'tail -f /dev/null'
docker exec -d "$PROXY_CONTAINER_NAME" /bin/bash -c \
  "python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port 8000 --prefiller-hosts 127.0.0.1 --prefiller-ports $PREFILL_PORT --decoder-hosts $DECODE_HEAD_IP --decoder-ports $DECODE_PORT > /root/logs/proxy.txt 2>&1"
echo "Waiting for Toy Proxy Server on 127.0.0.1:8000..."
wait_server 127.0.0.1 8000
run_ngram_disagg_probe

if [[ "$TEST_MODE" == 2 || "$TEST_MODE" == 3 ]]; then
  echo "--- Running correctness test ---"
  docker exec "$PROXY_CONTAINER_NAME" /bin/bash -c \
    "python3 /workspace/tpu_inference/examples/disagg/test_disagg_correctness.py --baseline_url http://$DECODE_HEAD_IP:$DECODE_PORT/v1/completions --disagg_url http://127.0.0.1:8000/v1/completions --model '$MODEL' --num_requests 20 --input_length 32 --output_length 64 --prompt-mode repeated-ngram > /root/logs/correctness.txt 2>&1"
  docker exec "$PROXY_CONTAINER_NAME" cat /root/logs/correctness.txt
fi

if [[ "$TEST_MODE" == 1 || "$TEST_MODE" == 3 ]]; then
  echo "--- Running benchmark test ---"
  docker exec "$PROXY_CONTAINER_NAME" /bin/bash -c \
    "vllm bench serve --backend vllm --host 127.0.0.1 --port 8000 --model '$MODEL' --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN --num-prompts $NUM_PROMPTS --request-rate inf --max-concurrency $MAX_CONCURRENCY --trust-remote-code --seed $RANDOM_SEED > /root/logs/benchmark.txt 2>&1"
  docker exec "$PROXY_CONTAINER_NAME" cat /root/logs/benchmark.txt
fi

echo "--- Tests completed successfully ---"
echo "Multi-host P/D speculative test completed: config=$SPECULATIVE_CONFIG model=$MODEL"
