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

# Run the JAX 112-MiB all-reduce microbenchmark on a multi-host tpu7x slice.
# Modeled on run_multihost.sh (worker discovery, image build+push, SSH fanout)
# but runs a plain JAX SPMD script on every host instead of a Ray cluster.

set -euo pipefail
set -x

export SSH_USER="${SSH_USER:-$(whoami)}"
CONTAINER_NAME="arjax"
# Per-build directory: files inside are written by root from the container, so
# the agent user cannot rm an old one; stale dirs are removed through docker.
RESULTS_DIR="${RESULTS_DIR:-$HOME/allreduce_jax_results_${BUILDKITE_BUILD_NUMBER:-$$}}"
CLEAN_OLD_RESULTS="docker run --rm -v \$HOME:/h --entrypoint /bin/sh ${DOCKER_IMAGE_FOR_CLEAN:-busybox} -c 'rm -rf /h/allreduce_jax_results*' >/dev/null 2>&1 || true"
# Optional libtpu override for the run (e.g. LIBTPU_PIP_SPEC=libtpu==0.0.44.1) so the
# JAX numbers can be compared against a torch_tpu image pinned to a different libtpu.
PIP_PREFIX=""
if [[ -n "${LIBTPU_PIP_SPEC:-}" ]]; then
  PIP_PREFIX="pip install -q --no-deps ${LIBTPU_PIP_SPEC} && pip show libtpu | grep -i ^version && "
fi
BENCH_CMD="${PIP_PREFIX}timeout 2700 python /workspace/tpu_inference/tests/e2e/allreduce_bench_jax.py --out-dir /results ${ALLREDUCE_JAX_ARGS:-}"

# Automatic worker IP discovery (same as run_multihost.sh)
if [[ -z "${WORKER_IPS:-}" ]]; then
  echo "WORKER_IPS not provided. Discovering via gcloud..."
  ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
  TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"
  if [[ -z "$TPU_NAME" || -z "$ZONE" ]]; then
    echo "Could not determine TPU_NAME or ZONE from metadata."
    exit 1
  fi
  ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)")
  ALL_IPS="${ALL_IPS//;/ }"
  ALL_IPS="${ALL_IPS//,/ }"
  # shellcheck disable=SC2206
  ALL_IPS_ARRAY=($ALL_IPS)
  HEAD_INTERNAL_IP="${ALL_IPS_ARRAY[0]}"
  WORKER_IPS_LIST=("${ALL_IPS_ARRAY[@]:1}")
  WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
  echo "Head: $HEAD_INTERNAL_IP Workers: $WORKER_IPS"
fi

if [[ "${TPU_VERSION:-}" != "tpu7x" ]]; then
  echo "This script is strictly for TPU_VERSION=tpu7x. Exiting."
  exit 0
fi

if [ ! -f ~/.ssh/id_rsa ]; then
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i ~/.ssh/id_rsa)

IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"

cleanup() {
  echo "Cleaning up benchmark containers..."
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true" || true
  done
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

PROJECT="$(gcloud config get-value project)"
GCR_REPO="us-central1-docker.pkg.dev/${PROJECT}/tpu-inference"
IMAGE_NAME="${GCR_REPO}/vllm-tpu"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

echo "--- Pruning Docker on head node..."
docker system prune -a --volumes -f || true

# Build the image on the head node and push so workers can pull it.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
setup_environment "${IMAGE_NAME}" "true"

DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

cleanup

DOCKER_RUN_ARGS=(
  --privileged
  --net host
  --shm-size=16G
  --name "${CONTAINER_NAME}"
  --entrypoint /bin/bash
)

echo "--- Starting benchmark on workers..."
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f || true" || true
  # shellcheck disable=SC2029
  # JAX may place process 0 (the one that writes HLO + traces) on any host,
  # so every container gets the results volume and the head gathers them.
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "${CLEAN_OLD_RESULTS}; mkdir -p ${RESULTS_DIR} && \
     gcloud auth configure-docker us-central1-docker.pkg.dev -q && \
     docker pull ${DOCKER_IMAGE} && \
     docker run -d ${DOCKER_RUN_ARGS[*]} -v ${RESULTS_DIR}:/results ${DOCKER_IMAGE} -c '${BENCH_CMD}'"
done

echo "--- Starting benchmark on head node (foreground)..."
# setup_environment removes the local image after pushing; pull it back.
docker pull "${DOCKER_IMAGE}"
eval "${CLEAN_OLD_RESULTS}"
mkdir -p "${RESULTS_DIR}"
set +e
docker run "${DOCKER_RUN_ARGS[@]}" -v "${RESULTS_DIR}:/results" "${DOCKER_IMAGE}" -c "${BENCH_CMD}"
HEAD_EXIT=$?
set -e

echo "--- Worker logs ---"
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  echo "===== worker ${worker_ip} ====="
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker logs ${CONTAINER_NAME} 2>&1 | tail -n 100" || true
done

echo "--- Gathering results from workers"
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker wait ${CONTAINER_NAME} >/dev/null 2>&1 || true; cd ${RESULTS_DIR} && tar cf - ." \
    | tar xf - -C "${RESULTS_DIR}" || echo "no results from ${worker_ip}"
done
echo "--- Results in ${RESULTS_DIR}"
find "${RESULTS_DIR}" -type f | sed 's/^/  /' || true
mkdir -p perf_results && cp -r "${RESULTS_DIR}" perf_results/allreduce_jax || true
echo "Head exit code: ${HEAD_EXIT}"
exit "${HEAD_EXIT}"
