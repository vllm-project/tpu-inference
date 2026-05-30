#!/bin/bash
# Quick deploy script for v7x-16 PathwaysJob workload:
#   1. Tars the local tpu-inference repo (excluding heavy/unnecessary dirs)
#   2. Uploads it to GCS
#   3. Patches the colocated_python image with the latest tpu-inference code
#      and pushes it as a new tag (the original :latest is untouched).
#   4. Deletes the existing PathwaysJob workload
#   5. Applies the updated workload YAML
#
# Usage: ./deploy_v7x_16.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML_FILE="${SCRIPT_DIR}/pathways_job_v7x_16.yaml"

GCS_BUCKET="gs://wenxindong-multipod-dev"
GCS_PATCH_PATH="${GCS_BUCKET}/patches/tpu-inference.tar.gz"

BASE_COLOCATED_IMAGE="us-east5-docker.pkg.dev/cloud-tpu-inference-test/wenxindong/colocated_python:latest"
PATCHED_COLOCATED_IMAGE="us-east5-docker.pkg.dev/cloud-tpu-inference-test/wenxindong/colocated_python:patched"

WORKLOAD_NAME="wenxindongtest"
KUBE_CONTEXT="gke_cloud-tpu-inference-test_us-central1_wenxindong-pw-tpu7x-16"

echo "=== Packaging tpu-inference ==="
cd "${REPO_DIR}"
tar czf /tmp/tpu-inference.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.tox' \
  --exclude='.nox' \
  --exclude='dist' \
  --exclude='build' \
  .

echo "=== Uploading to ${GCS_PATCH_PATH} ==="
gcloud storage cp /tmp/tpu-inference.tar.gz "${GCS_PATCH_PATH}"


echo "=== Patching colocated_python image with latest tpu-inference ==="
gcloud auth configure-docker us-east5-docker.pkg.dev --quiet

docker pull "${BASE_COLOCATED_IMAGE}"

# Capture the base image's ENTRYPOINT/CMD as JSON so we can restore them in
# `docker commit` — commit captures the *container's* config (which we override
# with --entrypoint sleep below to keep the container alive for exec), NOT the
# base image's. Without these --change flags the patched image would just run
# `sleep infinity` and the colocated_python gRPC server would never start.
BASE_ENTRYPOINT_JSON=$(docker image inspect "${BASE_COLOCATED_IMAGE}" --format '{{json .Config.Entrypoint}}')
BASE_CMD_JSON=$(docker image inspect "${BASE_COLOCATED_IMAGE}" --format '{{json .Config.Cmd}}')

PATCH_CONTAINER="colocated-python-patch-$$"
docker run -d --name "${PATCH_CONTAINER}" --entrypoint sleep "${BASE_COLOCATED_IMAGE}" infinity
trap 'docker rm -f "${PATCH_CONTAINER}" >/dev/null 2>&1 || true' EXIT

# Copy the freshly-built tarball (identical bytes to what we just uploaded
# to GCS) into the container and install it editable.
docker cp /tmp/tpu-inference.tar.gz "${PATCH_CONTAINER}:/tmp/tpu-inference.tar.gz"
docker exec "${PATCH_CONTAINER}" bash -c '
  set -e
  mkdir -p /workspace/tpu-inference
  tar xzf /tmp/tpu-inference.tar.gz -C /workspace/tpu-inference
  pip install --no-deps -e /workspace/tpu-inference
'

docker commit \
  --change="ENTRYPOINT ${BASE_ENTRYPOINT_JSON}" \
  --change="CMD ${BASE_CMD_JSON}" \
  "${PATCH_CONTAINER}" "${PATCHED_COLOCATED_IMAGE}"
docker push "${PATCHED_COLOCATED_IMAGE}"
docker rm -f "${PATCH_CONTAINER}" >/dev/null 2>&1 || true
trap - EXIT


echo "=== Deleting existing workload '${WORKLOAD_NAME}' (if any) ==="
kubectl --context="${KUBE_CONTEXT}" delete jobset "${WORKLOAD_NAME}" --ignore-not-found

echo "=== Applying ${YAML_FILE} ==="
kubectl --context="${KUBE_CONTEXT}" apply -f "${YAML_FILE}"

echo "=== Done! Workload '${WORKLOAD_NAME}' deployed with latest local changes. ==="

echo "=== Waiting for pod to be created ==="
while true; do
  POD_NAME=$(kubectl --context="${KUBE_CONTEXT}" get pods -l "jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" -o name | head -n 1)
  if [ -n "${POD_NAME}" ]; then
    break
  fi
  sleep 2
done

sleep 3
echo "=== Following logs of pod ${POD_NAME} ==="
kubectl --context="${KUBE_CONTEXT}" logs -f "${POD_NAME}" -c jax-tpu

