#!/bin/bash
# Quick deploy script for v5p PathwaysJob workload:
#   1. Tars the local tpu-inference repo (excluding heavy/unnecessary dirs)
#   2. Uploads it to GCS
#   3. Deletes the existing PathwaysJob workload
#   4. Applies the updated workload YAML
#
# Usage: ./deploy_v5p.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML_FILE="${SCRIPT_DIR}/pathways_job_v7x_16.yaml"

GCS_BUCKET="gs://wenxindong-multipod-dev"
GCS_PATCH_PATH="${GCS_BUCKET}/patches/tpu-inference.tar.gz"

WORKLOAD_NAME="wenxindongtest"

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

echo "=== Deleting existing workload '${WORKLOAD_NAME}' (if any) ==="
kubectl delete jobset "${WORKLOAD_NAME}" --ignore-not-found

echo "=== Applying ${YAML_FILE} ==="
kubectl apply -f "${YAML_FILE}"

echo "=== Done! Workload '${WORKLOAD_NAME}' deployed with latest local changes. ==="

echo "=== Waiting for pod to be created ==="
while true; do
  POD_NAME=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" -o name | head -n 1)
  if [ -n "${POD_NAME}" ]; then
    break
  fi
  sleep 2
done

echo "=== Following logs of pod ${POD_NAME} ==="
kubectl logs -f "${POD_NAME}" -c jax-tpu

