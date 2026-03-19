#!/bin/bash
# Quick deploy script: uploads local tpu-inference changes to GCS and redeploys the Pathways job.
#
# Usage: ./deploy.sh
#
# This script:
#   1. Tars the local tpu-inference repo (excluding heavy/unnecessary dirs)
#   2. Uploads it to GCS
#   3. Deletes the existing JobSet
#   4. Applies the updated YAML
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML_FILE="${SCRIPT_DIR}/pathways_job.yaml"

GCS_BUCKET="gs://wenxindong-cloud-tpu-inference-test"
GCS_PATCH_PATH="${GCS_BUCKET}/patches/tpu-inference.tar.gz"

JOBSET_NAME="wenxindong-test"

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

echo "=== Deleting existing JobSet '${JOBSET_NAME}' (if any) ==="
kubectl delete jobset "${JOBSET_NAME}" --ignore-not-found

echo "=== Applying ${YAML_FILE} ==="
kubectl apply -f "${YAML_FILE}"

echo "=== Done! Job '${JOBSET_NAME}' deployed with latest local changes. ==="
