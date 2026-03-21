#!/bin/bash
# Quick deploy script: uploads local tpu-inference changes to GCS and redeploys
# the Pathways workload using xpk commands.
#
# Usage: ./deploy_xpk.sh
#
# This script:
#   1. Tars the local tpu-inference repo (excluding heavy/unnecessary dirs)
#   2. Uploads it to GCS
#   3. Deletes the existing xpk workload (if any)
#   4. Creates a new xpk Pathways workload
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── GCS settings ──
GCS_BUCKET="gs://wenxindong-multipod-dev"
GCS_PATCH_PATH="${GCS_BUCKET}/patches/tpu-inference.tar.gz"

# ── xpk / cluster settings ──
WORKLOAD_NAME="wenxindongtest"
DOCKER_IMAGE="vllm/vllm-tpu:nightly"
CLUSTER_NAME="mlperf-v5p-512"
TPU_TYPE="v5p-512"
NUM_SLICES=1
ZONE="europe-west4"
PROJECT="cloud-tpu-multipod-dev"
PRIORITY="very-high"

# ── Command: download tpu-inference tar from GCS, extract, and run the entrypoint ──
read -r -d '' COMMAND << 'EOCMD' || true
gcloud storage cp gs://wenxindong-multipod-dev/patches/tpu-inference.tar.gz /tmp/tpu-inference.tar.gz

mkdir -p /tmp/tpu-inference
tar xzf /tmp/tpu-inference.tar.gz -C /tmp/tpu-inference

bash /tmp/tpu-inference/scripts/pathways/run_pathways_xpk.sh
EOCMD

# ── Package & upload ──
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

# ── Delete existing workload (if any) ──
echo "=== Deleting existing workload '${WORKLOAD_NAME}' (if any) ==="
xpk workload delete \
  --workload "${WORKLOAD_NAME}" \
  --cluster "${CLUSTER_NAME}" \
  --zone "${ZONE}" \
  --project "${PROJECT}" \
  || true   # don't fail if the workload doesn't exist

# ── Create new Pathways workload ──
echo "=== Creating Pathways workload '${WORKLOAD_NAME}' ==="
xpk workload create-pathways \
  --workload "${WORKLOAD_NAME}" \
  --docker-image "${DOCKER_IMAGE}" \
  --cluster "${CLUSTER_NAME}" \
  --tpu-type="${TPU_TYPE}" \
  --num-slices="${NUM_SLICES}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --priority="${PRIORITY}" \
  --command "${COMMAND}"

echo "=== Done! Workload '${WORKLOAD_NAME}' deployed with latest local changes. ==="
