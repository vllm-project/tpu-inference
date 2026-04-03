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
YAML_FILE="${SCRIPT_DIR}/pathways_job_new_cluster.yaml"

GCS_BUCKET="gs://wenxindong-multipod-dev"
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

WORKER_JOB="${JOBSET_NAME}-worker-0"
HEAD_JOB="${JOBSET_NAME}-pathways-head-0"

echo "=== Waiting for jobs to be created ==="
until kubectl get job "${WORKER_JOB}" &>/dev/null && kubectl get job "${HEAD_JOB}" &>/dev/null; do
  echo "  Waiting for jobs to appear..."
  sleep 2
done

echo "=== Adding Kueue management labels ==="
kubectl label job "${WORKER_JOB}" kueue.x-k8s.io/managed=true --overwrite
kubectl label job "${HEAD_JOB}" kueue.x-k8s.io/managed=true --overwrite

echo "=== Waiting for workload to be admitted ==="
WORKLOAD_NAME=""
# Discover the workload name (jobset-<JOBSET_NAME>-<hash>)
until [[ -n "${WORKLOAD_NAME}" ]]; do
  WORKLOAD_NAME=$(kubectl get workloads -o custom-columns=NAME:.metadata.name --no-headers | grep "^jobset-${JOBSET_NAME}-" | head -1)
  if [[ -z "${WORKLOAD_NAME}" ]]; then
    echo "  Waiting for workload to appear..."
    sleep 2
  fi
done
echo "  Found workload: ${WORKLOAD_NAME}"

# Wait for the workload to be admitted
until [[ "$(kubectl get workload "${WORKLOAD_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" == "True" ]]; do
  echo "  Waiting for workload '${WORKLOAD_NAME}' to be admitted..."
  sleep 5
done
echo "  Workload admitted!"

echo "=== Unsuspending jobs ==="
kubectl patch job "${WORKER_JOB}" --type=merge -p '{"spec":{"suspend":false}}'
kubectl patch job "${HEAD_JOB}" --type=merge -p '{"spec":{"suspend":false}}'

echo "=== Waiting for head pod to be created ==="
while true; do
  POD_NAME=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" -o name | head -n 1)
  if [[ -n "${POD_NAME}" ]]; then
    break
  fi
  echo "  Waiting for head pod to appear..."
  sleep 2
done

echo "=== Waiting for container 'jax-tpu' to start in ${POD_NAME} ==="
kubectl wait --for=condition=Ready "${POD_NAME}" --timeout=600s 2>/dev/null || true
sleep 3

echo "=== Following logs of ${POD_NAME} -c jax-tpu ==="
kubectl logs -f "${POD_NAME}" -c jax-tpu
