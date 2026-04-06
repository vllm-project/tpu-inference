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
KUBE_CONTEXT="gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-alpha-cluster"

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
kubectl --context="${KUBE_CONTEXT}" delete jobset "${JOBSET_NAME}" --ignore-not-found

echo "=== Applying ${YAML_FILE} ==="
kubectl --context="${KUBE_CONTEXT}" apply -f "${YAML_FILE}"

# Kueue manages the lifecycle via the queue-name label on the worker Job.
# It handles admission, topology assignment, and unsuspension.
# The head job (CPU-only) has no Kueue label and is managed by the JobSet directly.

echo "=== Waiting for workload to be admitted by Kueue ==="
WORKLOAD_NAME=""
until [[ -n "${WORKLOAD_NAME}" ]]; do
  WORKLOAD_NAME=$(kubectl --context="${KUBE_CONTEXT}" get workloads -o custom-columns=NAME:.metadata.name --no-headers | grep "^jobset-${JOBSET_NAME}-" | head -1)
  if [[ -z "${WORKLOAD_NAME}" ]]; then
    echo "  Waiting for workload to appear..."
    sleep 2
  fi
done
echo "  Found workload: ${WORKLOAD_NAME}"

until [[ "$(kubectl --context="${KUBE_CONTEXT}" get workload "${WORKLOAD_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" == "True" ]]; do
  echo "  Waiting for workload '${WORKLOAD_NAME}' to be admitted..."
  sleep 5
done
echo "  Workload admitted!"

echo "=== Waiting for jobs to be unsuspended by Kueue ==="
WORKER_JOB="${JOBSET_NAME}-worker-0"
HEAD_JOB="${JOBSET_NAME}-pathways-head-0"
until [[ "$(kubectl --context="${KUBE_CONTEXT}" get job "${WORKER_JOB}" -o jsonpath='{.spec.suspend}' 2>/dev/null)" == "false" ]] && \
      [[ "$(kubectl --context="${KUBE_CONTEXT}" get job "${HEAD_JOB}" -o jsonpath='{.spec.suspend}' 2>/dev/null)" == "false" ]]; do
  echo "  Waiting for Kueue to unsuspend jobs..."
  sleep 5
done
echo "  Jobs unsuspended!"

echo "=== Waiting for head pod to be created ==="
while true; do
  POD_NAME=$(kubectl --context="${KUBE_CONTEXT}" get pods -l "jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" -o name | head -n 1)
  if [[ -n "${POD_NAME}" ]]; then
    break
  fi
  echo "  Waiting for head pod to appear..."
  sleep 2
done

echo "=== Waiting for container 'jax-tpu' to start in ${POD_NAME} ==="
kubectl --context="${KUBE_CONTEXT}" wait --for=condition=Ready "${POD_NAME}" --timeout=600s 2>/dev/null || true
sleep 3

echo "=== Following logs of ${POD_NAME} -c jax-tpu ==="
kubectl --context="${KUBE_CONTEXT}" logs -f "${POD_NAME}" -c jax-tpu
