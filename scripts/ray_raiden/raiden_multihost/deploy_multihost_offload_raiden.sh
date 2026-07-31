#!/usr/bin/env bash
# Copyright 2026 Google LLC
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

# Deploy Raiden-accelerated Multihost (no disagg) KV Cache Offloading on piv-cluster-europe
set -euo pipefail

CLUSTER="piv-cluster-europe"
PROJECT="cloud-tpu-inference-test"
ZONE="europe-west4-a"
KUBE_CONTEXT="gke_${PROJECT}_${ZONE}_${CLUSTER}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/multihost-serving-v6-offload.yaml"
MODEL_NAME="Qwen/Qwen3-8B"
IMAGE_TAG="us-east5-docker.pkg.dev/cloud-tpu-inference-test/piv/vllm-tpu-raiden:latest"

# Set BUILD_IMAGE=true to build and push Dockerfile before deploying
if [ "${BUILD_IMAGE:-false}" = "true" ]; then
  echo "=== Step 0: Building & pushing custom image locally (${IMAGE_TAG}) ==="
  gcloud auth configure-docker us-east5-docker.pkg.dev --quiet 2>/dev/null || true
  ADC_FILE="${HOME}/.config/gcloud/application_default_credentials.json"
  if [ ! -f "${ADC_FILE}" ]; then
    echo "Running gcloud auth application-default login..."
    gcloud auth application-default login
  fi
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  DOCKER_BUILDKIT=1 docker build --secret "id=adc,src=${ADC_FILE}" -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"
  docker push "${IMAGE_TAG}"
fi

echo "=== Step 1: Connecting to cluster ${CLUSTER} (${PROJECT} / ${ZONE}) ==="
gcloud container clusters get-credentials "${CLUSTER}" \
  --project="${PROJECT}" \
  --zone="${ZONE}"

echo "=== Step 2: Deleting old Raiden serving & disagg pods and waiting for cleanup ==="
kubectl --context="${KUBE_CONTEXT}" delete lws vllm-serving vllm-prefill vllm-decode --ignore-not-found=true
kubectl --context="${KUBE_CONTEXT}" delete deployment vllm-disagg-proxy --ignore-not-found=true
kubectl --context="${KUBE_CONTEXT}" delete pod -l llm-d.ai/inferenceServing=true --ignore-not-found=true
kubectl --context="${KUBE_CONTEXT}" delete pod -l app=vllm-proxy --ignore-not-found=true

# echo "=== Cleaning PVC storage (pvc-vllm-serving, pvc-vllm-p, pvc-vllm-d) ==="
# kubectl --context="${KUBE_CONTEXT}" delete pvc pvc-vllm-serving pvc-vllm-p pvc-vllm-d --ignore-not-found=true

echo "Waiting for old pods to be completely deleted..."
kubectl --context="${KUBE_CONTEXT}" wait --for=delete pod -l llm-d.ai/inferenceServing=true --timeout=300s 2>/dev/null || true
kubectl --context="${KUBE_CONTEXT}" wait --for=delete pod -l app=vllm-proxy --timeout=300s 2>/dev/null || true

echo "=== Step 3: Deploying Raiden Multihost Offloading Manifest (${MANIFEST_FILE}) ==="
kubectl --context="${KUBE_CONTEXT}" apply -f "${MANIFEST_FILE}"

echo "=== Step 4: Waiting for vLLM Raiden Multihost Engine ==="
START_TIME=$(date +%s)
while true; do
  ENGINE_COUNT=$(kubectl --context="${KUBE_CONTEXT}" logs vllm-serving-0 -c vllm-leader 2>/dev/null | grep -c -E "Application startup complete\." || true)

  if [ "$ENGINE_COUNT" -gt 0 ]; then
    echo "✅ vLLM Raiden Multihost Serving Engine is ready!"
    break
  fi

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo -ne "Waiting for Raiden engine... (${ELAPSED}s elapsed | engine_count: ${ENGINE_COUNT})\r"
  sleep 10
done
echo ""

echo "=== Step 5: Sending benchmark request to vLLM engine ==="
kubectl --context="${KUBE_CONTEXT}" exec vllm-serving-0 -c vllm-leader -- \
  curl -s "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"${MODEL_NAME}"'",
      "messages": [
        {"role": "user", "content": "Write a Python function to check if a number is prime."}
      ],
      "max_tokens": 100
    }' | jq .

echo "=== Deployment completed successfully! ==="
