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

# Deploy Raiden-accelerated Disaggregated Serving on piv-cluster-europe
set -euo pipefail

CLUSTER="piv-cluster-europe"
PROJECT="cloud-tpu-inference-test"
ZONE="europe-west4-a"
KUBE_CONTEXT="gke_${PROJECT}_${ZONE}_${CLUSTER}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/disagg-serving-v6-raiden.yaml"
MODEL_NAME="Qwen/Qwen3-8B"
IMAGE_TAG="us-east5-docker.pkg.dev/cloud-tpu-inference-test/piv/vllm-tpu-raiden:latest"

# Set BUILD_IMAGE=true to build and push ray_lws_configs/Dockerfile via Cloud Build before deploying
if [ "${BUILD_IMAGE:-false}" = "true" ]; then
  echo "=== Step 0: Building custom image from ray_lws_configs/Dockerfile (${IMAGE_TAG}) ==="
  gcloud builds submit --tag "${IMAGE_TAG}" --dockerfile="${SCRIPT_DIR}/Dockerfile" "${WORKSPACE_ROOT}" --project="${PROJECT}"
fi

echo "=== Step 1: Connecting to cluster ${CLUSTER} (${PROJECT} / ${ZONE}) ==="
gcloud container clusters get-credentials "${CLUSTER}" \
  --project="${PROJECT}" \
  --zone="${ZONE}"

echo "=== Step 2: Deploying Raiden Disaggregated Serving Manifest (${MANIFEST_FILE}) ==="
kubectl --context="${KUBE_CONTEXT}" apply -f "${MANIFEST_FILE}"

echo "=== Step 3: Waiting for vLLM Raiden Prefill & Decode Engines ==="
START_TIME=$(date +%s)
while true; do
  PREFILL_COUNT=$(kubectl --context="${KUBE_CONTEXT}" logs vllm-prefill-0 2>/dev/null | grep -c -E "Uvicorn running on|Engine 000:" || true)
  DECODE_COUNT=$(kubectl --context="${KUBE_CONTEXT}" logs vllm-decode-0 2>/dev/null | grep -c -E "Uvicorn running on|Engine 000:" || true)

  if [ "$PREFILL_COUNT" -gt 0 ] && [ "$DECODE_COUNT" -gt 0 ]; then
    echo "✅ Both Prefill and Decode vLLM Raiden engines are ready!"
    break
  fi

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo -ne "Waiting for Raiden engines... (${ELAPSED}s elapsed | prefill_count: ${PREFILL_COUNT}, decode_count: ${DECODE_COUNT})\r"
  sleep 10
done
echo ""

echo "=== Step 4: Sending benchmark request through disaggregated proxy ==="
kubectl --context="${KUBE_CONTEXT}" exec deployment/vllm-disagg-proxy -- \
  curl -s "http://localhost:10000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"${MODEL_NAME}"'",
      "messages": [
        {"role": "user", "content": "Write a Python function to check if a number is prime."}
      ],
      "max_tokens": 100
    }' | jq .

echo "=== Deployment completed successfully! ==="
