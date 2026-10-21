#!/bin/bash
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

set -euo pipefail

# Every docker removal below is best-effort and tolerates failure.
TARGET_IMAGES=("vllm-tpu" "vllm-tpu-local" "vllm-tpu-ci" "vllm-tpu-dev" "vllm-tpu-prod")

echo "=== Starting Docker resource cleanup ==="
echo "Target images: ${TARGET_IMAGES[*]}"

for IMG in "${TARGET_IMAGES[@]}"; do
  # Not "---...": Buildkite reads a leading `---` as a log group header.
  echo "========================================"
  echo "Cleaning up images matching: ${IMG}"

  OLD_IMAGES=$(docker images --format '{{.Repository}} {{.ID}}' | awk -v img="${IMG}" '$1 == img || $1 ~ "/"img"$" {print $2}' | sort -u)

  if [[ -n "$OLD_IMAGES" ]]; then
    echo "Found matching images. Checking for dependent containers..."

    TOTAL_CONTAINERS=""
    for img_id in $OLD_IMAGES; do
      TOTAL_CONTAINERS="$TOTAL_CONTAINERS $(docker ps -a -q --filter "ancestor=$img_id")"
    done

    CLEANED_CONTAINERS=$(echo "$TOTAL_CONTAINERS" | tr ' ' '\n' | grep -v '^$' | sort -u || true)
    if [[ -n "$CLEANED_CONTAINERS" ]]; then
      echo "Removing leftover containers using ${IMG} image(s)..."
      echo "$CLEANED_CONTAINERS" | xargs -r docker rm -f || true
    fi

    echo "Removing old ${IMG} image(s)..."
    echo "$OLD_IMAGES" | xargs -r docker rmi -f >/dev/null 2>&1 || true
  else
    echo "No images matching ${IMG} found to clean up."
  fi
done

echo "========================================"
echo "Pruning old Docker build cache..."
docker builder prune -f >/dev/null 2>&1 || true

echo "=== Cleanup complete ==="
