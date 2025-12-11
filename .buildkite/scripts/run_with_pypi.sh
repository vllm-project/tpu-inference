#!/bin/bash
# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# Get the nightly TPU_INFERENCE_VERSION based on the latest stable tag and current date.
LATEST_STABLE_TAG=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)
BASE_VERSION=${LATEST_STABLE_TAG#v}
DATETIME_STR=$(date -u +%Y%m%d)
TPU_INFERENCE_VERSION="${BASE_VERSION}.dev${DATETIME_STR}"
echo "Target Nightly Version: ${TPU_INFERENCE_VERSION}"

# Configuration to check if the version is available on PyPI
PACKAGE_NAME="tpu-inference"
MAX_RETRIES=20
SLEEP_SEC=60
FOUND_VERSION=false

echo "Checking PyPI for ${PACKAGE_NAME} == ${TPU_INFERENCE_VERSION}..."

# Retry logic to check if the version is available on PyPI
for ((i=1; i<=MAX_RETRIES; i++)); do
    if pip index versions "${PACKAGE_NAME}" --pre 2>/dev/null | grep -q "${TPU_INFERENCE_VERSION}"; then
        echo "Success! Found version ${TPU_INFERENCE_VERSION} on PyPI."
        FOUND_VERSION=true
        break
    fi

    echo "[Attempt $i/$MAX_RETRIES] Version not found yet. Waiting ${SLEEP_SEC} seconds..."
    if [ "$i" -lt "$MAX_RETRIES" ]; then
        sleep "$SLEEP_SEC"
    fi
done

if [ "$FOUND_VERSION" = "false" ]; then
    echo "The version ${TPU_INFERENCE_VERSION} was not found on PyPI."
    exit 1
fi

# Build vllm-tpu with nightly tpu-inference from PyPI (using docker/Dockerfile.pypi instead of docker/Dockerfile).
export RUN_WITH_PYPI="true"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_in_docker.sh"
