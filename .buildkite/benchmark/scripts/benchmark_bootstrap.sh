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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# Resolve the absolute directory path of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared pipeline config file.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../scripts/configs/pipeline_config.sh"

# Read CASE_TYPE from the first argument or default to DAILY.
BM_CASE_TYPE="${BM_CASE_TYPE:-DAILY}"
DEPENDENCY_STEP="${1:-}"

# Validate BM_CASE_TYPE input.
case "${BM_CASE_TYPE}" in
    DAILY|HOURLY|CI|DEV)
        ;;
    *)
        echo "🚨 Error: Invalid BM_CASE_TYPE '${BM_CASE_TYPE}'. Allowed values are DAILY, HOURLY, CI, DEV." >&2
        exit 1
        ;;
esac

JOB_PRIORITY="$PRIORITY_BENCHMARK"
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"

TIMEZONE="America/Los_Angeles"
JOB_REFERENCE="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
buildkite-agent meta-data set "JOB_REFERENCE" "${JOB_REFERENCE}"

# Check whether the RUN_TYPE is set to POST_KERNEL_AUTOTUNE_BM_RERUN, if so checkout the tuned parameter patched branch
if [[ "${RUN_TYPE:-}" == "POST_KERNEL_AUTOTUNE_BM_RERUN" ]]; then
    # use the KERNEL_AUTOTUNE_ID from the EXTRA_ENVS and construct the branch name and checkout the remote branch for the kernel autotune result evaluation. 
    # This is to ensure that the benchmark runs with the correct tuned parameters.
    # Extract the KERNEL_AUTOTUNE_ID from the EXTRA_ENVS
    # Example EXTRA_ENVS="KERNEL_AUTOTUNE_ID=$$KERNEL_AUTOTUNE_ID"
    KERNEL_AUTOTUNE_ID=$(echo "$EXTRA_ENVS" | tr ' ,' '\n' | grep '^KERNEL_AUTOTUNE_ID=' | cut -d= -f2-)
    if [ -z "$KERNEL_AUTOTUNE_ID" ]; then
        echo "Error: KERNEL_AUTOTUNE_ID is not set in EXTRA_ENVS."
        exit 1
    fi
    # Construct the branch name, this should match the branch name used in the kernel_auto_tune_invoker.py
    BRANCH_NAME="kernel_autotune.update_tuned_params_${KERNEL_AUTOTUNE_ID}"
    git checkout "${BRANCH_NAME}"
    COMMIT_MESSAGE=$(git log -1 --pretty=%B)
    echo "🚀 Running in POST_KERNEL_AUTOTUNE_BM_RERUN mode in branch ${BRANCH_NAME}"
    echo "Last commit message: ${COMMIT_MESSAGE}"
fi

upload_benchmark_pipeline() {
    local dependency_step="${1:-}"
    local target_case_type="$BM_CASE_TYPE"

    VLLM_COMMIT_HASH=${VLLM_COMMIT_HASH:-$(get_vllm_commit_hash)}
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    TPU_COMMIT_HASH=$(git rev-parse HEAD)
    CODE_HASH="${VLLM_COMMIT_HASH}-${TPU_COMMIT_HASH}-"
    buildkite-agent meta-data set "CODE_HASH" "${CODE_HASH}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    echo "Using vllm-tpu commit hash: $(buildkite-agent meta-data get "CODE_HASH")"

    # Convert uppercase target_case_type to lowercase for the directory path.
    local folder_name="${target_case_type,,}"
    # Set benchmark cases directory dynamically based on target_case_type.
    local case_folder=".buildkite/benchmark/cases/${folder_name}"
    local generator_script="${SCRIPT_DIR}/generate_bk_pipeline.py"
    process_json_benchmark_cases "$case_folder" "$generator_script" "$JOB_PRIORITY" "" "$dependency_step"

    # If the folder is not empty, upload the docker build pipeline.
    if [ "$(ls -A "$case_folder" 2>/dev/null)" ]; then
        # Since Buildkite inserts steps in reverse order, uploading this last 
        # ensures the Docker build steps appear at the very top of the UI.
        upload_pipeline_build_once "$JOB_PRIORITY"
    else
        echo "No benchmark cases found in $case_folder. Skipping docker build."
    fi
}

upload_benchmark_pipeline "$DEPENDENCY_STEP"
