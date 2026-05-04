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

# Validate BM_CASE_TYPE input.
case "${BM_CASE_TYPE}" in
    DAILY|HOURLY|CI)
        ;;
    *)
        echo "🚨 Error: Invalid BM_CASE_TYPE '${BM_CASE_TYPE}'. Allowed values are DAILY, HOURLY, CI." >&2
        exit 1
        ;;
esac

JOB_PRIORITY="$PRIORITY_BENCHMARK"
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"

TIMEZONE="America/Los_Angeles"
JOB_REFERENCE="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
buildkite-agent meta-data set "JOB_REFERENCE" "${JOB_REFERENCE}"

upload_benchmark_pipeline() {
    local target_case_type="$BM_CASE_TYPE"

    VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
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
    process_json_benchmark_cases "$case_folder" "$generator_script" "$JOB_PRIORITY"
}

upload_benchmark_pipeline
