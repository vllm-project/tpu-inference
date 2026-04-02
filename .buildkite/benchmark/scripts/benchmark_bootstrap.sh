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

JOB_PRIORITY="$PRIORITY_BENCHMARK"
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"

TIMEZONE="America/Los_Angeles"
JOB_REFERENCE="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
buildkite-agent meta-data set "JOB_REFERENCE" "${JOB_REFERENCE}"

process_json_test_cases() {
    local folder="$1"
    local generator="$2"
    local priority="$3"

    echo "--- Generating pipelines from $folder"

    # Check if directory is empty to avoid errors
    shopt -s nullglob
    local files=("$folder"/*.json)
    
    if [ ${#files[@]} -eq 0 ]; then
        echo "No JSON files found in $folder."
        return
    fi

    for case_file in "${files[@]}"; do
        case_name=$(basename "$case_file" .json)
        echo "Processing case file: $case_file"

        # Create a unique temp file to avoid collisions in parallel environments
        local tmp_yml
        tmp_yml="/tmp/${case_name}_$(date +%s).yml"
        
        # Run generator and check for success
        if python3 "$generator" --input "$case_file" > "$tmp_yml"; then
            upload_with_priority "$tmp_yml" "$priority"
        else
            echo "Error: Failed to generate pipeline for $case_file"
            rm -f "$tmp_yml"
            exit 1
        fi
        
        rm -f "$tmp_yml"
    done
}

upload_benchmark_pipeline() {
    VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    TPU_COMMIT_HASH=$(git rev-parse HEAD)
    CODE_HASH="${VLLM_COMMIT_HASH}-${TPU_COMMIT_HASH}-"
    buildkite-agent meta-data set "CODE_HASH" "${CODE_HASH}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    echo "Using vllm-tpu commit hash: $(buildkite-agent meta-data get "CODE_HASH")"

    # Upload benchmark pipelines
    local case_folder=".buildkite/benchmark/cases"
    local generator_script=".buildkite/benchmark/scripts/generate_bk_pipeline.py"
    process_json_test_cases "$case_folder" "$generator_script" "$JOB_PRIORITY"

    # upload_with_priority .buildkite/benchmark/cases/benchmark_dev_test_v7x.yml "$JOB_PRIORITY"
}

upload_benchmark_pipeline
