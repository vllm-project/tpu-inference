#!/bin/bash
# Copyright 2025 Google LLC
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

# The "nightly_results_recorded" barrier. Sourced by
# upload_models_and_features.sh.
#
# nightly_verify.yml's support-matrix and "Notify test results" steps used to
# be ordered only by its `wait`. A wait is satisfied once, and jobs blocked by
# a failed dependency count as finished, so when a Docker image build failed
# the matrices and Notify ran at once on empty results. After the image was
# retried and the tests passed, nothing ran them again, and the build stayed
# FAILED until someone retried Notify by hand, after the last test (seen in
# tpu-inference-ci #1675).
#
# The barrier depends on the image builds (hard) and on every Record step
# uploaded (allow_failure, so a failed test is still reported).
# check_metadata_health depends on the barrier, so the matrices and Notify run
# once, after the last Record. When an image build fails, the barrier waits
# instead of failing; retrying the image re-runs the tests, their Records, the
# barrier and Notify, with no further retries.
#
# Buildkite behaviour this relies on (checked on tpu-inference-dev):
# - A job waiting on a failed hard dependency stays waiting_failed, and runs
#   when that dependency is retried and passes.
# - An allow_failure dependency on a step whose job is waiting again after a
#   retry waits for it.
# - A depends_on key may be uploaded later than the step naming it, and a step
#   skipped by `if:` satisfies an allow_failure dependency.

NIGHTLY_RESULTS_BARRIER_KEY="nightly_results_recorded"
RESULT_RECORD_KEYS=()
RESULTS_BARRIER_UPLOADED=0

# Reads uploaded pipeline YAML on stdin and remembers the keys of its Record
# steps, with ${TPU_VERSION:-<default>} resolved as the upload resolves it.
# Call it with the same TPU_VERSION as the upload.
add_record_keys() {
    local key ref default
    local re='\$\{TPU_VERSION:-([^}]*)\}'
    while IFS= read -r key; do
        while [[ "${key}" =~ ${re} ]]; do
            ref="${BASH_REMATCH[0]}"
            default="${BASH_REMATCH[1]}"
            key="${key//"${ref}"/${TPU_VERSION:-${default}}}"
        done
        if [[ "${key}" == *'$'* ]]; then
            echo "Warning: Record key '${key}' has an unresolved variable; the results barrier does not wait for it."
            continue
        fi
        RESULT_RECORD_KEYS+=("${key}")
    done < <(grep -E '^[[:space:]]*key:[[:space:]]*"?[^"]*record[^"]*"?[[:space:]]*$' \
        | sed -E 's/^[[:space:]]*key:[[:space:]]*"?//; s/"?[[:space:]]*$//' || true)
}

# Uploads the barrier once. Exits 1 if the upload fails, so "Upload Tests"
# turns red instead of leaving check_metadata_health waiting on a key that
# never arrives. Run as an EXIT trap: after a failed upload script (the ERR
# trap does not fire inside functions) it also sets stop_support_matrix, so
# check_metadata_health stops instead of building matrices from part of the
# suites.
upload_results_barrier() {
    local status=$?
    [[ "${RESULTS_BARRIER_UPLOADED}" == "0" ]] || return 0
    RESULTS_BARRIER_UPLOADED=1
    if [[ "${status}" -ne 0 ]]; then
        buildkite-agent meta-data set "stop_support_matrix" "true" || true
    fi
    local key
    local -a keys=()
    if [[ "${#RESULT_RECORD_KEYS[@]}" -gt 0 ]]; then
        mapfile -t keys < <(printf '%s\n' "${RESULT_RECORD_KEYS[@]}" | sort -u)
    fi
    echo "--- :checkered_flag: Uploading ${NIGHTLY_RESULTS_BARRIER_KEY} (waits for ${#keys[@]} Record steps)"
    {
        echo "priority: ${JOB_PRIORITY:-1}"
        echo "steps:"
        echo "  - label: \":checkered_flag: All suite results recorded\""
        echo "    key: \"${NIGHTLY_RESULTS_BARRIER_KEY}\""
        echo "    depends_on:"
        echo "      - \"build_docker_tpu6e\""
        echo "      - \"build_docker_tpu7x\""
        for key in "${keys[@]}"; do
            echo "      - step: \"${key}\""
            echo "        allow_failure: true"
        done
        echo "    agents:"
        echo "      queue: cpu"
        echo "    command: echo \"All ${#keys[@]} Record steps have finished.\""
    } | buildkite-agent pipeline upload || {
        echo "ERROR: could not upload ${NIGHTLY_RESULTS_BARRIER_KEY}."
        exit 1
    }
}
