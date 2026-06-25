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

set -Eeuo pipefail

# Define the dictionary
declare -A kernel_auto_tune_mapping
# The key should be one of tools.kernel.tuner.v1.kernel_tuner_runner.KERNEL_TUNER_REGISTRY keys, and the value is the corresponding file path to be updated.
# This must be kept in sync with the tools.kernel.tuner.v1.autotune.kernel_auto_tune_config.kernel_auto_tune_mapping dictionary.
kernel_auto_tune_mapping["mla_kernel_tuner"]="/workspace/tpu_inference/tpu_inference/kernels/mla/v2/tuned_params.py"

# Path to the script you want to append
KERNEL_TUNED_PARAMS_UPDATE_SCRIPT="/workspace/tpu_inference/tools/kernel/tuner/v1/autotune/update_tuned_params.py"

get_tpu_from_device() {
    if [[ "$DEVICE" == *"6e"* ]]; then
        echo "tpu6e"
    elif [[ "$DEVICE" == *"7x"* ]]; then
        echo "tpu7x"
    else
        echo "Error: DEVICE must contain '6e' or '7x'" >&2
        exit 1
    fi
}

update_all_tuned_params_py() {
    # Iterate over all keys in the dictionary
    for key in "${!kernel_auto_tune_mapping[@]}"; do
        target_file="${kernel_auto_tune_mapping[$key]}"

        echo "Processing $target_file with tuner: $key..."

        # Step 1: Replace 'get_tuned_params' with '_get_tuned_params'
        sed -i 's/get_tuned_params/_get_tuned_params/g' "$target_file"

        # Step 2: Concatenate the update script to the end
        # (Adding an empty echo first to guarantee we start on a fresh line)
        echo "" >> "$target_file"
        cat "$KERNEL_TUNED_PARAMS_UPDATE_SCRIPT" >> "$target_file"

        tpu="$(get_tpu_from_device)"

        # Steps 3, 4, and 5: Replace the placeholders
        sed -i "s|KERNEL_TUNER_NAME_PLACEHOLDER|$key|g; \
                s|CASE_SET_ID_PLACEHOLDER|$KERNEL_AUTOTUNE_ID|g; \
                s|TPU_PLACEHOLDER|$tpu|g" "$target_file"

        echo "Successfully updated $target_file"
        echo "----------------------------------------"
    done
}

checkout_updated_tuned_params_py_branch() {
    # EXTRA_ENVS="KERNEL_AUTOTUNE_ID=$$KERNEL_AUTOTUNE_ID,KERNEL_AUTOTUNE_STAGE=POST_KERNEL_AUTOTUNE_BM_RERUN"
    # extract KERNEL_AUTOTUNE_STAGE from EXTRA_ENVS and set it as an environment variable
    if [[ -n "${EXTRA_ENVS:-}" ]]; then
        KERNEL_AUTOTUNE_STAGE=$(echo "$EXTRA_ENVS" | tr ' ,' '\n' | grep '^KERNEL_AUTOTUNE_STAGE=' | cut -d= -f2-)
        if [[ -z "${KERNEL_AUTOTUNE_STAGE}" ]]; then
            unset KERNEL_AUTOTUNE_STAGE
        else
            echo "🚀 KERNEL_AUTOTUNE_STAGE set to ${KERNEL_AUTOTUNE_STAGE}"
        fi
    fi
    if [[ "${KERNEL_AUTOTUNE_STAGE:-}" == "POST_KERNEL_AUTOTUNE_BM_RERUN" ]]; then
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
        git fetch
        git checkout "${BRANCH_NAME}"
        COMMIT_MESSAGE=$(git log -1 --pretty=%B)
        echo "🚀 Running in POST_KERNEL_AUTOTUNE_BM_RERUN mode in branch ${BRANCH_NAME}"
        echo "Last commit message: ${COMMIT_MESSAGE}"
    fi
}
