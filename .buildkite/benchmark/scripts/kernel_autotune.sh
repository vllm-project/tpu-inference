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
declare -A kernel_autotune_mapping
# Load the mapping dynamically from the Python config file to ensure they stay in sync
while IFS="=" read -r key value; do
    kernel_autotune_mapping["$key"]="$value"
done < <(PYTHONPATH=. python3 -c "
from tools.kernel.tuner.v1.autotune.kernel_autotune_config import kernel_autotune_mapping
for k, v in kernel_autotune_mapping.items():
    print(f'{k}={v}')
")

# Path to the script you want to append
KERNEL_TUNED_PARAMS_UPDATE_SCRIPT="/workspace/tpu_inference/tools/kernel/tuner/v1/autotune/update_tuned_params_template.py"

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
    if [[ "${KERNEL_AUTOTUNE_STAGE:-}" == "PRE_KERNEL_AUTOTUNE_CASES_COLLECTION" ]]; then
        # Validate all target tuned_params.py files have the expected structure
        # (def get_tuned_params, tuned_params_mapping dict, no _get_tuned_params)
        # before any patching begins. Reuses the Python test to avoid duplicating logic.
        echo "--- Validating target tuned_params.py file structure..."
        python3 -m pytest tools/kernel/tuner/v1/tests/test_tuned_params_structure.py -v
        echo "✅ All target files validated successfully."

        # Iterate over all keys in the dictionary
        for key in "${!kernel_autotune_mapping[@]}"; do
            target_file="${kernel_autotune_mapping[$key]}"

            echo "Processing $target_file with tuner: $key..."

            # Step 1: Rename the function 'get_tuned_params' → '_get_tuned_params'
            # Uses 'def ' prefix to only match the function definition, not calls.
            sed -i 's/def get_tuned_params/def _get_tuned_params/g' "$target_file"

            # Step 2: Concatenate the update script to the end
            # (Adding an empty echo first to guarantee we start on a fresh line)
            echo "" >> "$target_file"
            cat "$KERNEL_TUNED_PARAMS_UPDATE_SCRIPT" >> "$target_file"

            tpu="$(get_tpu_from_device)"

            # Verify all placeholders are present before substitution.
            for placeholder in KERNEL_TUNER_NAME_PLACEHOLDER CASE_SET_ID_PLACEHOLDER TPU_PLACEHOLDER; do
                if ! grep -q "$placeholder" "$target_file"; then
                    echo "Error: Placeholder '$placeholder' not found in $target_file. The template may not have been appended correctly." >&2
                    return 1
                fi
            done

            # Steps 3, 4, and 5: Replace the placeholders
            sed -i "s|KERNEL_TUNER_NAME_PLACEHOLDER|$key|g; \
                    s|CASE_SET_ID_PLACEHOLDER|$KERNEL_TUNING_AUTOTUNE_ID|g; \
                    s|TPU_PLACEHOLDER|$tpu|g" "$target_file"

            # Validate generated Python before claiming success.
            python3 -m py_compile "$target_file"

            echo "🚀 Successfully updated $target_file"
        done
    else
        echo "❌ Not in PRE_KERNEL_AUTOTUNE_CASES_COLLECTION stage, skipping update of tuned_params.py"
        exit 1
    fi
}

checkout_updated_tuned_params_py_branch() {
    if [[ "${KERNEL_AUTOTUNE_STAGE:-}" == "POST_KERNEL_AUTOTUNE_BM_RERUN" ]]; then
        if [ -z "$KERNEL_TUNING_AUTOTUNE_ID" ]; then
            echo "Error: KERNEL_TUNING_AUTOTUNE_ID is not set."
            exit 1
        fi
        # Construct the branch name which matches the one created by the autotune_result_processing.py
        BRANCH_NAME="kernel_autotune.update_tuned_params_${KERNEL_TUNING_AUTOTUNE_ID}"
        git fetch
        git checkout "${BRANCH_NAME}"
        COMMIT_MESSAGE=$(git log -1 --pretty=%B)
        echo "Last commit message: ${COMMIT_MESSAGE}"
        echo "🚀 POST_KERNEL_AUTOTUNE_BM_RERUN stage runs in branch ${BRANCH_NAME}"
    else
        echo "❌ Not in POST_KERNEL_AUTOTUNE_BM_RERUN stage, should not invoke this function"
        exit 1
    fi
}
