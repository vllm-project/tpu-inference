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
kernel_auto_tune_mapping["mla_kernel_tuner"]="/workspace/tpu_inference/tpu_inference/kernels/mla/v2/tuned_params.py"

# Path to the script you want to append
UPDATE_SCRIPT="/workspace/tpu_inference/tools/kernel/tuner/v1/autotune/update_tuned_params.py"

# Iterate over all keys in the dictionary
for key in "${!kernel_auto_tune_mapping[@]}"; do
    target_file="${kernel_auto_tune_mapping[$key]}"
    
    echo "Processing $target_file with tuner: $key..."

    # Step 1: Replace 'get_tuned_params' with '_get_tuned_params'
    sed -i 's/get_tuned_params/_get_tuned_params/g' "$target_file"

    # Step 2: Concatenate the update script to the end
    # (Adding an empty echo first to guarantee we start on a fresh line)
    echo "" >> "$target_file"
    cat "$UPDATE_SCRIPT" >> "$target_file"

    # Steps 3, 4, and 5: Replace the placeholders
    # Note: We use '|' as the sed delimiter instead of the standard '/' 
    # just in case your $DEVICE environment variable contains slashes.
    sed -i "s|KERNEL_TUNER_NAME_PLACEHOLDER|$key|g; \
            s|CASE_SET_ID_PLACEHOLDER|$KERNEL_AUTOTUNE_ID|g; \
            s|TPU_PLACEHOLDER|$DEVICE|g" "$target_file"

    echo "Successfully updated $target_file"
    echo "----------------------------------------"
done