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

#
# This File host the environment variables related to kernel tuning pipeline
# when the pipeline is invoked through Buildkite. 
#

# Required environment variables for kernel tuning pipeline.
KERNEL_TUNING_ENV_VARS=(
    -e KERNEL_TUNING_CASE_SET_ID="${KERNEL_TUNING_CASE_SET_ID:-}"
    -e KERNEL_TUNING_RUN_ID="${KERNEL_TUNING_RUN_ID:-}"
    -e KERNEL_TUNING_KERNEL_NAME="${KERNEL_TUNING_KERNEL_NAME:-}"
    -e KERNEL_TUNING_TPU_VERSION="${KERNEL_TUNING_TPU_VERSION:-}"
    -e KERNEL_TUNING_TPU_CORES="${KERNEL_TUNING_TPU_CORES:-}"
)

# Validation logic
for entry in "${KERNEL_TUNING_ENV_VARS[@]}"; do
    # Extract the variable name from the '-e NAME=VALUE' string
    # This pattern removes everything before '=' and the '-e ' prefix
    var_name=$(echo "$entry" | sed 's/^-e //;s/=.*//')
  
    # Check if the variable is defined and not empty
    if [[ -z "${!var_name}" ]]; then
        echo "Error: Required environment variable $var_name is not defined or is empty."
        exit 1
    fi
          if [ -z "${KERNEL_TUNING_RUN_ID:-}" ] || [ -z "${KERNEL_TUNING_CASE_SET_ID:-}" ]; then
            echo "Error: KERNEL_TUNING_RUN_ID or KERNEL_TUNING_CASE_SET_ID not set"
            exit 1
          fi
done

# Optional environment variables that can be added if they are set in the environment
KERNEL_TUNING_ENV_VARS+=(
    -e KERNEL_TUNING_CASE_SET_DESC="${KERNEL_TUNING_CASE_SET_DESC:-}"
    -e KERNEL_TUNING_JOB_PRIORITY="${PRIORITY_KERNEL_TUNING:--10}"
    -e KERNEL_TUNING_MAX_EXECUTION_MINUTES="${KERNEL_TUNING_MAX_EXECUTION_MINUTES:-20}"
)

# Creates a string of just the variable names (e.g., "KERNEL_TUNING_CASE_SET_ID KERNEL_TUNING_RUN_ID")
EXISTING_VARS=$(echo "${KERNEL_TUNING_ENV_VARS[@]}" | grep -oP 'KERNEL_TUNING_[A-Z_]+')

# Gather and append only if not already present, these are kernel specific environment variables
for var in "${!KERNEL_TUNING_@}"; do
    # Skip if the variable name is already in the EXISTING_VARS list
    if [[ "$EXISTING_VARS" =~ $var ]]; then
        continue
    fi

    # Ensure the variable is not empty before adding
    if [[ -n "${!var:-}" ]]; then
        KERNEL_TUNING_ENV_VARS+=(-e "${var}=${!var}")
    fi
done