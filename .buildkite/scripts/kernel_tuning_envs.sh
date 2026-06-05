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

KERNEL_TUNING_ENV_VARS=(
    -e KERNEL_TUNING_CASE_SET_ID="${KERNEL_TUNING_CASE_SET_ID:-}"
    -e KERNEL_TUNING_RUN_ID="${KERNEL_TUNING_RUN_ID:-}"
    -e KERNEL_TUNING_KERNEL_NAME="${KERNEL_TUNING_KERNEL_NAME:-}"
    -e KERNEL_TUNING_CASE_SET_DESC="${KERNEL_TUNING_CASE_SET_DESC:-}"
    -e KERNEL_TUNING_TPU_VERSION="${KERNEL_TUNING_TPU_VERSION:-}"
    -e KERNEL_TUNING_TPU_CORES="${KERNEL_TUNING_TPU_CORES:-}"
    -e KERNEL_TUNING_JOB_PRIORITY="${PRIORITY_KERNEL_TUNING:--10}"
    -e KERNEL_TUNING_MAX_EXECUTION_MINUTES="${KERNEL_TUNING_MAX_EXECUTION_MINUTES:-20}"
)

# Add kernel tuner specific env vars below
VARS_TO_CHECK=(
    # MLA Kernel Tuning specific env vars
    KERNEL_TUNING_MLA_TOTAL_NUM_PAGES
    KERNEL_TUNING_MLA_PAGE_SIZE_PER_KV_PACKING
    KERNEL_TUNING_MLA_KV_PACKING
    KERNEL_TUNING_MLA_MAX_NUM_SEQS
    KERNEL_TUNING_MLA_PAGES_PER_SEQ
    KERNEL_TUNING_MLA_ACTUAL_NUM_Q_HEADS
    KERNEL_TUNING_MLA_ACTUAL_LKV_DIM
    KERNEL_TUNING_MLA_ACTUAL_R_DIM
    KERNEL_TUNING_MLA_KV_DTYPE
    KERNEL_TUNING_MLA_Q_DTYPE
    # Add more env vars here as needed
)

for var in "${VARS_TO_CHECK[@]}"; do
    # ${!var} is bash magic that gets the value of the variable name stored in 'var'
    if [[ -n "${!var:-}" ]]; then
        KERNEL_TUNING_ENV_VARS+=(-e "${var}=${!var}")
    fi
done