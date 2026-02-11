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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

BUILDKITE_DIR=".buildkite"
MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"

declare -a TARGET_FOLDERS=(
    "quantization"
    "parallelism"
    "models"
    "features"
)


# Use find to append the kernel_microbenchmarks subdirectories
KERNEL_PARENT_DIR=".buildkite/kernel_microbenchmarks"

if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        folder_path_to_add="${dir#"${BUILDKITE_DIR}"/}"
        TARGET_FOLDERS+=("$folder_path_to_add")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
else
    echo "Warning: Kernel microbenchmarks directory '$KERNEL_PARENT_DIR' not found. Skipping dynamic folder discovery."
fi

# Arrays to store YAML content fragments (without 'steps:' header)
pipeline_v6e_fragments=()
pipeline_v7x_fragments=()

# Declare separate arrays for each list
declare -a model_list
declare -a feature_list


for folder_path in "${TARGET_FOLDERS[@]}"; do
  folder=$BUILDKITE_DIR/$folder_path
  # Check if the folder exists
  if [[ ! -d "$folder" ]]; then
    echo "Warning: Folder '$folder' not found. Skipping."
    continue
  fi

  echo "Processing config ymls in ${folder}"

  # Use find command to locate all .yml or .yaml files
  # -print0 and read -r -d '' are a safe way to handle filenames with special characters (like spaces)
  while IFS= read -r -d '' yml_file; do
    echo "--- handling yml file: ${yml_file}"

    # Targeted Extraction: Find the line starting with '# pipeline-name:'
    subject_name_line=$(awk '/^# ?pipeline-name:/ {print $0; exit}' "${yml_file}")

    if [[ -n "$subject_name_line" ]]; then
      # Extract value: Remove everything up to and including the colon and optional space
      subject_name="${subject_name_line#*:[[:space:]]}"
      # Trim trailing whitespace/carriage returns
      subject_name="${subject_name%"${subject_name##*[![:space:]]}"}"

      case "$folder_path" in
        "models")
          model_list+=("$subject_name")
          ;;
        "features" | "parallelism" | "quantization" | "kernel_microbenchmarks"/*)
          feature_list+=("${subject_name}")
          ;;
      esac
    fi

    # Read the YAML file and strip the top-level 'steps:' line
    # This is required because we wrap them inside a 'group' later
    yml_content=$(grep -v "^steps:" "${yml_file}")

    # Store the content for both hardware types
    pipeline_v6e_fragments+=("${yml_content}")
    pipeline_v7x_fragments+=("${yml_content}")

  done < <(find "$folder" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done

# Convert array to a newline-separated string
model_list_string=$(printf "%s\n" "${model_list[@]}")
feature_list_string=$(printf "%s\n" "${feature_list[@]}")

if [[ -n "$model_list_string" ]]; then
  echo "${model_list_string}" | buildkite-agent meta-data set "${MODEL_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${MODEL_LIST_KEY}")"
fi

if [[ -n "$feature_list_string" ]]; then
  echo "${feature_list_string}" | buildkite-agent meta-data set "${FEATURE_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${FEATURE_LIST_KEY}")"
fi

# --- Upload Dynamic Pipeline ---
# Final Uploads (Two separate calls to handle variables) ---
if [[ "${#pipeline_v6e_fragments[@]}" -gt 0 ]]; then
  echo "--- Uploading TPU v6e Pipeline Group"
  buildkite-agent meta-data set "run_v6_matrix" "true"
  {
    echo "steps:"
    echo "  - group: \"TPU v6e nightly Tests (${MODEL_IMPL_TYPE:-auto})\""
    echo "    key: \"v6e-group\""
    echo "    steps:"
    printf "%s\n" "${pipeline_v6e_fragments[@]}" | sed 's/^/      /'
  } | buildkite-agent pipeline upload
else
  echo "--- No .yml files found, nothing to upload."
  exit 0
fi

if [[ "${#pipeline_v7x_fragments[@]}" -gt 0 ]]; then
  echo "--- Uploading TPU v7x Pipeline Group"
  # Export v7x specific variables (overwrites previous exports)
  export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
  export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
  export IS_FOR_V7X="true"
  export TPU_VERSION="tpu7x"
  buildkite-agent meta-data set "run_v7_matrix" "true"
  {
    echo "steps:"
    echo "  - group: \"TPU v7x nightly Tests (${MODEL_IMPL_TYPE:-auto})\""
    echo "    key: \"v7x-group\""
    echo "    steps:"
    printf "%s\n" "${pipeline_v7x_fragments[@]}" | sed 's/^/      /'
  } | buildkite-agent pipeline upload
else
  echo "--- No .yml files found, nothing to upload."
  exit 0
fi