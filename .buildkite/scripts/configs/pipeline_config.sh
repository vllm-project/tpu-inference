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

# Priority constants for pipeline jobs.
# Post-merge > Pre-merge > Integration pipeline > Benchmark > Other/Default > Nightly
export PRIORITY_POST_MERGE=10
export PRIORITY_PRE_MERGE=5
export PRIORITY_INTEGRATION=3
export PRIORITY_BENCHMARK=2
export PRIORITY_DEFAULT=1
export PRIORITY_NIGHTLY=0
export PRIORITY_KERNEL_TUNING=-10

# Implemented dynamic job prioritization by injecting integers during upload
upload_with_priority() {
  local yaml_file=$1
  local JOB_PRIORITY=$2
  echo "--- :pipeline: Uploading $yaml_file with priority ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
  { 
    echo "priority: ${JOB_PRIORITY:-PRIORITY_DEFAULT}"; 
    cat "$yaml_file"; 
  } | buildkite-agent pipeline upload
}

get_vllm_commit_hash() {
  # load vllm commit hash from vllm_lkg.version file, if not exists, get the latest commit hash from vllm repo
  local commit_hash=""
  local version_file=".buildkite/vllm_lkg.version"

  if [ -f "$version_file" ]; then
    commit_hash="$(cat "$version_file")"
  fi
  if [ -z "${commit_hash:-}" ]; then
    commit_hash=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
  fi

  echo "$commit_hash"
}

# Function to upload a light-weight step that intentionally fails to block PR merge
upload_blocking_failure_step() {
  local label="$1"
  local detailed_message="$2"
  local artifact_name="benchmark_errors.log"

  echo "🚨 Error: $label"
  
  # Write detailed errors to a file and upload as an artifact
  echo "$detailed_message" > "$artifact_name"
  buildkite-agent artifact upload "$artifact_name"

  cat <<- YAML | buildkite-agent pipeline upload
steps:
  - label: "$label"
    agents:
      queue: cpu
    command: |
      echo "=========================================================="
      echo "❌ Benchmark Pipeline Generation/Validation Failures"
      echo "=========================================================="
      echo "Detailed error log is available in Buildkite Artifacts: $artifact_name"
      echo ""
      buildkite-agent artifact download "$artifact_name" .
      cat "$artifact_name"
      exit 1
YAML
}

# Function to process JSON benchmark cases from a folder and/or a list of specific files
process_json_benchmark_cases() {
  local case_folder="${1:-}"
  local generator="${2:-}"
  local priority="${3:-}"
  local extra_files="${4:-}" # Optional: space-separated list of files
  local error_msgs=()

  echo "--- Generating dynamic pipelines from $case_folder and $extra_files"

  shopt -s nullglob
  local files=()
  
  # Add files from folder if provided
  if [ -n "$case_folder" ] && [ -d "$case_folder" ]; then
    files+=("$case_folder"/*.json)
  fi

  # Add extra files if provided
  if [ -n "$extra_files" ]; then
    for f in $extra_files; do
      # Skip deleted files in PR to avoid generation errors
      if [ -f "$f" ]; then
        files+=("$f")
      fi
    done
  fi

  if [ ${#files[@]} -eq 0 ]; then
    echo "No JSON files found to process (Folder: $case_folder, Extra: $extra_files)."
    return
  fi

  echo "Found ${#files[@]} files to process."

  for case_file in "${files[@]}"; do
    echo "Processing case file: $case_file"

    # 1. Generate pipeline and capture output/exit code (including stderr)
    local py_output
    if ! py_output=$(python3 "$generator" --input "$case_file" 2>&1); then
      echo "🚨 Validation failed for $case_file"
      error_msgs+=("❌ Validation Failure in $case_file:\n$py_output")
      continue
    fi

    # 2. Upload the captured YAML (stdout from python)
    if ! upload_with_priority <(echo "$py_output") "$priority"; then
      echo "🚨 Upload failed for $case_file"
      error_msgs+=("❌ Upload Failure for $case_file")
    fi
  done

  # 3. If any errors occurred, upload ONE aggregate failure step to block CI
  if [ ${#error_msgs[@]} -gt 0 ]; then
    local final_report
    final_report=$(printf "%b\n\n" "${error_msgs[@]}")
    upload_blocking_failure_step "❌ Benchmark Pipeline Generation Failures" "$final_report"
  fi
}
