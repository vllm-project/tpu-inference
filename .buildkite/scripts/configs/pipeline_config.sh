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
# On-call fix > Post-merge > Pre-merge > Integration pipeline > Benchmark > Other/Default > Nightly
# vllm-torchtpu uses 12 for on-call fixes too, so an on-call fix in either repo
# runs ahead of both repos' post-merge builds on the shared TPU queues.
export PRIORITY_ONCALL_FIX=12
export PRIORITY_POST_MERGE=10
export PRIORITY_PRE_MERGE=5
export PRIORITY_INTEGRATION=3
export PRIORITY_BENCHMARK=2
export PRIORITY_DEFAULT=1
export PRIORITY_NIGHTLY=0
export PRIORITY_KERNEL_TUNING=-10

# How many TPU jobs on the kube fleet one author's pull-request builds may have
# in flight at once, waiting in Kueue or running. The rest wait in Buildkite as
# `limited`, holding no agent pod and no place in Kueue's queue, so one author's
# burst of PRs does not queue ahead of everyone else's. A PR build runs about
# 19, so this is two PRs at a time; 40 capped 6 of 45 authors in the week to
# 10-16. Nightly, post-merge and release builds are not PR builds and have no
# cap.
export PR_TPU_JOBS_PER_AUTHOR=40

# Implemented dynamic job prioritization by injecting integers during upload
upload_with_priority() {
  local yaml_file=$1
  local JOB_PRIORITY=$2
  local capped
  echo "--- :pipeline: Uploading $yaml_file with priority ${JOB_PRIORITY:-$PRIORITY_DEFAULT}"
  # A pull-request build's TPU steps go in its author's concurrency group (set
  # by bootstrap.sh). The file is read once by the agent, so that the group is
  # added to the steps it would upload, anchors and interpolation resolved,
  # then uploaded as is. If any of that fails the file goes up uncapped rather
  # than failing the build.
  if [[ -n "${PR_TPU_CONCURRENCY_GROUP:-}" ]]; then
    if capped=$({
        echo "priority: ${JOB_PRIORITY:-$PRIORITY_DEFAULT}";
        cat "$yaml_file";
      } | buildkite-agent pipeline upload --dry-run --format json \
        | python3 "$(dirname "${BASH_SOURCE[0]}")/../cap_pr_tpu_concurrency.py" \
            "$PR_TPU_CONCURRENCY_GROUP" "$PR_TPU_JOBS_PER_AUTHOR"); then
      buildkite-agent pipeline upload --no-interpolation <<< "$capped"
      return
    fi
    echo "WARNING: could not put $yaml_file in ${PR_TPU_CONCURRENCY_GROUP}; uploading it uncapped" >&2
  fi
  {
    echo "priority: ${JOB_PRIORITY:-$PRIORITY_DEFAULT}";
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
  local extra_files="${4:-}" # Optional: newline-separated list of files
  local error_msgs=()

  echo "--- Generating dynamic pipelines from $case_folder and $extra_files"

  # Helper function to process a single benchmark file. 
  # Any failure (generation or upload) is captured in error_msgs.
  _process_benchmark_file() {
    local f="$1"
    local no_verify="${2:-false}"

    if [ ! -f "$f" ]; then
      # If the file does not exist (e.g., deleted in the current PR), skip it gracefully.
      echo "--- Skipping non-existent file (might be deleted): $f"
      return
    fi

    echo "--- Processing case file: $f (no-verify: $no_verify)"
    
    local py_output
    # 1. Generate pipeline and capture output/exit code (including stderr)
    if ! py_output=$(python3 "$generator" --input "$f" --no-verify "$no_verify" 2>&1); then
      echo "🚨 Generator failed for $f"
      error_msgs+=("❌ Generation Failure in $f:\n$py_output")
      return
    fi

    # 2. Upload the captured YAML (stdout from python)
    # A regular file, not <(...): upload_with_priority may read it twice (cap fallback).
    local tmp_yaml
    tmp_yaml=$(mktemp)
    printf '%s\n' "$py_output" > "$tmp_yaml"
    if ! upload_with_priority "$tmp_yaml" "$priority"; then
      rm -f "$tmp_yaml"
      echo "🚨 Upload failed for $f"
      error_msgs+=("❌ Upload Failure for $f")
      return
    fi
    rm -f "$tmp_yaml"
  }

  # 1. Process files from case_folder (Baseline: non-blocking business validation)
  if [ -n "$case_folder" ] && [ -d "$case_folder" ]; then
    shopt -s nullglob
    local folder_files=("$case_folder"/*.json)
    for f in "${folder_files[@]}"; do
      # Skip files that are explicitly listed in extra_files to avoid duplicate uploads
      # and ensure they are processed with mandatory verification.
      if [[ "$extra_files" == *"$f"* ]]; then
        echo "--- Skipping $f in folder pass (will be verified in extra_files pass)"
        continue
      fi
      _process_benchmark_file "$f" "true"
    done
  fi

  # 2. Process extra_files (Target: with full mandatory validation)
  if [ -n "$extra_files" ]; then
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      _process_benchmark_file "$f" "false"
    done <<< "$extra_files"
  fi

  # 3. If any errors occurred (in either pass), upload ONE aggregate failure step to block CI
  if [ ${#error_msgs[@]} -gt 0 ]; then
    local final_report
    final_report=$(printf "%b\n\n" "${error_msgs[@]}")
    upload_blocking_failure_step "❌ Benchmark Pipeline Generation Failures" "$final_report"
  fi
}
