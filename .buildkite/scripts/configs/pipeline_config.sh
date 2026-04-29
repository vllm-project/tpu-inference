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
export PRIORITY_KERNEL_TUNING=-1

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
