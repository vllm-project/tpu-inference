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

# --- Skip build if only docs/icons changed ---
echo "--- :git: Checking changed files"

BASE_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-"main"}

if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
    echo "PR detected. Target branch: $BASE_BRANCH"

    # Fetch base and current commit to ensure local history exists for diff
    git fetch origin "$BASE_BRANCH" --depth=20 --quiet || echo "Base fetch failed"
    git fetch origin "$BUILDKITE_COMMIT" --depth=20 --quiet || true

    # Get all changes in this PR using triple-dot diff (common ancestor to HEAD)
    # This correctly captures changes even if the last commit is a merge from main
    FILES_CHANGED=$(git diff --name-only origin/"$BASE_BRANCH"..."$BUILDKITE_COMMIT" 2>/dev/null || true)

    # Fallback to single commit diff if PR history is unavailable
    if [ -z "$FILES_CHANGED" ]; then
        echo "Warning: PR diff failed. Falling back to single commit check."
        FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r -m "$BUILDKITE_COMMIT")
    fi
    
    echo "Files changed:"
    echo "$FILES_CHANGED"

    # Filter out files we want to skip builds for.
    NON_SKIPPABLE_FILES=$(echo "$FILES_CHANGED" | grep -vE "(\.md$|\.ico$|\.png$|^README$|^docs\/|support_matrices\/.*\.csv$)" || true)

    if [ -z "$NON_SKIPPABLE_FILES" ]; then
      echo "Only documentation or icon files changed. Skipping build."
      # No pipeline will be uploaded, and the build will complete.
      exit 0
    else
      echo "Code files changed. Proceeding with pipeline upload."
    fi

    # Validate modified YAML pipelines using bk pipeline validate
    if .buildkite/scripts/validate_all_pipelines.sh "$NON_SKIPPABLE_FILES"; then
      echo "All pipelines syntax are valid. Proceeding with pipeline upload."
    else
      echo "Some pipelines syntax are invalid. Failing build."
      exit 1
    fi
else
    echo "Non-PR build. Bypassing file change check."
fi

upload_pipeline() {
    if [ "${MODEL_IMPL_TYPE:-auto}" == "auto" ]; then
      # Upload JAX pipeline for v6 (default)
      buildkite-agent pipeline upload .buildkite/features/DCN-Based_P-D_disaggregation.yml
      buildkite-agent pipeline upload .buildkite/pipeline_jax.yml

      # Upload JAX pipeline for v7
      export TESTS_GROUP_LABEL="[jax] TPU7x Tests Group"
      export TPU_VERSION="tpu7x"
      export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
      export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
      export IS_FOR_V7X="true"
      export COV_FAIL_UNDER="67"
      buildkite-agent pipeline upload .buildkite/pipeline_jax.yml
      unset TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI IS_FOR_V7X COV_FAIL_UNDER

      # buildkite-agent pipeline upload .buildkite/pipeline_torch.yml
      buildkite-agent pipeline upload .buildkite/nightly_releases.yml
    fi

    buildkite-agent pipeline upload .buildkite/nightly_verify.yml
    buildkite-agent pipeline upload .buildkite/pipeline_pypi.yml
}

echo "--- Starting Buildkite Bootstrap"
echo "Running in pipeline: $BUILDKITE_PIPELINE_SLUG"

echo "Configure notification"
ONCALL_EMAIL="ullm-test-notifications-external@google.com"
NOTIFY_FILE="generated_notification.yml"

# Logic
# 1. Official Integration/Nightly: If it's triggered by schedule -> Notify Oncall & Slack.
# 2. Everything else (PRs, Manual Triggers): Notify the creator of this build.
#    - This ensures that if you manually trigger the integration pipeline for debugging, 
#      it won't alert the oncall team.

if [[ "$BUILDKITE_PIPELINE_SLUG" == "tpu-vllm-integration" && "$BUILDKITE_SOURCE" == "schedule" ]] || \
   [[ "${NIGHTLY:-0}" == "1" && "$BUILDKITE_SOURCE" == "schedule" ]]; then
    echo "Context: Scheduled Integration/Nightly. Notifying Oncall."
    cat <<EOF > "$NOTIFY_FILE"
notify:
  - email: "$ONCALL_EMAIL"
    if: build.state == "failed"
  - slack: "vllm#tpu-ci-notifications"
    if: build.state == "failed"
EOF
else
    echo "Context: PR/Manual. Notifying Owner ($BUILDKITE_BUILD_CREATOR_EMAIL)."
    cat <<EOF > "$NOTIFY_FILE"
notify:
  - email: "$BUILDKITE_BUILD_CREATOR_EMAIL"
    if: build.state == "failed"
EOF

fi

buildkite-agent pipeline upload "$NOTIFY_FILE"
rm "$NOTIFY_FILE"

echo "Configure testing logic"
if [[ $BUILDKITE_PIPELINE_SLUG == "tpu-vllm-integration" ]]; then
    # Note: Integration pipeline always fetch latest vllm version
    VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    # Note: upload are inserted in reverse order, so promote LKG should upload before tests
    buildkite-agent pipeline upload .buildkite/integration_promote.yml
  
    # Upload JAX pipeline for v7
    export TESTS_GROUP_LABEL="[jax] TPU7x Tests Group"
    export TPU_VERSION="tpu7x"
    export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
    export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
    export IS_FOR_V7X="true"
    export COV_FAIL_UNDER="67"
    buildkite-agent pipeline upload .buildkite/pipeline_jax.yml
    unset TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI IS_FOR_V7X COV_FAIL_UNDER

    # Upload JAX pipeline for v6 (default)
    buildkite-agent pipeline upload .buildkite/pipeline_jax.yml

else
  # Note: PR and Nightly pipelines will load VLLM_COMMIT_HASH from vllm_lkg.version file, if not exists, get the latest commit hash from vllm repo
  if [ -f .buildkite/vllm_lkg.version ]; then
      VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version)"
  fi
  if [ -z "${VLLM_COMMIT_HASH:-}" ]; then
      VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
  fi
  buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
  echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    
  # Check if the current build is a pull request
  if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
    echo "This is a Pull Request build."

    # Wait for GitHub API to sync labels
    echo "Sleeping for 5 seconds to ensure GitHub API is updated..."
    sleep 5

    API_URL="https://api.github.com/repos/vllm-project/tpu-inference/pulls/$BUILDKITE_PULL_REQUEST"
    echo "Fetching PR details from: $API_URL"

    # Fetch the response body and save to a temporary file
    GITHUB_PR_RESPONSE_FILE="github_api_logs.json"
    curl -s "$API_URL" -o "$GITHUB_PR_RESPONSE_FILE"
    
    # Upload the full response body as a Buildkite artifact
    echo "Uploading GitHub API response as artifact..."
    buildkite-agent artifact upload "$GITHUB_PR_RESPONSE_FILE"

    # Extract labels using input redirection
    PR_LABELS=$(jq -r '.labels[].name' < "$GITHUB_PR_RESPONSE_FILE")
    echo "Extracted PR Labels: $PR_LABELS"

    # If it's a PR, check for the specific label
    if [[ $PR_LABELS == *"ready"* ]]; then
      echo "Found 'ready' label on PR. Uploading main pipeline..."
      upload_pipeline
    else
      # Explicitly fail the build because the required 'ready' label is missing.
      echo "Missing 'ready' label on PR. Failing build."
      exit 1
    fi
  else
    # If it's NOT a Pull Request (e.g., branch push, tag, manual build)
    echo "This is not a Pull Request build. Uploading main pipeline."
    upload_pipeline
  fi
fi


echo "--- Buildkite Bootstrap Finished"