#!/bin/bash

# --- Skip build if only docs/icons changed ---
echo "--- :git: Checking changed files"

# Get a list of all files changed in this commit
FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r "$BUILDKITE_COMMIT")

echo "Files changed:"
echo "$FILES_CHANGED"

# Filter out files we want to skip builds for.
NON_SKIPPABLE_FILES=$(echo "$FILES_CHANGED" | grep -vE "(\.md$|\.ico$|\.png$|^README$|^docs\/)")

if [ -z "$NON_SKIPPABLE_FILES" ]; then
  echo "Only documentation or icon files changed. Skipping build."
  # No pipeline will be uploaded, and the build will complete.
  exit 0
else
  echo "Code files changed. Proceeding with pipeline upload."
fi

upload_fast_check_pipeline() {
    buildkite-agent pipeline upload .buildkite/pipeline_fast_check.yml
}

upload_pipeline() {
    buildkite-agent pipeline upload .buildkite/pipeline_jax.yml
    # buildkite-agent pipeline upload .buildkite/pipeline_torch.yml
    buildkite-agent pipeline upload .buildkite/main.yml
    buildkite-agent pipeline upload .buildkite/nightly_releases.yml
    upload_fast_check_pipeline
}

fetch_latest_upstream_vllm_commit() {
    # To help with debugging (when needed), perform setup to:
    #    1. Use the same upstream vllm commit for all jobs in this CI run for consistency
    #    2. Record which upstream commit this CI run is using
    VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
}

echo "--- Starting Buildkite Bootstrap ---"
fetch_latest_upstream_vllm_commit
# Check if the current build is a pull request
if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
  echo "This is a Pull Request build."
  PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/tpu-inference/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')

  # If it's a PR, check for the specific label
  if [[ $PR_LABELS == *"ready"* ]]; then
    echo "Found 'ready' label on PR. Uploading main pipeline..."
    upload_pipeline
  else
    echo "No 'ready' label found on PR. Uploading fast check pipeline"
    upload_fast_check_pipeline
  fi
else
  # If it's NOT a Pull Request (e.g., branch push, tag, manual build)
  echo "This is not a Pull Request build. Uploading main pipeline."
  upload_pipeline
fi

echo "--- Buildkite Bootstrap Finished ---"
