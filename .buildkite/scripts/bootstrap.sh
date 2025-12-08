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

upload_pipeline() {
    VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    buildkite-agent pipeline upload .buildkite/pipeline_jax.yml
    # buildkite-agent pipeline upload .buildkite/pipeline_torch.yml
    buildkite-agent pipeline upload .buildkite/main.yml
    buildkite-agent pipeline upload .buildkite/nightly_releases.yml
}

echo "--- Starting Buildkite Bootstrap ---"

# Check if the current build is a pull request
if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
  echo "This is a Pull Request build. Uploading main pipeline..."
  upload_pipeline
fi

echo "--- Buildkite Bootstrap Finished ---"
