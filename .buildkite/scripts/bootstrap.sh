#!/bin/bash

upload_pipeline() {
    #buildkite-agent pipeline upload .buildkite/pipeline_jax.yml
    # buildkite-agent pipeline upload .buildkite/pipeline_torch.yml
    buildkite-agent pipeline upload .buildkite/main.yml
}

echo "--- Starting Buildkite Bootstrap ---"

# Check if the current build is a pull request
if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
  echo "This is a Pull Request build."
  PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/tpu-inference/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')

  # If it's a PR, check for the specific label
  if [[ $PR_LABELS == *"ready"* ]]; then
    echo "Found 'ready' label on PR. Uploading main pipeline..."
    upload_pipeline
  else
    echo "No 'ready' label found on PR. Skipping main pipeline upload."
    exit 0 # Exit with 0 to indicate success (no error, just skipped)
  fi
else
  # If it's NOT a Pull Request (e.g., branch push, tag, manual build)
  echo "This is not a Pull Request build. Uploading main pipeline."
  upload_pipeline
fi

echo "--- Buildkite Bootstrap Finished ---"
