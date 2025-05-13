#!/bin/bash

REQUIRED_PR_LABEL="ready"

echo "--- Starting Buildkite Bootstrap ---"

# Check if the current build is a pull request
if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
  echo "This is a Pull Request build."
  # If it's a PR, check for the specific label
  if buildkite-agent meta-data get "buildkite:pull_request_labels" | grep -q "$REQUIRED_PR_LABEL"; then
    echo "Found '$REQUIRED_PR_LABEL' label on PR. Uploading main pipeline..."
    buildkite-agent pipeline upload .buildkite/pipeline.yml
  else
    echo "No '$REQUIRED_PR_LABEL' label found on PR. Skipping main pipeline upload."
    exit 0 # Exit with 0 to indicate success (no error, just skipped)
  fi
else
  # If it's NOT a Pull Request (e.g., branch push, tag, manual build)
  echo "This is not a Pull Request build. Uploading main pipeline."
  buildkite-agent pipeline upload .buildkite/pipeline.yml
fi

echo "--- Buildkite Bootstrap Finished ---"
