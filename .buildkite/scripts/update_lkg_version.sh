#!/bin/bash
# .buildkite/scripts/update_lkg_version.sh

set -e
NEW_LKG_HASH=$1
if [[ -z "$NEW_LKG_HASH" ]]; then
    echo "Error: No hash provided."
    exit 1
fi

# Configuration
REPO_URL="https://github.com/vllm-project/tpu-inference.git"
TARGET_BRANCH="feat/buildkite-vllm-pipeline"

NEW_LKG_HASH=$1
VERSION_FILE=".buildkite/vllm_lkg.version"

# Construct the repository URL with the access token for authentication
AUTHENTICATED_REPO_URL="https://x-access-token:${GITHUB_PAT}@${REPO_URL#https://}"
git remote set-url origin "${AUTHENTICATED_REPO_URL}"

# Ensure the GITHUB_PAT is available before proceeding
if [ -z "${GITHUB_PAT:-}" ]; then
  echo "--- ERROR: GITHUB_PAT secret not found. Cannot proceed."
  exit 1
fi

# Configure credentials
git config user.name "Buildkite Bot"
git config user.email "buildkite-bot@users.noreply.github.com"

# Fetch and checkout
git fetch origin "${TARGET_BRANCH}"
git checkout -f "${TARGET_BRANCH}"
git reset --hard origin/"${TARGET_BRANCH}"

# Update vllm_lkg.version
echo "Current directory: $(pwd)" # debug
echo "Updating $VERSION_FILE to: $NEW_LKG_HASH"
echo "$NEW_LKG_HASH" > "$VERSION_FILE"

# Check in file
git add "$VERSION_FILE"

# Check if we have change anything
if git diff --cached --quiet; then
    echo "No changes in LKG version. Skipping push."
else
    git commit -s -m "[skip ci] Update vLLM LKG to $NEW_LKG_HASH"
    echo "Pushing LKG update to $TARGET_BRANCH..."
    git push origin $TARGET_BRANCH 
    echo "Successfully updated LKG to $NEW_LKG_HASH"
fi