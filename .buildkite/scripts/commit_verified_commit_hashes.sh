#!/bin/sh
set -e

# --- Configuration ---
# $BUILDKITE_PULL_REQUEST_REPO would be "" if not a pull request. (ex. manual trigger)
REPO_URL="${BUILDKITE_PULL_REQUEST_REPO:-https://github.com/vllm-project/tpu-inference.git}"
TARGET_BRANCH="$BUILDKITE_BRANCH"

COMMIT_MESSAGE="Update verified commit hashes"

# Construct the repository URL with the access token for authentication.
AUTHENTICATED_REPO_URL="https://x-access-token:${GITHUB_PAT}@${REPO_URL#https://}"

# Ensure the GITHUB_PAT is available before proceeding.
if [ -z "${GITHUB_PAT:-}" ]; then
  echo "--- ERROR: GITHUB_PAT secret not found. Cannot proceed."
  exit 1
fi

echo "--- Configuring Git user details"
git config user.name "Buildkite Bot"
git config user.email "buildkite-bot@users.noreply.github.com"

echo "--- Fetching and checking out the target branch"
git fetch origin "${TARGET_BRANCH}"
git checkout "${TARGET_BRANCH}"
git reset --hard origin/"${TARGET_BRANCH}"

VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")

if [ -z "${VLLM_COMMIT_HASH}" ]; then
    echo "VLLM_COMMIT_HASH not found in buildkite meta-data"
    exit 1
fi

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "BUILDKITE_COMMIT not found"
    exit 1
fi

if [ ! -f commit_hashes.csv ]; then
    echo "timestamp,vllm_commit_hash,tpu_inference_commit_hash" > commit_hashes.csv
fi
echo "$(date '+%Y-%m-%d %H:%M:%S'),${VLLM_COMMIT_HASH},${BUILDKITE_COMMIT}" >> commit_hashes.csv

git add commit_hashes.csv

# --- Check for changes before committing ---
if git diff --quiet --cached; then
  echo "No changes to commit. Exiting successfully."
  exit 0
else
  echo "--- Committing changes"
  git commit -s -m "${COMMIT_MESSAGE}"

  echo "--- Pushing changes to '${TARGET_BRANCH}'"
  git push "${AUTHENTICATED_REPO_URL}" "HEAD:${TARGET_BRANCH}"
fi
