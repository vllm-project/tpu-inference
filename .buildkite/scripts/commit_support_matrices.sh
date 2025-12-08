#!/bin/sh
set -e

# --- Configuration ---
REPO_URL="https://github.com/vllm-project/tpu-inference.git"
TARGET_BRANCH="main"

# Conditional Configuration for Release vs. Nightly
if [ "${NIGHTLY}" = "1" ]; then
  # Set path and commit message for nightly builds.
  ARTIFACT_DOWNLOAD_PATH="support_matrices/nightly"
  COMMIT_MESSAGE="[skip ci] Update nightly support matrices"
else
  # Set path and commit message for release tag builds.
  COMMIT_TAG="${BUILDKITE_TAG:-unknown-tag}"
  ARTIFACT_DOWNLOAD_PATH="support_matrices"
  COMMIT_MESSAGE="[skip ci] Update support matrices for ${COMMIT_TAG}"
fi
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

echo "--- Downloading CSV artifacts"
mkdir -p "${ARTIFACT_DOWNLOAD_PATH}"
buildkite-agent artifact download "*.csv" "${ARTIFACT_DOWNLOAD_PATH}/" --flat

echo "--- Staging downloaded artifacts"
git add "${ARTIFACT_DOWNLOAD_PATH}"/*.csv

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
