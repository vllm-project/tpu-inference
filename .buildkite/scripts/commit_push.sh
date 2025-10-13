#!/bin/bash
set -e

# Please do `chmod +x <path/to/pre_commit_script.sh>`
if test ! -x "$1"; then
    echo "ERROR: The first argument must be a path to an executable pre-commit script."
    echo "Usage: ./commit_push.sh <path/to/pre_commit_script.sh>"
    exit 1
fi

# --- Configuration ---

# $BUILDKITE_PULL_REQUEST_REPO would be "" if not a pull request. (ex. manual trigger)
REPO_URL="${BUILDKITE_PULL_REQUEST_REPO:-https://github.com/vllm-project/tpu-inference.git}"

TARGET_BRANCH="$BUILDKITE_BRANCH"

# $BUILDKITE_PULL_REQUEST would be "false" if not a pull request. (ex. manual trigger)
if [ "$BUILDKITE_PULL_REQUEST" = "false" ] || [ "$BUILDKITE_PULL_REQUEST" = "" ]; then
  PR_NUM="unknown"
else
  PR_NUM="$BUILDKITE_PULL_REQUEST"
fi

COMMIT_MESSAGE="[skip ci] Update verified commit hashes for PR Num $PR_NUM"

# Construct the repository URL with the access token for authentication.
AUTHENTICATED_REPO_URL="https://x-access-token:${GITHUB_PAT}@${REPO_URL#https://}"

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

echo "Executing custom pre-commit script: $1"
if ! "$1"; then
    echo "ERROR: Pre-commit script failed (files not added to staging). Aborting."
    exit 1
fi

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
