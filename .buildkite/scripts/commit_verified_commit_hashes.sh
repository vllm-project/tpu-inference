#!/bin/sh
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

set -e

# --- Configuration ---
REPO_URL="https://github.com/vllm-project/tpu-inference.git"
TARGET_BRANCH="main"

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

if [ ! -f verified_commit_hashes.csv ]; then
    echo "timestamp,vllm_commit_hash,tpu_inference_commit_hash" > verified_commit_hashes.csv
fi
echo "$(date '+%Y-%m-%d %H:%M:%S'),${VLLM_COMMIT_HASH},${BUILDKITE_COMMIT}" >> verified_commit_hashes.csv

git add verified_commit_hashes.csv

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
