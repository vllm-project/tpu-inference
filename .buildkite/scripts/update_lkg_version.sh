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

set -e

NEW_LKG_HASH=$1
if [[ -z "$NEW_LKG_HASH" ]]; then
    echo "Error: No hash provided."
    exit 1
fi

# Configuration
REPO_URL="https://github.com/vllm-project/tpu-inference.git"
TARGET_BRANCH="main"

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