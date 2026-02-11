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
TARGET_BRANCH="${BUILDKITE_BRANCH:-main}"

# Conditional Configuration for Release vs. Nightly
if [ "${NIGHTLY}" = "1" ]; then
  # Set path and commit message for nightly builds.
  BASE_PATH="support_matrices/nightly"
  
  # Check for specific model implementation type to enable directory isolation.
  # If MODEL_IMPL_TYPE is 'vllm' or 'flax_nnx', use a implementation-specific subfolder.
  if [ "${MODEL_IMPL_TYPE:-auto}" = "vllm" ] || [ "${MODEL_IMPL_TYPE:-auto}" = "flax_nnx" ]; then
    ARTIFACT_DOWNLOAD_PATH="${BASE_PATH}/${MODEL_IMPL_TYPE}"
    COMMIT_MESSAGE="[skip ci] Update nightly support matrices for ${MODEL_IMPL_TYPE} (v6/v7)"
  else
    # Default case: support_matrices/nightly
    ARTIFACT_DOWNLOAD_PATH="${BASE_PATH}"
    COMMIT_MESSAGE="[skip ci] Update nightly support matrices (v6/v7)"
  fi
else
  # Set path and commit message for release tag builds.
  COMMIT_TAG="${BUILDKITE_TAG:-unknown-tag}"
  ARTIFACT_DOWNLOAD_PATH="support_matrices"
  COMMIT_MESSAGE="[skip ci] Update support matrices for ${COMMIT_TAG} (v6/v7)"
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
# Download without --flat to preserve v6/ or v7/ folder structure
buildkite-agent artifact download "v*/*.csv" "."

# Iterate through v6 and v7 folders if they exist
for ver in v6 v7; do
  if [ -d "$ver" ]; then
    TARGET_DIR="${ARTIFACT_DOWNLOAD_PATH}/${ver}"
    echo "Syncing ${ver} artifacts to ${TARGET_DIR}..."
    
    mkdir -p "${TARGET_DIR}"
    
    # Move the files to the final destination in the repo
    mv "${ver}"/*.csv "${TARGET_DIR}/"
    
    # Clean up the temporary download directory
    rmdir "${ver}"
  else
    echo "No artifacts found for version: ${ver}. Skipping."
  fi
done

echo "--- Staging changes"
git add support_matrices/

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
