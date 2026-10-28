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

set -euo pipefail

NEW_LKG_HASH="${1:-}"
if [ -z "${NEW_LKG_HASH}" ]; then
  echo "[ERROR] No hash provided. Usage: $0 <new_lkg_hash>" >&2
  exit 1
fi

TARGET_BRANCH="${TARGET_BRANCH:-main}"
MAX_ATTEMPTS=3
LKG_PROMOTE_ENABLED="${LKG_PROMOTE_ENABLED:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Create a throwaway temporary worktree directory
WORKTREE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-lkg-worktree.XXXXXX")"

cleanup() {
  echo "Cleaning up worktree at ${WORKTREE_DIR}..."
  git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE_DIR}" 2>/dev/null || true
  git -C "${REPO_ROOT}" worktree prune 2>/dev/null || true
  rm -rf "${WORKTREE_DIR}"
}
trap cleanup EXIT INT TERM

success=false
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "--- [Attempt ${attempt}/${MAX_ATTEMPTS}] Fetching and preparing worktree for ${TARGET_BRANCH}"

  # Clean up previous attempt if needed
  git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE_DIR}" 2>/dev/null || true
  rm -rf "${WORKTREE_DIR}"

  if ! git -C "${REPO_ROOT}" fetch origin "${TARGET_BRANCH}" || \
     ! git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE_DIR}" "origin/${TARGET_BRANCH}"; then
    echo "Attempt ${attempt} failed during fetch or worktree add. Retrying..."
    sleep 2
    continue
  fi

  if (
    cd "${WORKTREE_DIR}"

    echo "Updating pyproject.toml with new LKG hash: ${NEW_LKG_HASH}"
    sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${NEW_LKG_HASH}|g" pyproject.toml

    if git diff --quiet pyproject.toml; then
      echo "No change in LKG version. Skipping push."
      exit 0
    fi

    if [ "${LKG_PROMOTE_ENABLED}" != "true" ] && [ "${LKG_PROMOTE_ENABLED}" != "1" ]; then
      echo "LKG_PROMOTE_ENABLED is not 'true'. Performing dry-run."
      echo "Diff for pyproject.toml:"
      git diff pyproject.toml
      if command -v buildkite-agent &> /dev/null; then
        buildkite-agent annotate ":information_source: LKG dry-run to ${NEW_LKG_HASH} (promotion disabled)" --style "info"
      fi
      exit 0
    fi

    git add pyproject.toml
    # [skip ci] is crucial to prevent infinite Buildkite loop triggers
    git -C "${WORKTREE_DIR}" -c "user.name=vllm-ci-bot[bot]" -c "user.email=vllm-ci-bot[bot]@users.noreply.github.com" commit -s -m "[skip ci] Update vLLM LKG to ${NEW_LKG_HASH}"

    echo "Pushing updated version to remote ${TARGET_BRANCH}..."
    git push origin HEAD:"${TARGET_BRANCH}"
    echo "Successfully promoted LKG pin to ${NEW_LKG_HASH}"

    if command -v buildkite-agent &> /dev/null; then
      buildkite-agent annotate ":white_check_mark: LKG updated to ${NEW_LKG_HASH}" --style "success"
    fi
  ); then
    success=true
    break
  else
    echo "Attempt ${attempt} failed (possible push race condition). Retrying..."
    sleep 2
  fi
done

if [ "${success}" != "true" ]; then
  echo "ERROR: Failed to promote LKG version after ${MAX_ATTEMPTS} attempts." >&2
  exit 1
fi