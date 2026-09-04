#!/bin/bash
# Copyright 2026 Google LLC
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

# Pushes the validated tokamax pin to main. Called from the promote step of the
# tokamax integration pipeline, which only runs after every test step passed.
# Mirrors update_lkg_version.sh.

set -e

NEW_VERSION=$1
if [[ -z "$NEW_VERSION" ]]; then
    echo "Error: No tokamax version provided."
    exit 1
fi

# Configuration. TARGET_BRANCH is overridable so the flow can be rehearsed
# against a scratch branch without writing to main.
TARGET_BRANCH="${TARGET_BRANCH:-main}"
REQUIREMENTS_FILE="requirements.txt"

# Configure credentials
git config user.name "vllm-ci-bot[bot]"
git config user.email "vllm-ci-bot[bot]@users.noreply.github.com"

# Fetch and checkout
git fetch origin "${TARGET_BRANCH}"
git checkout -f "${TARGET_BRANCH}"
git reset --hard origin/"${TARGET_BRANCH}"

# Update the tokamax pin
echo "Updating $REQUIREMENTS_FILE to: tokamax==$NEW_VERSION"
sed -i "s/^tokamax==.*$/tokamax==${NEW_VERSION}/" "$REQUIREMENTS_FILE"

# Check in file
git add "$REQUIREMENTS_FILE"

# Check if we have changed anything
if git diff --cached --quiet; then
    echo "No changes in the tokamax pin. Skipping push."
else
    # "[skip ci]" matches update_lkg_version.sh: the v6e+v7x suite that just
    # passed is the validation for this bump, so a post-merge re-run would
    # largely duplicate it. Note the reset above lands the bump on whatever
    # main is now, which may have moved past the commit this build tested.
    git commit -s -m "[skip ci] Update tokamax pin to $NEW_VERSION"
    echo "Pushing the tokamax bump to $TARGET_BRANCH..."
    git push origin "$TARGET_BRANCH"
    echo "Successfully bumped tokamax to $NEW_VERSION"
fi
