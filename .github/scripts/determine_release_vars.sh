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

set -eu pipefail

# --- SCHEDULE TRIGGER ---
if [[ "$GH_EVENT_NAME"  == "schedule" ]]; then
    echo "Trigger: Schedule - Generating nightly build"

    # --- Get Base Version from Tag ---
    echo "Fetching latest tags..."
    git fetch --tags --force
    echo "Finding the latest stable version tag (vX.Y.Z)..."
    LATEST_STABLE_TAG=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)
    if [[ -z "$LATEST_STABLE_TAG" ]]; then
        echo "Warning: No stable tag found."
        exit 1
    else
        BASE_VERSION=${LATEST_STABLE_TAG#v}
    fi
    echo "Using BASE_VERSION=${BASE_VERSION}"

    # --- Generate Nightly Version ---
    DATETIME_STR=$(date -u +%Y%m%d)
    VERSION="${BASE_VERSION}.dev${DATETIME_STR}"

# --- PUSH TAG TRIGGER ---
elif [[ "$GH_EVENT_NAME" == "push" && "$GH_REF" == refs/tags/* ]]; then
    echo "Trigger: Push Tag - Generating stable build"
    TAG_NAME="$GH_REF_NAME"
    VERSION=${TAG_NAME#v}

else
    echo "Error: Unknown or unsupported trigger."
    exit 1
fi

# --- output ---
echo "Final determined values: VERSION=${VERSION}"
echo "VERSION=${VERSION}" >> "$GITHUB_OUTPUT"
