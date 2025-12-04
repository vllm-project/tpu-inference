#!/bin/bash
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
