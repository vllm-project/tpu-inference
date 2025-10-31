#!/bin/bash

# Export the commit message base on Release or Nightly
if [ "${NIGHTLY}" = "1" ]; then
    # Set path and commit message for nightly builds.
    export COMMIT_MESSAGE="Update nightly support matrices"
else
    if [ -z "${BUILDKITE_TAG:-}" ]; then
        echo "BUILDKITE_TAG not found"
        exit 1
    fi
    # Set path and commit message for release tag builds.
    export COMMIT_MESSAGE="Update support matrices for ${BUILDKITE_TAG}"
fi
