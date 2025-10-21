#!/bin/bash
set -e

# Conditional Configuration for Release vs. Nightly
if [ "${NIGHTLY}" = "1" ]; then
  ARTIFACT_DOWNLOAD_PATH="support_matrices/nightly"
else
  ARTIFACT_DOWNLOAD_PATH="support_matrices"
fi

echo "--- Downloading CSV artifacts"
mkdir -p "${ARTIFACT_DOWNLOAD_PATH}"
buildkite-agent artifact download "*.csv" "${ARTIFACT_DOWNLOAD_PATH}/" --flat

echo "--- Staging downloaded artifacts"
git add "${ARTIFACT_DOWNLOAD_PATH}"/*.csv
