#!/usr/bin/env bash
# Copyright 2026 Google LLC
# Wrapper script for Kubernetes pipeline test steps to sync new JAX cache entries back to GCS on success.

set -o pipefail

# Execute the test command passed to this script
"$@"
TEST_EXIT_CODE=$?

# If the test passed, sync any new cache entries back to central GCS
if [ $TEST_EXIT_CODE -eq 0 ]; then
  CACHE_DIR="/cache/tpu_jax_cache"
  GCS_DEST="gs://ullm-ci-cache/jax_cache/jax0.11.0_tputpu6e"
  
  if [ -d "$CACHE_DIR" ] && [ "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[INFO] Syncing new JAX compilation cache entries to GCS ($GCS_DEST)..."
    gcloud storage rsync \
      --recursive \
      --no-clobber \
      --exclude=".*_.gstmp$" \
      --no-user-output-enabled \
      "$CACHE_DIR" "$GCS_DEST" || \
      echo "[WARN] Failed to sync JAX cache to GCS."
  fi
fi

exit $TEST_EXIT_CODE
