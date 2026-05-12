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

#
# Sync vllm-env, tpu-inference source, and vllm source from the head node to
# all worker nodes in parallel.
#
# Run this after making local code changes that need to be reflected on all
# hosts before launching a multi-host job.
#
# Usage (run on the head/coordinator node):
#   bash rsync_workers.sh
#
# Configure the worker list and paths via environment variables or by editing
# the Configuration section below.

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
# All values below must be set by the caller — no pod-specific defaults
# are committed to the repo.
: "${WORKERS:?set WORKERS to the space-separated hostname / IP list of the non-coordinator hosts in this pod}"
: "${SSH_USER:?set SSH_USER to the account name that has passwordless SSH to every host}"
: "${VENV_DIR:?set VENV_DIR to the Python venv path (must exist at the same path on every host)}"
: "${TPU_INFERENCE_DIR:?set TPU_INFERENCE_DIR to the local tpu-inference checkout path, same on every host}"
# VLLM_DIR is optional — set it if you have a vllm checkout to sync alongside tpu-inference;
# unset (or set empty) to skip syncing vllm.
VLLM_DIR="${VLLM_DIR-}"
# ──────────────────────────────────────────────────────────────────────────────

RSYNC_OPTS="-az --timeout=300 --delete \
  --exclude=__pycache__ \
  --exclude='*.pyc' \
  --exclude='.git'"

echo "=== Syncing to workers: $WORKERS ==="

for host in $WORKERS; do
  (
    echo "[$(date +%T)] $host: starting..."

    echo "[$(date +%T)] $host: vllm-env..."
    # shellcheck disable=SC2086
    timeout 600 rsync $RSYNC_OPTS "$VENV_DIR/" \
      "$SSH_USER@$host:$VENV_DIR/" 2>&1 | tail -1
    echo "[$(date +%T)] $host: vllm-env done"

    echo "[$(date +%T)] $host: tpu-inference..."
    # shellcheck disable=SC2086
    timeout 300 rsync $RSYNC_OPTS "$TPU_INFERENCE_DIR/" \
      "$SSH_USER@$host:$TPU_INFERENCE_DIR/" 2>&1 | tail -1
    echo "[$(date +%T)] $host: tpu-inference done"

    echo "[$(date +%T)] $host: vllm src..."
    # shellcheck disable=SC2086
    timeout 300 rsync $RSYNC_OPTS "$VLLM_DIR/" \
      "$SSH_USER@$host:$VLLM_DIR/" 2>&1 | tail -1
    echo "[$(date +%T)] $host: vllm src done"

    echo "[$(date +%T)] $host: ALL DONE"
  ) &
done

wait
echo "=== ALL $(echo "$WORKERS" | wc -w) WORKERS SYNCED ==="
