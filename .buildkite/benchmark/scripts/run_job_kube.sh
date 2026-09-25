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

# run_job.sh for the Kueue fleet. It runs on the kube agent, which holds no
# chips: the benchmark itself runs in a workload pod that
# .buildkite/kubernetes/run.sh submits. What run_job.sh reads from the TPU VM -
# the agent's queue, a host artifact folder shared with the container - comes
# from the step instead, and the pod uploads the logs as artifacts itself,
# since its filesystem is gone once it exits.
set -euo pipefail

CASE_FILE="${1:-}"
TARGET_CASE_NAME="${2:-}"
if [ -z "$CASE_FILE" ] || [ -z "$TARGET_CASE_NAME" ]; then
    echo "Usage: $0 <case.json> <TARGET_CASE_NAME>"
    exit 1
fi

: "${BUILDKITE_STEP_ID:?[ERROR] The BUILDKITE_STEP_ID variable is missing or empty!}"
# Use Buildkite step ID to ensure retries map to the same RecordId
RECORD_ID="${BUILDKITE_STEP_ID}"
CODE_HASH=$(buildkite-agent meta-data get "CODE_HASH")
JOB_REFERENCE=$(buildkite-agent meta-data get "JOB_REFERENCE")
RUN_TYPE="${RUN_TYPE:-DAILY}"
buildkite-agent step update "label" " RecordId: ${RECORD_ID}" --append

# The DB records the device by the bare-metal queue the case names, so kube
# results sit next to bare-metal ones. The generator puts it in ci_queue.
: "${ci_queue:?[ERROR] ci_queue is missing; the step was not made by generate_bk_pipeline.py}"
if [[ "$ci_queue" =~ ^tpu_v(7x|6e)_([0-9]+)_queue$ ]]; then
    if [ "${BASH_REMATCH[1]}" == "7x" ]; then
        DEVICE="tpu7x-${BASH_REMATCH[2]}"
    else
        DEVICE="v6e-${BASH_REMATCH[2]}"
    fi
elif [ "$ci_queue" == "tpu_v6e_queue" ]; then
    DEVICE="v6e-1"
else
    DEVICE="$ci_queue"
fi
echo "[INFO] Dynamic mapping complete: $ci_queue -> $DEVICE"

# Passed on the command line rather than through run.sh's FORWARD list: these
# are per-run values only this script computes. The pod starts in the image's
# WORKDIR, the checkout run_bm.sh's relative paths expect.
#
# UPLOAD_DB is off unless the build turns it on. The rows it writes keep the
# bare-metal Device, so kube and bare-metal results sit side by side, and
# RUN_TYPE_SUFFIX marks the run type as kube's (see report_result.sh).
#
# run_bm.sh writes its logs under ARTIFACT_FOLDER/temp_logs, which run.sh then
# uploads through ARTIFACTS_DIR: the server and benchmark logs run_job.sh
# uploads on bare metal, and the rest of what report_result.sh copies to GCS.
export ARTIFACTS_DIR=artifacts/temp_logs
exec .buildkite/kubernetes/run.sh env \
    ARTIFACT_FOLDER=/workspace/tpu_inference/artifacts \
    DEVICE="$DEVICE" \
    RECORD_ID="$RECORD_ID" \
    RUN_TYPE="$RUN_TYPE" \
    RUN_TYPE_SUFFIX=_KUBE \
    CODE_HASH="$CODE_HASH" \
    JOB_REFERENCE="$JOB_REFERENCE" \
    EXTRA_ENVS="${EXTRA_ENVS:-}" \
    UPLOAD_DB="${UPLOAD_DB:-false}" \
    BM_INFRA=true \
    BUILDKITE_AGENT_META_DATA_QUEUE="$ci_queue" \
    BUILDKITE_RETRY_COUNT="${BUILDKITE_RETRY_COUNT:-0}" \
    MLCOMPASS_EXECUTION_MODE="${MLCOMPASS_EXECUTION_MODE:-}" \
    MLCOMPASS_EXPORT_ENABLED="${MLCOMPASS_EXPORT_ENABLED:-}" \
    MLCOMPASS_TEST_NAME="${MLCOMPASS_TEST_NAME:-}" \
    MLCOMPASS_TRACKING_ID="${MLCOMPASS_TRACKING_ID:-}" \
    MLCOMPASS_SPONGE_ID="${MLCOMPASS_SPONGE_ID:-}" \
    bash .buildkite/benchmark/scripts/run_bm.sh "$CASE_FILE" "$TARGET_CASE_NAME"
