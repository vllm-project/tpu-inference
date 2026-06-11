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

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <step_key>"
    exit 1
fi

STEP_KEY="$1"

echo "--- Checking ${STEP_KEY} Outcome (Hardware: ${CI_TPU_VERSION:-v6})"

# Try to get the custom string you saved
CUSTOM_STATUS=$(buildkite-agent meta-data get "${STEP_KEY}" --default "")

if [ -n "$CUSTOM_STATUS" ]; then
    OUTCOME="$CUSTOM_STATUS"
else
    OUTCOME=$(buildkite-agent step get "outcome" --step "${STEP_KEY}" || echo "skipped")
fi

echo "Step ${STEP_KEY} outcome: ${OUTCOME}"
message=""

case $OUTCOME in
  "passed")
    message="✅ Passing"
    ;;
  "skipped")
    message="❓ Untested"
    ;;
  "unverified")
    message="❓ Untested"
    ;;
  "not enough HBM")
    message="not enough HBM"
    ;;
  "transformers version too low")
    message="transformers version too low"
    ;;
  *)
    message="❌ Failing"
    ;;
esac

# Save the results using the hardware-specific prefix.
SAFE_TARGET=$(echo "$CI_TARGET" | tr '/:.[[:space:]]' '_' | sed 's/^_//;s/_$//;s/__\+/_/g')
SAFE_STAGE=$(echo "$CI_STAGE" | tr '/:.[[:space:]]' '_' | sed 's/^_//;s/_$//;s/__\+/_/g')
SAFE_META_KEY="${CI_TPU_VERSION}_${MODEL_IMPL_TYPE}_${SAFE_TARGET}_${SAFE_STAGE}"

buildkite-agent meta-data set "${CI_TPU_VERSION}_${SAFE_TARGET}_category" "${CI_CATEGORY}"
buildkite-agent meta-data set "${SAFE_META_KEY}" "${message}"

if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] && [ "${OUTCOME}" != "unverified" ] && [ "${OUTCOME}" != "not enough HBM" ] && [ "${OUTCOME}" != "transformers version too low" ]; then
    exit 1
fi
