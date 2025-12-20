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

echo "--- Checking ${STEP_KEY} Outcome"

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
    message="✅"
    ;;
  "skipped")
    message="N/A"
    ;;
  "unverified")
    message="unverified"
    ;;
  *)
    message="❌"
    ;;
esac

buildkite-agent meta-data set "${CI_TARGET}_category" "${CI_CATEGORY}"
buildkite-agent meta-data set "${CI_TARGET}:${CI_STAGE}" "${message}"

if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] && [ "${OUTCOME}" != "unverified" ]; then
    exit 1
fi
