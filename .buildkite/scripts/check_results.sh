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

ANY_FAILED=false
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <failure_label> <step_key_1> <step_key_2> ..."
    exit 1
fi

FAILURE_LABEL="$1"
shift

echo "--- Checking Test Outcomes"

for KEY in "$@"; do
    OUTCOME=$(buildkite-agent step get "outcome" --step "${KEY}" || echo "skipped")
    if [ -z "$OUTCOME" ]; then
        OUTCOME="skipped"
    fi
    echo "Step ${KEY} outcome: ${OUTCOME}"

    if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] ; then
        ANY_FAILED=true
    fi
done

if [ "${ANY_FAILED}" = "true" ] ; then
    cat <<- YAML | buildkite-agent pipeline upload
steps:
   - label: "${FAILURE_LABEL}"
     agents:
       queue: cpu
     command: echo "${FAILURE_LABEL}"
YAML
    exit 1
else
    echo "All relevant TPU tests passed (or were skipped)."
fi
