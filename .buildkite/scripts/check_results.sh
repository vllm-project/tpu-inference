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

# Aggregates the outcomes of the TPU test steps and fails the build if any of
# them did not pass. When steps are marked ``soft_fail: true`` their individual
# GitHub status is green, so this aggregator is what surfaces a real failure on
# the umbrella ``buildkite/tpu-inference-ci/pr`` context. It therefore names the
# steps that actually failed (build log + a Buildkite annotation) so the red
# umbrella is actionable instead of a generic "Tests Failed".

set -e

ANY_FAILED=false
FAILED_KEYS=""
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
        # Newline-separated so each failing "key (outcome)" stays on one line.
        FAILED_KEYS="${FAILED_KEYS}
${KEY} (${OUTCOME})"
    fi
done

if [ "${ANY_FAILED}" = "true" ] ; then
    # Drop the leading blank line for readability.
    FAILED_KEYS=$(printf '%s\n' "${FAILED_KEYS}" | sed '/^$/d')
    echo "--- ^^^ Failed steps:"
    printf '%s\n' "${FAILED_KEYS}" | sed 's/^/  - /'

    # Strip everything outside a conservative charset before interpolating the
    # caller-supplied label into YAML/markdown. Prevents YAML / shell injection if
    # a pipeline file passes an attacker-controlled string. Use a fixed command
    # body instead of echoing the label.
    SAFE_LABEL=$(printf '%s' "${FAILURE_LABEL}" | tr -cd '[:alnum:] _.:/-')

    # Surface the failing step keys in a Buildkite annotation so the umbrella red
    # is actionable. Best-effort: never let the annotation itself change the exit
    # status of this aggregator (it must still exit 1 below).
    if command -v buildkite-agent >/dev/null 2>&1; then
        printf '**%s**\n\nFailed steps:\n%s\n' \
            "${SAFE_LABEL}" \
            "$(printf '%s\n' "${FAILED_KEYS}" | sed '/^$/d; s/^/- /')" \
            | buildkite-agent annotate --style "error" --context "check-results" 2>/dev/null || true
    fi

    cat <<- YAML | buildkite-agent pipeline upload
steps:
   - label: "${SAFE_LABEL}"
     agents:
       queue: cpu
     command: 'echo "test failure recorded"'
YAML
    exit 1
else
    echo "All relevant TPU tests passed (or were skipped)."
fi
