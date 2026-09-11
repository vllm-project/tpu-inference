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

# TOLERATED_SOFT_FAIL_STEPS is an optional, space-separated list of step keys
# whose soft_failed outcome must not fail this gate. Set it in the pipeline
# environment while a known-broken step is being fixed, and clear it once the
# step is green again. Every other non-passed, non-skipped outcome still fails
# the build, so the gate keeps blocking real regressions.
TOLERATED_SOFT_FAIL_STEPS="${TOLERATED_SOFT_FAIL_STEPS:-}"

is_tolerated() {
    case " ${TOLERATED_SOFT_FAIL_STEPS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

echo "--- Checking Test Outcomes"

for KEY in "$@"; do
    OUTCOME=$(buildkite-agent step get "outcome" --step "${KEY}" || echo "skipped")
    if [ -z "$OUTCOME" ]; then
        OUTCOME="skipped"
    fi
    echo "Step ${KEY} outcome: ${OUTCOME}"

    if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] ; then
        if [ "${OUTCOME}" = "soft_failed" ] && is_tolerated "${KEY}"; then
            echo "Step ${KEY} is in TOLERATED_SOFT_FAIL_STEPS; not failing the build."
        else
            ANY_FAILED=true
        fi
    fi
done

if [ "${ANY_FAILED}" = "true" ] ; then
    # Strip everything outside a conservative charset before interpolating the
    # caller-supplied label into YAML. Prevents YAML / shell injection if a
    # pipeline file passes an attacker-controlled string. Use a fixed command
    # body instead of echoing the label.
    SAFE_LABEL=$(printf '%s' "${FAILURE_LABEL}" | tr -cd '[:alnum:] _.:/-')
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
