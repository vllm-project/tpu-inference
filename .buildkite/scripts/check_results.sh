#!/bin/sh
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
