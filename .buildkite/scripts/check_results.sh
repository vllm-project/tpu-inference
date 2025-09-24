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
    echo "Step ${KEY} outcome: ${OUTCOME}"

    if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] ; then
        ANY_FAILED=true
    fi
done

# Check Test Result and upload to buildkite meta-data
if [ -n "${EXECUTE_ENTITY:-}" ] && \
   [ -n "${EXECUTE_STAGE:-}" ] && \
   [[ "${BUILDKITE_STEP_KEY:-}" == "notifications_"* ]]; then

    # If all conditions are true, execute the logic here.
    echo "EXECUTE_ENTITY: $EXECUTE_ENTITY"
    echo "EXECUTE_STAGE: $EXECUTE_STAGE"
    echo "BUILDKITE_STEP_KEY: $BUILDKITE_STEP_KEY"
    
    if [ "${ANY_FAILED}" = "true" ]; then
      echo "The step failed. Uploading $EXECUTE_ENTITY:$EXECUTE_STAGE result..."
      buildkite-agent meta-data set "$EXECUTE_ENTITY:$EXECUTE_STAGE" "failed"
    else
      echo "The step passed. Uploading $EXECUTE_ENTITY:$EXECUTE_STAGE result..."
      buildkite-agent meta-data set "$EXECUTE_ENTITY:$EXECUTE_STAGE" "passed"
    fi
fi

if [ "${ANY_FAILED}" = "true" ]; then
    cat <<- YAML | buildkite-agent pipeline upload
    steps:
    - label: "${FAILURE_LABEL}"
        agents:
        queue: tpu_v6_queue
        command: echo "${FAILURE_LABEL}"
YAML
    exit 1
else
    echo "All relevant TPU tests passed (or were skipped)."
fi
