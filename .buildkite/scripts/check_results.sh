#!/bin/sh
set -e

ANY_FAILED=false

STEP_KEYS="tpu_test_0 tpu_test_1 tpu_test_2 tpu_test_3 tpu_test_4 tpu_test_5 tpu_test_6 tpu_test_7 tpu_test_8 tpu_test_9 tpu_test_10 tpu_test_11 tpu_test_12 tpu_test_13"

echo "--- Checking Test Outcomes"

for KEY in $STEP_KEYS; do
    OUTCOME=$(buildkite-agent step get "outcome" --step "${KEY}" || echo "skipped")
    echo "Step ${KEY} outcome: ${OUTCOME}"

    if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] ; then
    ANY_FAILED=true
    fi
done

if [ "${ANY_FAILED}" = "true" ] ; then
    cat <<- YAML | buildkite-agent pipeline upload
    steps:
    - label: "TPU V1 Test failed"
        agents:
        queue: tpu_v6_queue
        command: echo "TPU V1 Test failed"
YAML
    exit 1
else
    echo "All relevant TPU tests passed (or were skipped)."
fi
