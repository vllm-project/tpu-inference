#!/bin/sh
set -e

# We assume the first argument is the build type (e.g. "pr" or "daily")
run_type="${1:-pr}"

echo "--- Build type is: ${run_type}"

ANY_FAILED=false

if [ "${run_type}" = "pr" ] ; then
    STEP_KEYS="tpu_test_0 tpu_test_1 tpu_test_2 tpu_test_3 tpu_test_4 tpu_test_5 tpu_test_6 tpu_test_7 tpu_test_8 tpu_test_9 tpu_test_10 tpu_test_11 tpu_test_12 tpu_test_13"
elif [ "${run_type}" = "daily" ] ; then
    STEP_KEYS="tpu_daily_test_0"
else
    echo "Unknown build type: ${run_type}"
    exit 1
fi

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
