#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

waitForServerReady() {
    # Inside `[[ x -ge y ]]` the operands go through bash arithmetic evaluation,
    # which would execute command-substitution syntax in a non-integer value.
    if [[ ! "${TIMEOUT_SECONDS:-}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: TIMEOUT_SECONDS must be a non-negative integer, got: '${TIMEOUT_SECONDS:-}'" >&2
        exit 1
    fi

    # shellcheck disable=SC2155
    local start_time=$(date +%s)
    echo "Waiting for server ready message: '$READY_MESSAGE'"

    local fatal_error_patterns=(
        "RuntimeError:"
        "ValueError:"
        "FileNotFoundError:"
        "TypeError:"
        "ImportError:"
        "NotImplementedError:"
        "AssertionError:"
        "TimeoutError:"
        "OSError:"
        "AttributeError:"
        "NVMLError:"
    )

    local error_regex
    error_regex=$(IFS=\|; echo "${fatal_error_patterns[*]}")

    while true; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        sleep 5

        if [[ "$elapsed_time" -ge "$TIMEOUT_SECONDS" ]]; then
            echo "TIMEOUT: Waited $elapsed_time seconds (limit was $TIMEOUT_SECONDS). The string '$READY_MESSAGE' was NOT found."
            # Cleanup is the calling script's trap.
            exit 1
        fi

        # JAX logs a losing compilation-cache write race as "UserWarning: ...
        # OSError: [Errno 116] Stale file handle" and then compiles anyway.
        # Excused by its whole signature, not by the word "Warning", so the
        # patterns still catch a hung startup.
        #
        # Assigned rather than piped into the test: with no match the pipeline
        # exits nonzero, which under the callers' `set -e` would end the run.
        local fatal_lines
        fatal_lines=$(grep -E "$error_regex" "$LOG_FILE" \
            | grep -v -E 'UserWarning.*Stale file handle' || true)
        if [[ -n "$fatal_lines" ]]; then
            echo "FATAL ERROR DETECTED: The server log contains a fatal error pattern."
            exit 1
        fi

        if grep -Fq "$READY_MESSAGE" "$LOG_FILE" ; then
            echo "Server is ready."
            return 0
        fi
    done
}

# Usage: cleanUp <MODEL_NAME>
cleanUp() {
    echo "Stopping the vLLM server and cleaning up log files..."
    pkill -f "vllm serve $1"
    pgrep -f -i vllm | xargs -r kill -9

    rm -f "$LOG_FILE"
    rm -f "$BENCHMARK_LOG_FILE"
    echo "Cleanup complete."
}
