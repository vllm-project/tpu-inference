#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# BENCHMARK UTILITY FUNCTIONS
# This file is sourced by various performance scripts (e.g., mlperf.sh,
# llama_guard_perf_recipe.sh) to share common functions.
# -----------------------------------------------------------------------------

# waitForServerReady: Blocks execution until the server prints the READY_MESSAGE or times out.
# This logic is shared across all benchmark scripts.
waitForServerReady() {
    # Reject non-integer TIMEOUT_SECONDS up front. Inside `[[ x -ge y ]]` the
    # operands go through bash arithmetic evaluation, which will execute
    # command-substitution syntax in the value if a caller ever sets it to
    # something exotic. Easier to fail loudly than to rely on the caller.
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
            # Call cleanup and exit (cleanup must be handled by the calling script's trap)
            exit 1
        fi

        # One line is excused, by its whole signature rather than by the word
        # "Warning": JAX logs a failed compilation-cache write as
        # "UserWarning: ... OSError: [Errno 116] Stale file handle" and goes on
        # to compile. The cache is a shared gcsfuse mount, and when two pods
        # write the same content-addressed key at once, Cloud Storage keeps the
        # first and fails the second's precondition; the loser's write was a
        # duplicate, so losing it costs nothing - but matching the OSError it
        # quotes killed a run whose server was healthy.
        #
        # Excusing every line that says "Warning:" would be the wider fix and
        # the wrong one: these patterns are what stands between a hung startup
        # and a three-hour timeout, and a real fatal line that happens to carry
        # the word would stop being seen here and on bare metal both.
        #
        # Assigned rather than tested through a pipe: with no match the pipeline
        # exits nonzero, which under the callers' `set -e` would end the run.
        local fatal_lines
        fatal_lines=$(grep -E "$error_regex" "$LOG_FILE" \
            | grep -v -E 'UserWarning.*Stale file handle' || true)
        if [[ -n "$fatal_lines" ]]; then
            echo "FATAL ERROR DETECTED: The server log contains a fatal error pattern."
            # Call cleanup and exit (cleanup must be handled by the calling script's trap)
            exit 1
        fi

        if grep -Fq "$READY_MESSAGE" "$LOG_FILE" ; then
            echo "Server is ready."
            return 0
        fi
    done
}

# cleanUp: Stops the vLLM server process and deletes log files.
# Usage: cleanUp <MODEL_NAME>
cleanUp() {
    echo "Stopping the vLLM server and cleaning up log files..."
    # $1 is the MODEL_NAME passed as argument
    pkill -f "vllm serve $1"
    # Kill all processes related to vllm.
    pgrep -f -i vllm | xargs -r kill -9

    # Clean up log files. Use -f to avoid errors if files don't exist.
    rm -f "$LOG_FILE"
    rm -f "$BENCHMARK_LOG_FILE"
    echo "Cleanup complete."
}
