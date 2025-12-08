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

        if grep -Eq "$error_regex" "$LOG_FILE"; then
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
