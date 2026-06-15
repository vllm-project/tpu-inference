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

    # Add any substrings or warnings that shouldn't trigger a build failure here
    local ignore_patterns=(
        # This warning happens when JAX unable to read cache. If we have a GCSFuse
        # folder to store cache, a race condition where the other build was creating
        # the same cache enrty could produce this logs plus 
        # "OSError: [Errno 116] Stale file handle", which contians the fatal logs ketwords. 
        # However, this should not affect the server run as in this case, 
        # JAX will just recompile the cache.
        "UserWarning: Error reading persistent compilation cache entry"
    )

    local error_regex
    error_regex=$(IFS=\|; echo "${fatal_error_patterns[*]}")

    local ignore_regex
    ignore_regex=$(IFS=\|; echo "${ignore_patterns[*]}")

    while true; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        sleep 5

        if [[ "$elapsed_time" -ge "$TIMEOUT_SECONDS" ]]; then
            echo "TIMEOUT: Waited $elapsed_time seconds (limit was $TIMEOUT_SECONDS). The string '$READY_MESSAGE' was NOT found."
            # Call cleanup and exit (cleanup must be handled by the calling script's trap)
            exit 1
        fi

        # Filter out ignored patterns FIRST, then check what remains for fatals
        if grep -Ev "$ignore_regex" "$LOG_FILE" | grep -Eq "$error_regex"; then
            echo "FATAL ERROR DETECTED: The server log contains a fatal error pattern."
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
