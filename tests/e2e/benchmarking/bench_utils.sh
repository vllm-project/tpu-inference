#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# BENCHMARK UTILITY FUNCTIONS
# This file is sourced by various performance scripts (e.g., mlperf.sh,
# llama_guard_perf_recipe.sh) to share common functions.
# -----------------------------------------------------------------------------

# Resolve the two repositories independently. Agent Stack checks out the build
# under /workspace while the image keeps vLLM outside that mount; bare-metal
# images historically keep both repositories under /workspace.
resolve_benchmark_workspace() {
    local requested_root="${1:-}"
    local caller_script="${BASH_SOURCE[1]}"
    local checkout_root
    checkout_root=$(cd -- "$(dirname -- "$caller_script")/../../.." &>/dev/null && pwd)

    tpu_inf_dir="${TPU_INFERENCE_DIR:-$checkout_root}"
    if [[ ! -d "$tpu_inf_dir/tpu_inference" ]]; then
        echo "ERROR: Could not resolve the tpu-inference checkout from $caller_script" >&2
        return 1
    fi

    vllm_dir=""
    local candidate
    for candidate in \
        "${requested_root:+$requested_root/vllm}" \
        /tpu-inference/workspace/vllm \
        /workspace/vllm \
        /vllm; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            vllm_dir="$candidate"
            break
        fi
    done
    if [[ -z "$vllm_dir" ]]; then
        echo "ERROR: Could not find the vLLM checkout" >&2
        return 1
    fi

    if [[ -z "$requested_root" || ! -d "$requested_root" ]]; then
        root_dir=$(dirname -- "$vllm_dir")
    else
        root_dir="$requested_root"
    fi

    echo "Using TPU Inference checkout at $tpu_inf_dir"
    echo "Using vLLM checkout at $vllm_dir"
    echo "Using benchmark workspace at $root_dir"
}

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
