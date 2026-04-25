#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Verification script for the TPU KV-cache offload connector.
#
# Runs three sanity checks:
#   1. Qwen3-0.6B (non-hybrid)         — attn-only fast path; bit-exact vs baseline.
#   2. Qwen3.5-35B (hybrid attn+mamba) — HMA mode; bit-exact vs baseline (no real
#      D2H/H2D — vLLM has spare HBM, no eviction).
#   3. Qwen3.5-35B + prefix caching    — HMA mode; per-chunk payload now carries
#      both attn block AND the request's mamba state; both are scattered back on
#      load. Bit-exact vs baseline proves the multi-group save/load round-trip
#      preserves correctness.
#
# All three pass = TPU offload connector handles non-hybrid + hybrid (attn+mamba)
# end-to-end and beats GPU OffloadingConnector (which still asserts single-group
# in scheduler.py and crashes on hybrid).

set -e
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

OFFLOAD_CFG='{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector","kv_role":"kv_both"}'

run_and_diff() {
    local label="$1"
    shift
    local args=("$@")
    echo "--- $label ---"
    local logfile
    logfile=$(mktemp -t verify_offload_XXXXXX.log)

    # Baseline (no connector)
    SKIP_JAX_PRECOMPILE=1 \
    TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY=1 \
    python examples/offline_inference.py \
        "${args[@]}" --temperature 0.0 \
        > "${logfile}.baseline" 2>&1
    grep -A1 "^Prompt:" "${logfile}.baseline" > "${logfile}.baseline.out"

    # Offload connector enabled
    SKIP_JAX_PRECOMPILE=1 \
    TPU_OFFLOAD_SKIP_JAX_PRECOMPILE=1 \
    TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY=1 \
    python examples/offline_inference.py \
        "${args[@]}" --temperature 0.0 \
        --kv-transfer-config "$OFFLOAD_CFG" \
        > "${logfile}.offload" 2>&1
    grep -A1 "^Prompt:" "${logfile}.offload" > "${logfile}.offload.out"

    if diff -q "${logfile}.baseline.out" "${logfile}.offload.out" > /dev/null; then
        echo "  $label: outputs MATCH baseline (bit-exact)"
    else
        echo "  $label: outputs DIVERGE (see ${logfile}.{baseline,offload}.out)"
        diff "${logfile}.baseline.out" "${logfile}.offload.out" | head -20
        return 1
    fi
}

run_and_diff "Qwen3-0.6B (non-hybrid)" \
    --model Qwen/Qwen3-0.6B --tensor-parallel-size 1 \
    --max-model-len 1024 --max-num-batched-tokens 1024 --block-size 128 \
    --max-tokens 16

run_and_diff "Qwen3.5-35B (hybrid attn+mamba, no prefix caching)" \
    --model Qwen/Qwen3.5-35B-A3B-FP8 --tensor-parallel-size 1 \
    --max-model-len 5120 --max-num-batched-tokens 16384 --block-size 256 \
    --max-tokens 8 --no-disable-hybrid-kv-cache-manager

run_and_diff "Qwen3.5-35B (hybrid + prefix caching → real D2H/H2D)" \
    --model Qwen/Qwen3.5-35B-A3B-FP8 --tensor-parallel-size 1 \
    --max-model-len 5120 --max-num-batched-tokens 16384 --block-size 256 \
    --max-tokens 8 --no-disable-hybrid-kv-cache-manager \
    --enable-prefix-caching

echo
echo "All verifications passed."
