# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture
def model_name():
    """Choose gemma-27b as the test model as it has both full attention and 
    sliding window attention."""
    return "google/gemma-3-27b-it"


@pytest.fixture
def test_prompts():
    """Simple test prompts for hybrid kv cache testing."""
    return [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team?",
        "In Greek mythology, who is the god of the sea?",
        "What is the capital of Australia?",
        "What is the largest planet in our solar system?",
        "Who developed the theory of general relativity?",
    ]


@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
    return SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=True,
        logprobs=1,
    )


def _run_inference_with_config(
        model_name: str,
        test_prompts: list,
        sampling_params: SamplingParams,
        tensor_parallel_size: int = 4,
        kv_cache_dtype: str = "auto",
        enable_prefix_caching: bool = False,
        disable_hybrid_kv_cache_manager: bool = False) -> list:
    """Helper function to run inference with specified configuration."""

    llm = LLM(
        model=model_name,
        max_model_len=64,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=256,
        max_num_seqs=16,
        enable_prefix_caching=enable_prefix_caching,
        kv_cache_dtype=kv_cache_dtype,
        disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
    )

    try:
        outputs = llm.generate(test_prompts, sampling_params)
        return outputs
    finally:
        del llm
        # Wait for TPUs to be released
        time.sleep(10)


def test_hybrid_kv_cache(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test hybrid kv cache works on gemma vLLM models.
    """

    os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    # Test with hybrid kv cache alloctaion enabled.
    outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        disable_hybrid_kv_cache_manager=False,
    )

    # Verify we got outputs for all prompts
    assert len(outputs) == len(test_prompts)

    # Verify each output has generated text
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0

    print(f"✓ Hybrid KV cache test passed with {len(outputs)} outputs")


def _check_correctness(test_name: str, baseline_outputs: list,
                       hybrid_kvcache_outputs: list):
    """Verify outputs match between baseline and hybrid kvcache runs."""
    assert len(baseline_outputs) == len(hybrid_kvcache_outputs)

    text_matches = 0
    logprob_matches = 0
    total_compared_logprobs = 0
    max_logprob_diff = 0.0

    for i, (baseline, hybrid_kvcache_result) in enumerate(
            zip(baseline_outputs, hybrid_kvcache_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        hybrid_kvcache_text = hybrid_kvcache_result.outputs[0].text.strip()

        # Calculate word overlap for fuzzy matching
        baseline_words = set(baseline_text.split())
        hybrid_kvcache_words = set(hybrid_kvcache_text.split())
        overlap = baseline_words & hybrid_kvcache_words
        match_percent = len(overlap) / len(
            baseline_words) if baseline_words else 0

        if match_percent >= 0.7:
            text_matches += 1

        if baseline_text != hybrid_kvcache_text:
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Hybrid KV Cache: {hybrid_kvcache_text}")
            print(f"  Match percent: {match_percent:.2%}")

        # Compare log probabilities
        baseline_logprobs = baseline.outputs[0].logprobs
        hybrid_kvcache_logprobs = hybrid_kvcache_result.outputs[0].logprobs

        if baseline_logprobs is None or hybrid_kvcache_logprobs is None:
            continue

        assert len(baseline_logprobs) == len(hybrid_kvcache_logprobs), (
            f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(hybrid_kvcache_logprobs)}"
        )

        for token_idx, (base_lp, hybrid_kvcache_lp) in enumerate(
                zip(baseline_logprobs, hybrid_kvcache_logprobs)):
            if not (base_lp and hybrid_kvcache_lp):
                continue

            base_top_token = list(base_lp.keys())[0]
            hybrid_kvcache_top_token = list(hybrid_kvcache_lp.keys())[0]

            # Only compare logprobs if tokens match
            if base_top_token != hybrid_kvcache_top_token:
                continue

            base_logprob_val = base_lp[base_top_token].logprob
            hybrid_kvcache_logprob_val = hybrid_kvcache_lp[
                hybrid_kvcache_top_token].logprob
            diff = abs(base_logprob_val - hybrid_kvcache_logprob_val)
            max_logprob_diff = max(max_logprob_diff, diff)
            total_compared_logprobs += 1

            if diff < 0.1:
                logprob_matches += 1
            else:
                print(
                    f"  Logprob mismatch in prompt {i}, token {token_idx}: "
                    f"Baseline={base_logprob_val}, HybridKV={hybrid_kvcache_logprob_val}, "
                    f"Diff={diff:.6e}")

    # Report results
    logprob_match_rate = (logprob_matches / total_compared_logprobs
                          if total_compared_logprobs > 0 else 0)
    print(f"✓ {test_name} correctness test results:")
    print(f"  Text: {text_matches}/{len(baseline_outputs)} matches")
    print("  Target text match rate: >=60%")
    print(
        f"  Logprobs: {logprob_matches}/{total_compared_logprobs} ({logprob_match_rate:.2%}) matches (diff < 0.1)"
    )
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")

    # Validate thresholds
    text_match_rate = text_matches / len(baseline_outputs)
    assert text_match_rate >= 0.6, f"Text match rate {text_match_rate:.2%} is too low"

    if total_compared_logprobs > 0:
        assert logprob_match_rate >= 0.9, f"Logprob match rate {logprob_match_rate:.2%} is too low"


def test_hybrid_kv_cache_correctness(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that hybrid kv cache allocation produces consistent results compared 
    to standard kv cache allocation.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    small_prompts = test_prompts

    # Run baseline (no hybrid kv cache)
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        disable_hybrid_kv_cache_manager=True,
    )

    # Run with hybrid kv cache enabled.
    hybrid_kvcache_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        disable_hybrid_kv_cache_manager=False,
    )

    _check_correctness("Hybrid KV Cache", baseline_outputs,
                       hybrid_kvcache_outputs)
