# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture(autouse=True)
def setup_new_model_design():
    os.environ['NEW_MODEL_DESIGN'] = '1'
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'


@pytest.fixture
def test_prompts(num_prompts: int = 256) -> list:
    base_text = (
        "The rapid advancement of artificial intelligence has transformed numerous industries "
        "and continues to reshape our understanding of technology's potential. Machine learning "
        "algorithms have become increasingly sophisticated, enabling computers to perform tasks "
        "that were once thought to require human intelligence. From natural language processing "
        "to computer vision, AI systems are now capable of understanding context, recognizing "
        "patterns, and making decisions with remarkable accuracy. " *
        20  # Repeat to reach ~1k tokens
    )
    return [
        f"Prompt {i}: {base_text} What are your thoughts on this topic?"
        for i in range(num_prompts)
    ]


@pytest.fixture
def sampling_params():
    return SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=True,
        logprobs=1,
    )


def _run_inference_with_config(model_name: str,
                               test_prompts: list,
                               sampling_params: SamplingParams,
                               tensor_parallel_size: int = 1,
                               data_parallel_size: int = 1,
                               additional_config: dict = {},
                               kv_cache_dtype: str = "auto",
                               enable_prefix_caching: bool = False,
                               async_scheduling: bool = False,
                               max_model_len: int = 32,
                               max_num_batched_tokens: int = 128,
                               max_num_seqs: int = 16,
                               gpu_memory_utilization: float = 0.90,
                               trace_dir: str = None) -> list:

    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        additional_config=additional_config,
        kv_cache_dtype=kv_cache_dtype,
        async_scheduling=async_scheduling,
    )

    start_time = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed_time = time.time() - start_time

    del llm
    time.sleep(10)
    return outputs, elapsed_time


def _check_performance(test_name: str, baseline_time: float, dp_time: float,
                       num_prompts: int, tol: float):

    speedup = baseline_time / dp_time if dp_time > 0 else 0

    print(f"✓ {test_name} performance test results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Data parallel time: {dp_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(f"  Data parallel throughput: {num_prompts/dp_time:.2f} prompts/s")

    assert speedup >= tol, f"Data parallelism did not provide expected speedup ({tol:.2f}x): {speedup:.2f}x"


def _check_correctness(test_name, baseline_outputs, dp_outputs):

    assert len(baseline_outputs) == len(dp_outputs)

    text_matches = 0
    logprob_matches = 0
    total_compared_logprobs = 0
    max_logprob_diff = 0.0

    for i, (baseline, dp_result) in enumerate(zip(baseline_outputs,
                                                  dp_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        dp_text = dp_result.outputs[0].text.strip()

        baseline_words = baseline_text.split()
        dp_words = dp_text.split()
        overlap_set = set(baseline_words) & set(dp_words)
        match_percent = len(overlap_set) / len(set(baseline_words))
        if match_percent >= 0.7:
            text_matches += 1

        # Check text output
        if baseline_text != dp_text:
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Data Parallel: {dp_text}")
            print(f"  Match percent: {match_percent:.2%}")

        # Check log probabilities
        baseline_logprobs = baseline.outputs[0].logprobs
        dp_logprobs = dp_result.outputs[0].logprobs

        if baseline_logprobs is not None and dp_logprobs is not None:
            # Compare log probabilities for each token
            assert len(baseline_logprobs) == len(dp_logprobs), \
                f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(dp_logprobs)}"

            for token_idx, (base_lp, dp_lp) in enumerate(
                    zip(baseline_logprobs, dp_logprobs)):
                # Get the top logprob value for the selected token
                if base_lp and dp_lp:
                    # Get the top token's logprob from each
                    base_top_token = list(base_lp.keys())[0]
                    dp_top_token = list(dp_lp.keys())[0]

                    # Only compare logprobs if tokens match
                    if base_top_token == dp_top_token:
                        base_logprob_val = base_lp[base_top_token].logprob
                        dp_logprob_val = dp_lp[dp_top_token].logprob

                        # Calculate absolute difference
                        diff = abs(base_logprob_val - dp_logprob_val)
                        max_logprob_diff = max(max_logprob_diff, diff)

                        total_compared_logprobs += 1
                        # Count as match if difference is small
                        if diff < 0.1:
                            logprob_matches += 1
                        else:
                            print(
                                f"  Logprob mismatch in prompt {i}, token {token_idx}: "
                                f"Baseline logprob={base_logprob_val}, "
                                f"Data Parallel logprob={dp_logprob_val}, "
                                f"Diff={diff:.6e}")

    print(f"✓ {test_name} correctness test results:")
    print(f"  Text: {text_matches} matches (match percent >= 70%)")
    print(
        f"  Logprobs: {logprob_matches}/{total_compared_logprobs} ({logprob_matches / total_compared_logprobs:.2%}) matches (diff < 0.1)"
    )
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")

    # Allow for some variance due to potential numerical differences
    # but most outputs should match with greedy sampling
    text_match_rate = text_matches / len(baseline_outputs)
    assert text_match_rate >= 0.9, f"Text match rate {text_match_rate:.2%} is too low"

    # Log probabilities should match for most matching tokens
    if total_compared_logprobs > 0:
        logprob_match_rate = logprob_matches / total_compared_logprobs
        assert logprob_match_rate >= 0.9, f"Logprob match rate {logprob_match_rate:.2%} is too low"


def test_attention_data_parallelism(
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Correctness and performance test for attention DP     
    """

    os.environ['MODEL_IMPL_TYPE'] = "vllm"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Configuration for long sequences
    max_model_len = 2048
    max_num_batched_tokens = 4096
    max_num_seqs = 128

    # Run with attn_dp=2 tp=2
    dp_outputs, dp_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=4,
        async_scheduling=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": 1
                }
            }
        })

    # Run baseline (tp=2)
    baseline_outputs, baseline_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=2,
        data_parallel_size=1,
        async_scheduling=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    _check_correctness("Attention data parallelism", baseline_outputs,
                       dp_outputs)

    # Different hardware gives different performance. This test runs on v6e_8
    _check_performance("Attention data parallelism",
                       baseline_time,
                       dp_time,
                       len(test_prompts),
                       tol=1.1)


def test_data_parallelism(
    sampling_params: SamplingParams,
    test_prompts: list,
):
    """
    Correctness and performance test for model DP 
    """
    os.environ['MODEL_IMPL_TYPE'] = "flax_nnx"

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Configuration for long sequences
    max_model_len = 2048
    max_num_batched_tokens = 4096
    max_num_seqs = 128

    # Run with data parallelism (dp=2, tp=1)
    dp_outputs, dp_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
        async_scheduling=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    # Run baseline (tp=1)
    baseline_outputs, baseline_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=1,
        async_scheduling=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    _check_correctness("Data parallelism", baseline_outputs, dp_outputs)

    # Test is too small to see significant speedup, mainly for testing regression
    _check_performance("Data parallelism",
                       baseline_time,
                       dp_time,
                       len(test_prompts),
                       tol=1.1)
