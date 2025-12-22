# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams


@pytest.fixture(autouse=True)
def setup_new_model_design():
    """Automatically set NEW_MODEL_DESIGN=1 for all tests."""
    os.environ['NEW_MODEL_DESIGN'] = '1'


def _generate_prompts(prompt_type: str = "short",
                      num_prompts: int = 10,
                      num_repeats: int = 20) -> list[str]:
    """Generate test prompts of different types.
    
    Args:
        prompt_type: Type of prompts to generate ("short" or "long")
        num_prompts: Number of prompts to generate (used for long prompts)
        num_repeats: Number of times to repeat base text (used for long prompts)
    
    Returns:
        List of prompt strings
    """
    if prompt_type == "short":
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
    elif prompt_type == "long":
        base_text = (
            "The rapid advancement of artificial intelligence has transformed numerous industries "
            "and continues to reshape our understanding of technology's potential. Machine learning "
            "algorithms have become increasingly sophisticated, enabling computers to perform tasks "
            "that were once thought to require human intelligence. From natural language processing "
            "to computer vision, AI systems are now capable of understanding context, recognizing "
            "patterns, and making decisions with remarkable accuracy. " *
            num_repeats  # Repeat to reach ~1k tokens
        )
        return [
            f"Prompt {i}: {base_text} What are your thoughts on this topic?"
            for i in range(num_prompts)
        ]
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


@pytest.fixture
def test_prompts():
    """Simple test prompts for data parallelism testing."""
    return _generate_prompts(prompt_type="short")


@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
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
                               measure_time: bool = False,
                               max_model_len: int = 32,
                               max_num_batched_tokens: int = 128,
                               max_num_seqs: int = 16,
                               gpu_memory_utilization: float = 0.98) -> list:
    """Helper function to run inference with specified configuration.

    Returns:
        If measure_time=True: (outputs, elapsed_time) tuple
        If measure_time=False: outputs list
    """

    # Create LLM args using parser-based approach similar to offline_inference.py
    engine_args = EngineArgs(
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

    engine_args_dict = asdict(engine_args)
    llm = LLM(**engine_args_dict)

    try:
        start_time = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        elapsed_time = time.time() - start_time
        if measure_time:
            return outputs, elapsed_time
        else:
            return outputs
    finally:
        del llm
        # Wait for TPUs to be released
        time.sleep(5)

def _check_performance(test_name: str, baseline_time: float,
                               dp_time: float, num_prompts: int):

    speedup = baseline_time / dp_time if dp_time > 0 else 0

    print(f"✓ {test_name} results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Data parallel time: {dp_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(
        f"  Data parallel throughput: {num_prompts/dp_time:.2f} prompts/s"
    )
    
    assert speedup >= 1.5, f"Data parallelism did not provide expected speedup: {speedup:.2f}x"


def test_data_parallelism_performance(sampling_params: SamplingParams, ):
    """
    Test that data parallelism provides performance improvements compared to baseline.
    This test measures the execution time with 128 prompts of length ~1k tokens.

    Note: This is a performance benchmark test with large prompts.
    """
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['MODEL_IMPL_TYPE'] = 'flax_nnx'

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Generate 128 prompts of approximately 1k tokens each
    long_prompts = _generate_prompts(prompt_type="long", num_prompts=128)

    # Configuration for long sequences
    max_model_len = 2048
    max_num_batched_tokens = 4096
    max_num_seqs = 64

    # Run baseline (no data parallelism) with timing
    baseline_outputs, baseline_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=long_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=1,
        async_scheduling=True,
        measure_time=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    # Run with model data parallelism and async scheduling with timing
    dp_outputs, dp_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=long_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
        async_scheduling=True,
        measure_time=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    _check_performance(
        "Performance test",
        baseline_time,
        dp_time,
        len(long_prompts)
    )


def test_attention_data_parallelism_performance(sampling_params: SamplingParams, ):
    """
    Test that attention data parallelism provides performance improvements compared to baseline.
    This test measures the execution time with 128 prompts of length ~1k tokens.

    Note: This is a performance benchmark test with large prompts.
    """
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['MODEL_IMPL_TYPE'] = 'vllm'

    model_name = "Qwen/Qwen1.5-MoE-A2.7B"

    # Generate 128 prompts of approximately 1k tokens each
    long_prompts = _generate_prompts(prompt_type="long", num_prompts=128)

    # Configuration for long sequences
    max_model_len = 2048
    max_num_batched_tokens = 4096
    max_num_seqs = 64

    # Run baseline (no data parallelism) with timing
    baseline_outputs, baseline_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=long_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=1,
        async_scheduling=False,
        measure_time=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    # Run with model data parallelism and async scheduling with timing
    dp_outputs, dp_time = _run_inference_with_config(
        model_name=model_name,
        test_prompts=long_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
        async_scheduling=True,
        measure_time=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )

    _check_performance(
        "Attention data parallelism performance test",
        baseline_time,
        dp_time,
        len(long_prompts)
    )




def _check_correctness(baseline_outputs, dp_outputs):
    # Compare outputs - they should be identical for greedy sampling
    assert len(baseline_outputs) == len(dp_outputs)

    text_matches = 0
    text_mismatches = 0
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

    print("✓ Correctness test results:")
    print(
        f"  Text: {text_matches} matches, {text_mismatches} mismatches (match percent >= 70%)"
    )
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


def test_data_parallelism_correctness(
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that data parallelism produces consistent results compared to a baseline.
    This test compares outputs from a single-device run with data parallel runs
    to ensure correctness, including log probabilities.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'
    os.environ['MODEL_IMPL_TYPE'] = "flax_nnx"

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # Use a smaller subset of prompts for correctness testing
    small_prompts = test_prompts[:10]

    # Run baseline (no data parallelism)
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=1,
        async_scheduling=True,
    )

    # Run with model data parallelism and async scheduling
    dp_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
        async_scheduling=True,
    )

    _check_correctness(baseline_outputs, dp_outputs)


def test_attention_data_parallelism_correctness(
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that attention data parallelism produces consistent results compared to a baseline.
    This test compares outputs from a single-device run with attention data parallel run
    to ensure correctness, including log probabilities.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'
    os.environ['MODEL_IMPL_TYPE'] = "vllm"
    model_name = "Qwen/Qwen3-30B-A3B"

    # Use a smaller subset of prompts for correctness testing
    small_prompts = test_prompts[:10]

    # Run baseline (no data parallelism)
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=4,
        data_parallel_size=1,
        async_scheduling=False,
    )

    # Run with attention data parallelism
    dp_outputs = _run_inference_with_config(model_name=model_name,
                                            test_prompts=small_prompts,
                                            sampling_params=sampling_params,
                                            tensor_parallel_size=8,
                                            async_scheduling=False,
                                            additional_config={
                                                "sharding": {
                                                    "sharding_strategy": {
                                                        "enable_dp_attention":
                                                        1
                                                    }
                                                }
                                            })

    _check_correctness(baseline_outputs, dp_outputs)
