# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams


@pytest.fixture
def model_name():
    """Choose LLama3 8b as the test model as it supports PP on jax model impl."""
    return "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture
def test_prompts():
    """Simple test prompts for data parallelism testing."""
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


def _run_inference_with_config(model_name: str,
                               test_prompts: list,
                               sampling_params: SamplingParams,
                               tensor_parallel_size: int = 1,
                               pipeline_parallel_size: int = 1,
                               additional_config: dict = {},
                               kv_cache_dtype: str = "auto",
                               enable_prefix_caching: bool = False) -> list:
    """Helper function to run inference with specified configuration."""

    # Create LLM args using parser-based approach similar to offline_inference.py
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=128,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        enable_prefix_caching=enable_prefix_caching,
        additional_config=additional_config,
        kv_cache_dtype=kv_cache_dtype,
    )

    engine_args_dict = asdict(engine_args)
    llm = LLM(**engine_args_dict)

    try:
        outputs = llm.generate(test_prompts, sampling_params)
        return outputs
    finally:
        del llm
        # Wait for TPUs to be released
        time.sleep(5)


@pytest.mark.skip(reason="PP is not fully enabled.")
def test_pipeline_parallelism_jax_model(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test pipline parallelism works on Jax models

    Equivalent to:
    python examples/offline_inference.py --tensor_parallel_size=1 --pipeline_parallel_size=2
    """
    # Test with pipeline parallelism enabled
    outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    # Verify we got outputs for all prompts
    assert len(outputs) == len(test_prompts)

    # Verify each output has generated text
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0

    print(
        f"✓ Pipeline Parallelism Jax model test passed with {len(outputs)} outputs"
    )


@pytest.mark.skip(reason="PP is not fully enabled.")
def test_pipeline_parallelism_vllm_model(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test pipline parallelism works on vLLM models, and it also works with
    with tensor parallelism.

    Equivalent to:
    MODEL_IMPL_TYPE=vllm python examples/offline_inference.py --tensor_parallel_size=1 --pipeline_parallel_size=2
    """

    os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    # Test with data parallelism enabled
    outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    # Verify we got outputs for all prompts
    assert len(outputs) == len(test_prompts)

    # Verify each output has generated text
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0

    print(
        f"✓ Pipeline Parallelism vLLM model test passed with {len(outputs)} outputs"
    )


@pytest.mark.skip(reason="PP is not fully enabled.")
def test_pipeline_parallelism_jax_model_correctness(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that pipeline parallelism produces consistent results compared to a baseline.
    This test compares outputs from a single-device run with pipeline parallel runs
    to ensure correctness, including log probabilities.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    # Use a smaller subset of prompts for correctness testing
    small_prompts = test_prompts[:10]

    # Run baseline (no PP)
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    # Run with model data parallelism and async scheduling
    pp_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
    )

    # Compare outputs - in theory they should be identical for greedy sampling
    # in reality there may be some differences, but overall the outputs should
    # be very similar.

    # an example:
    # prompt: What is the capital of Australia?
    # both answers should be acceptable.
    # The capital of Australia is Canberra. It is located in the Australian Capital Territory (ACT) and is home to many
    # Canberra is the capital of Australia. It is located in the Australian Capital Territory (ACT) and is home to
    assert len(baseline_outputs) == len(pp_outputs)

    text_matches = 0
    text_mismatches = 0
    logprob_mismatches = 0
    max_logprob_diff = 0.0

    for i, (baseline, pp_result) in enumerate(zip(baseline_outputs,
                                                  pp_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        pp_text = pp_result.outputs[0].text.strip()

        # Check text output
        if baseline_text == pp_text:
            text_matches += 1
        else:
            text_mismatches += 1
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Pipeline Parallel: {pp_text}")

        # Check log probabilities
        baseline_logprobs = baseline.outputs[0].logprobs
        pp_logprobs = pp_result.outputs[0].logprobs
        if baseline_logprobs is not None and pp_logprobs is not None:
            # Compare log probabilities for each token
            assert len(baseline_logprobs) == len(pp_logprobs), \
                f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(pp_logprobs)}"
            for token_idx, (base_lp, pp_lp) in enumerate(
                    zip(baseline_logprobs, pp_logprobs)):
                # Get the top logprob value for the selected token
                if base_lp and pp_lp:
                    # Get the top token's logprob from each
                    base_top_token = list(base_lp.keys())[0]
                    pp_top_token = list(pp_lp.keys())[0]

                    base_logprob_val = base_lp[base_top_token].logprob
                    pp_logprob_val = pp_lp[pp_top_token].logprob

                    # Calculate absolute difference
                    diff = abs(base_logprob_val - pp_logprob_val)
                    max_logprob_diff = max(max_logprob_diff, diff)

                    # Allow small numerical differences (e.g., 1e-3)
                    if diff > 1e-3:
                        logprob_mismatches += 1
                        print(
                            f"Logprob mismatch in prompt {i}, token {token_idx}:"
                        )
                        print(
                            f"  Baseline token: {base_top_token}, logprob: {base_logprob_val:.6f}"
                        )
                        print(
                            f"  PP token: {pp_top_token}, logprob: {pp_logprob_val:.6f}"
                        )
                        print(f"  Difference: {diff:.6f}")

    print("✓ Correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")
    print(f"  Significant logprob mismatches (>1e-3): {logprob_mismatches}")

    # Allow for some variance due to potential numerical differences
    # but most outputs should match with greedy sampling
    text_match_rate = text_matches / len(baseline_outputs)
    assert text_match_rate >= 0.9, f"Text match rate {text_match_rate:.2%} is too low"

    # Log probabilities should be very close (allow small numerical errors)
    assert max_logprob_diff < 1, f"Max logprob difference {max_logprob_diff} is too large"
