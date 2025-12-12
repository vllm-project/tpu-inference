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
                               data_parallel_size: int = 1,
                               additional_config: dict = {},
                               kv_cache_dtype: str = "auto",
                               enable_prefix_caching: bool = False,
                               async_scheduling: bool = False) -> list:
    """Helper function to run inference with specified configuration."""

    # Create LLM args using parser-based approach similar to offline_inference.py
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=0.98,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        enable_prefix_caching=enable_prefix_caching,
        additional_config=additional_config,
        kv_cache_dtype=kv_cache_dtype,
        async_scheduling=async_scheduling,
        enforce_eager=True,
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


@pytest.mark.parametrize("model_impl_type", ["vllm", "flax_nnx"])
def test_model_data_parallelism(
    test_prompts: list,
    sampling_params: SamplingParams,
    model_impl_type: str,
):
    """
    Test model-wise data parallelism where data=2 in the mesh axis.
    This test verifies that the model can run with data parallelism enabled,
    duplicating the entire model across 2 data parallel workers.

    Equivalent to:
    python examples/offline_inference.py --tensor_parallel_size=4 --data_parallel_size=2
    """
    # Use Llama 1B for this test
    test_model = "meta-llama/Llama-3.2-1B-Instruct"
    os.environ['MODEL_IMPL_TYPE'] = model_impl_type

    # Test with data parallelism enabled
    outputs = _run_inference_with_config(
        model_name=test_model,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
        async_scheduling=False,
    )

    # Verify we got outputs for all prompts
    assert len(outputs) == len(test_prompts)

    # Verify each output has generated text
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0

    print(f"✓ Model data parallelism test passed with {len(outputs)} outputs")


def test_attention_data_parallelism(
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test attention data parallelism where only the attention layer gets duplicated,
    attn_dp=2 in the mesh axis. This is useful when num_kv_heads < TP to avoid
    wasting KV cache memory.

    Equivalent to:
    python examples/offline_inference.py --tensor_parallel_size=8 --kv-cache-dtype=fp8 \
        --additional_config='{"sharding":{"sharding_strategy": {"enable_dp_attention":1}}}'
    """
    # Use Llama 1B for this test
    test_model = "Qwen/Qwen3-0.6B"

    additional_config = {
        "sharding": {
            "sharding_strategy": {
                "enable_dp_attention": 1
            }
        }
    }

    # Test with attention data parallelism enabled
    outputs = _run_inference_with_config(
        model_name=test_model,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=8,
        data_parallel_size=1,
        additional_config=additional_config,
        kv_cache_dtype="fp8",
    )

    # Verify we got outputs for all prompts
    assert len(outputs) == len(test_prompts)

    # Verify each output has generated text
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0

    print(
        f"✓ Attention data parallelism test passed with {len(outputs)} outputs"
    )


def test_data_parallelism_correctness(
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that data parallelism produces consistent results compared to a baseline.
    This test compares outputs from a single-device run with data parallel runs
    to ensure correctness, including log probabilities.
    """
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'
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

    # Compare outputs - they should be identical for greedy sampling
    assert len(baseline_outputs) == len(dp_outputs)

    text_matches = 0
    text_mismatches = 0
    logprob_mismatches = 0
    max_logprob_diff = 0.0

    for i, (baseline, dp_result) in enumerate(zip(baseline_outputs,
                                                  dp_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        dp_text = dp_result.outputs[0].text.strip()

        # Check text output
        if baseline_text == dp_text:
            text_matches += 1
        else:
            text_mismatches += 1
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Data Parallel: {dp_text}")

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

                    base_logprob_val = base_lp[base_top_token].logprob
                    dp_logprob_val = dp_lp[dp_top_token].logprob

                    # Calculate absolute difference
                    diff = abs(base_logprob_val - dp_logprob_val)
                    max_logprob_diff = max(max_logprob_diff, diff)

                    # Allow small numerical differences
                    if diff > 0.15:
                        logprob_mismatches += 1
                        print(
                            f"Logprob mismatch in prompt {i}, token {token_idx}:"
                        )
                        print(
                            f"  Baseline token: {base_top_token}, logprob: {base_logprob_val:.6f}"
                        )
                        print(
                            f"  DP token: {dp_top_token}, logprob: {dp_logprob_val:.6f}"
                        )
                        print(f"  Difference: {diff:.6f}")

    print("✓ Correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")
    print(f"  Significant logprob mismatches (>0.15): {logprob_mismatches}")

    # Allow for some variance due to potential numerical differences
    # but most outputs should match with greedy sampling
    text_match_rate = text_matches / len(baseline_outputs)
    assert text_match_rate >= 0.9, f"Text match rate {text_match_rate:.2%} is too low"

    # Log probabilities should be very close (allow small numerical errors)
    assert max_logprob_diff < 0.15, f"Max logprob difference {max_logprob_diff} is too large"
