# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams


@pytest.fixture
def model_name():
    """Small model for faster testing."""
    return "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(autouse=True)
def setup_new_model_design():
    """Automatically set NEW_MODEL_DESIGN=True for all tests."""
    os.environ['NEW_MODEL_DESIGN'] = 'True'


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
    )


def _run_inference_with_config(model_name: str,
                               test_prompts: list,
                               sampling_params: SamplingParams,
                               tensor_parallel_size: int = 1,
                               data_parallel_size: int = 1,
                               additional_config: dict = {},
                               kv_cache_dtype: str = "auto",
                               enable_prefix_caching: bool = False) -> list:
    """Helper function to run inference with specified configuration."""

    # Create LLM args using parser-based approach similar to offline_inference.py
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=128,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
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


def test_model_data_parallelism(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test model-wise data parallelism where data=2 in the mesh axis.
    This test verifies that the model can run with data parallelism enabled,
    duplicating the entire model across 2 data parallel workers.

    Equivalent to:
    python examples/offline_inference.py --tensor_parallel_size=4 --data_parallel_size=2
    """
    # Test with data parallelism enabled
    outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
    )

    # Verify we got outputs for all prompts
    assert len(outputs) == len(test_prompts)

    # Verify each output has generated text
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0

    print(f"✓ Model data parallelism test passed with {len(outputs)} outputs")


def test_attention_data_parallelism(
    model_name: str,
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
    additional_config = {
        "sharding": {
            "sharding_strategy": {
                "enable_dp_attention": 1
            }
        }
    }

    # Test with attention data parallelism enabled
    outputs = _run_inference_with_config(
        model_name=model_name,
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
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that data parallelism produces consistent results compared to a baseline.
    This test compares outputs from a single-device run with data parallel runs
    to ensure correctness.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    # Use a smaller subset of prompts for correctness testing
    small_prompts = test_prompts[:10]

    # Run baseline (no data parallelism)
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    # Run with model data parallelism
    dp_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=small_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        data_parallel_size=2,
    )

    # Compare outputs - they should be identical for greedy sampling
    assert len(baseline_outputs) == len(dp_outputs)

    matches = 0
    mismatches = 0

    for baseline, dp_result in zip(baseline_outputs, dp_outputs):
        baseline_text = baseline.outputs[0].text.strip()
        dp_text = dp_result.outputs[0].text.strip()

        if baseline_text == dp_text:
            matches += 1
        else:
            mismatches += 1
            print("Mismatch found:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Data Parallel: {dp_text}")

    print(f"✓ Correctness test: {matches} matches, {mismatches} mismatches")

    # Allow for some variance due to potential numerical differences
    # but most outputs should match with greedy sampling
    match_rate = matches / len(baseline_outputs)
    assert match_rate >= 0.9, f"Match rate {match_rate:.2%} is too low"
