# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import pytest
from vllm import LLM, SamplingParams


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
                               data_parallel_size: int = 1,
                               enable_expert_parallel: bool = False,
                               additional_config: dict = {},
                               kv_cache_dtype: str = "auto",
                               enable_prefix_caching: bool = False) -> list:
    """Helper function to run inference with specified configuration."""

    llm = LLM(
        model=model_name,
        max_model_len=128,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=0.40,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        enable_prefix_caching=enable_prefix_caching,
        additional_config=additional_config,
        kv_cache_dtype=kv_cache_dtype,
        async_scheduling=False,
        enable_expert_parallel=enable_expert_parallel,
    )

    try:
        outputs = llm.generate(test_prompts, sampling_params)
        return outputs
    finally:
        del llm
        # Wait for TPUs to be released
        time.sleep(5)


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "jax_model",
            "model": None,
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "ep": False,
            "impl": "jax",
            "additional_config": {}
        },
        {
            "id": "vllm_model",
            "model": None,
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "ep": False,
            "impl": "vllm",
            "additional_config": {}
        },
        {
            "id": "ep",
            "model": "Qwen/Qwen1.5-MoE-A2.7B",
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "ep": True,
            "impl": "vllm",
            "additional_config": {
                "sharding": {
                    "sharding_strategy": {
                        "expert_parallelism": 2
                    }
                }
            }
        },
        {
            "id": "dp",
            "model": None,
            "tp": 1,
            "pp": 2,
            "dp": 2,
            "ep": False,
            "impl": "jax",
            "additional_config": {}
        },
        {
            "id": "tp",
            "model": None,
            "tp": 2,
            "pp": 2,
            "dp": 1,
            "ep": False,
            "impl": "jax",
            "additional_config": {}
        },
        {
            "id": "ep_tp",
            "model": "Qwen/Qwen1.5-MoE-A2.7B",
            "tp": 2,
            "pp": 2,
            "dp": 1,
            "ep": True,
            "impl": "vllm",
            "additional_config": {
                "sharding": {
                    "sharding_strategy": {
                        "expert_parallelism": 2,
                    }
                }
            }
        },
    ],
    ids=lambda x: x["id"],
)
def test_pipeline_parallel_configs(
    case,
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test various pipeline parallel configurations.
    """
    impl = case["impl"]
    if impl == 'vllm':
        os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    else:
        os.environ.pop('MODEL_IMPL_TYPE', None)

    model = case["model"]
    actual_model = model if model is not None else model_name

    outputs = _run_inference_with_config(
        model_name=actual_model,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=case["tp"],
        pipeline_parallel_size=case["pp"],
        data_parallel_size=case["dp"],
        enable_expert_parallel=case["ep"],
        additional_config=case["additional_config"],
    )

    assert len(outputs) == len(test_prompts)
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text.strip()) > 0


def _check_correctness(test_name: str, baseline_outputs: list,
                       dp_outputs: list):
    """Verify outputs match between baseline and data parallel runs."""
    assert len(baseline_outputs) == len(dp_outputs)

    text_matches = 0
    logprob_matches = 0
    total_compared_logprobs = 0
    max_logprob_diff = 0.0

    for i, (baseline, dp_result) in enumerate(zip(baseline_outputs,
                                                  dp_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        dp_text = dp_result.outputs[0].text.strip()

        # Calculate word overlap for fuzzy matching
        baseline_words = set(baseline_text.split())
        dp_words = set(dp_text.split())
        overlap = baseline_words & dp_words
        match_percent = len(overlap) / len(
            baseline_words) if baseline_words else 0

        if match_percent >= 0.7:
            text_matches += 1

        if baseline_text != dp_text:
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Data Parallel: {dp_text}")
            print(f"  Match percent: {match_percent:.2%}")

        # Compare log probabilities
        baseline_logprobs = baseline.outputs[0].logprobs
        dp_logprobs = dp_result.outputs[0].logprobs

        if baseline_logprobs is None or dp_logprobs is None:
            continue

        assert len(baseline_logprobs) == len(dp_logprobs), (
            f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(dp_logprobs)}"
        )

        for token_idx, (base_lp,
                        dp_lp) in enumerate(zip(baseline_logprobs,
                                                dp_logprobs)):
            if not (base_lp and dp_lp):
                continue

            base_top_token = list(base_lp.keys())[0]
            dp_top_token = list(dp_lp.keys())[0]

            # Only compare logprobs if tokens match
            if base_top_token != dp_top_token:
                continue

            base_logprob_val = base_lp[base_top_token].logprob
            dp_logprob_val = dp_lp[dp_top_token].logprob
            diff = abs(base_logprob_val - dp_logprob_val)
            max_logprob_diff = max(max_logprob_diff, diff)
            total_compared_logprobs += 1

            if diff < 0.1:
                logprob_matches += 1
            else:
                print(f"  Logprob mismatch in prompt {i}, token {token_idx}: "
                      f"Baseline={base_logprob_val}, DP={dp_logprob_val}, "
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

    _check_correctness("Pipeline Parallelism", baseline_outputs, pp_outputs)
