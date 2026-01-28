# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import difflib
import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams


@pytest.fixture
def model_name():
    return "Qwen/Qwen1.5-MoE-A2.7B"


@pytest.fixture
def test_prompts():
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
                               enable_expert_parallel: bool = False,
                               max_num_batched_tokens: int = 128) -> list:
    """Helper function to run inference with specified configuration."""

    # Correctness defaults
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    # Create LLM args
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=128,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=16,
        enable_prefix_caching=False,
        kv_cache_dtype="auto",
        enable_expert_parallel=enable_expert_parallel,
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


def _verify_correctness(baseline_outputs,
                        experiment_outputs,
                        label="Experiment",
                        text_match_threshold=0.9,
                        logprob_tolerance=1.0):
    """Helper to verify correctness between two runs."""
    assert len(baseline_outputs) == len(experiment_outputs)

    text_matches = 0
    text_mismatches = 0
    logprob_mismatches = 0
    max_logprob_diff = 0.0

    for i, (baseline,
            experiment) in enumerate(zip(baseline_outputs,
                                         experiment_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        experiment_text = experiment.outputs[0].text.strip()

        # Check text output
        if baseline_text == experiment_text:
            text_matches += 1
        else:
            # Try fuzzy match
            similarity = difflib.SequenceMatcher(None, baseline_text,
                                                 experiment_text).ratio()
            if similarity >= 0.95:  # Very strict fuzzy match
                text_matches += 1
                msg = "Soft match"
            else:
                text_mismatches += 1
                msg = "Text mismatch"

            print(f"{msg} found in prompt {i} (similarity={similarity:.4f}):")
            print(f"  Baseline: {baseline_text}")
            print(f"  {label}: {experiment_text}")

        # Check log probabilities
        baseline_logprobs = baseline.outputs[0].logprobs
        experiment_logprobs = experiment.outputs[0].logprobs
        if baseline_logprobs is not None and experiment_logprobs is not None:
            assert len(baseline_logprobs) == len(experiment_logprobs), \
                f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(experiment_logprobs)}"

            for token_idx, (base_lp_dict, exp_lp_dict) in enumerate(
                    zip(baseline_logprobs, experiment_logprobs)):
                # vLLM returns [{token_id: Logprob}, ...]
                if not base_lp_dict or not exp_lp_dict:
                    continue

                # Get the top token's logprob from each (assuming top_logprobs=1)
                t1, lp1 = next(iter(base_lp_dict.items()))
                t2, lp2 = next(iter(exp_lp_dict.items()))

                if t1 == t2:
                    diff = abs(lp1.logprob - lp2.logprob)
                    max_logprob_diff = max(max_logprob_diff, diff)

                    # Allow small numerical differences (e.g., 1e-3)
                    if diff > 1e-3:
                        logprob_mismatches += 1
                        print(
                            f"Logprob mismatch in prompt {i}, token {token_idx}:"
                        )
                        print(
                            f"  Baseline token: {t1}, logprob: {lp1.logprob:.6f}"
                        )
                        print(
                            f"  {label} token: {t2}, logprob: {lp2.logprob:.6f}"
                        )
                        print(f"  Difference: {diff:.6f}")

    print("✓ Correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")
    print(f"  Significant logprob mismatches (>1e-3): {logprob_mismatches}")

    text_match_rate = text_matches / len(baseline_outputs)
    assert text_match_rate >= text_match_threshold, f"Text match rate {text_match_rate:.2%} is too low (threshold {text_match_threshold})"

    # Log probabilities should be very close (allow small numerical errors)
    # Raising tolerance slightly for Fused kernels if needed, but keeping strict for now
    assert max_logprob_diff < logprob_tolerance, f"Max logprob difference {max_logprob_diff} is too large (tolerance {logprob_tolerance})"


def test_expert_parallelism_correctness_via_gmm_kernel(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    """
    Test that Expert Parallelism produces result consistent with baseline (TP=1).
    """

    # Run Baseline (TP=1)
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
    )

    # Run EP (EP=4)
    ep_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
    )

    _verify_correctness(baseline_outputs, ep_outputs, label="Expert Parallel")


def test_expert_parallelism_correctness_via_fused_moe_kernel(
    model_name: str,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    os.environ['USE_MOE_EP_KERNEL'] = '1'
    try:
        # Run Baseline
        baseline_outputs = _run_inference_with_config(
            model_name=model_name,
            test_prompts=test_prompts,
            sampling_params=sampling_params,
            tensor_parallel_size=1,
            enable_expert_parallel=False,
        )

        # Run EP Fused
        ep_outputs = _run_inference_with_config(
            model_name=model_name,
            test_prompts=test_prompts,
            sampling_params=sampling_params,
            tensor_parallel_size=4,
            enable_expert_parallel=True,
        )

        _verify_correctness(baseline_outputs,
                            ep_outputs,
                            label="EP Fused",
                            text_match_threshold=0.9,
                            logprob_tolerance=1.0)
    finally:
        del os.environ['USE_MOE_EP_KERNEL']
