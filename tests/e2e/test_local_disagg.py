# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict
from unittest.mock import patch

import pytest
import vllm.envs as vllm_envs
from vllm import LLM, EngineArgs, SamplingParams

from tpu_inference.core.core_tpu import DisaggEngineCore, DisaggEngineCoreProc


@pytest.fixture
def test_prompts():
    """Simple test prompts for disaggregated serving testing."""
    return [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team on the field at one time?",
        "In Greek mythology, who is the god of the sea?",
        "In what year did the Titanic sink?",
        "In which museum is the Mona Lisa displayed?",
        "Mount Everest is located in which mountain range?",
        "What ancient empire was ruled by Julius Caesar?",
        "What are the four fundamental forces of nature?",
        'What does "CPU" stand for?',
        'What does "HTML" stand for?',
        "What is the capital of Australia?",
        "What is the chemical symbol for gold?",
        "What is the currency of Switzerland?",
        "What is the distance from the Earth to the Sun called?",
        "What is the freezing point of water in Celsius?",
        "What is the hardest known natural substance on Earth?",
        "What is the largest planet in our solar system?",
        "What is the longest river in the world?",
        "What is the main function of the kidneys in the human body?",
        "What is the main ingredient in guacamole?",
        "What is the most spoken language in the world by number of native speakers?",
        "What is the process by which plants use sunlight to create food?",
        "Which country is known as the Land of the Rising Sun?",
        "Who developed the theory of general relativity?",
        'Who directed the original "Star Wars" trilogy?',
        "Who is credited with inventing the telephone?",
        "Who painted the ceiling of the Sistine Chapel?",
        "Who was the first female Prime Minister of the United Kingdom?",
        "Who was the first person to walk on the moon?",
        "Who wrote the American Declaration of Independence?",
        'Who wrote the novel "Pride and Prejudice"?',
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


def test_disaggregated_serving(test_prompts, sampling_params):
    """
    Test disaggregated serving end-to-end.

    Equivalent to:
    PREFILL_SLICES=4 DECODE_SLICES=4 python examples/offline_inference.py \
        --model=meta-llama/Meta-Llama-3.1-8B-Instruct --task=generate \
        --max_model_len=2048 --tensor_parallel_size 4
    """
    # Set environment variables for disaggregated serving
    # Using 4 slices for prefill and 4 for decode as requested
    # Note: The user example used PREFILL_SLICES=4 DECODE_SLICES=4
    # But usually slices are specified as "2x2" or similar if they are TPU topology.
    # However, disagg_utils.py _parse_slices handles "4" as well (1D).
    # We will stick to the user's example values.

    # We need to mock the environment variables for this test
    with patch.dict(
            os.environ, {
                "PREFILL_SLICES": "4",
                "DECODE_SLICES": "4",
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_XLA_CHECK_RECOMPILATION": "0"
            }):
        # Patch the EngineCore classes to use Disagg versions
        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), \
             patch("vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):

            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            os.system(f"rm -rf {vllm_envs.VLLM_XLA_CACHE_PATH}/*")
            engine_args = EngineArgs(
                model=model_name,
                max_model_len=2048,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.90,
                enforce_eager=False,
            )

            llm = LLM(**asdict(engine_args))

            try:
                outputs = llm.generate(test_prompts, sampling_params)

                # Verify outputs
                assert len(outputs) == len(test_prompts)
                for output in outputs:
                    assert len(output.outputs) > 0
                    assert len(output.outputs[0].text.strip()) > 0
                    print(f"Prompt: {output.prompt!r}")
                    print(f"Generated: {output.outputs[0].text!r}")

            finally:
                del llm
                time.sleep(10)
                pass


def _run_inference(model_name: str,
                   test_prompts: list,
                   sampling_params: SamplingParams,
                   tensor_parallel_size: int = 1,
                   is_disagg: bool = False,
                   prefill_slices: str = "4",
                   decode_slices: str = "4") -> list:
    """Helper function to run inference with specified configuration."""

    # Define the inner execution logic
    def run_inner():
        engine_args = EngineArgs(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.90,
            enforce_eager=False,
        )

        llm = LLM(**asdict(engine_args))
        try:
            return llm.generate(test_prompts, sampling_params)
        finally:
            del llm
            time.sleep(10)
            pass

    if is_disagg:
        # Mock environment variables and patch classes for disagg
        with patch.dict(
                os.environ, {
                    "PREFILL_SLICES": prefill_slices,
                    "DECODE_SLICES": decode_slices,
                    "SKIP_JAX_PRECOMPILE": "1",
                    "VLLM_XLA_CHECK_RECOMPILATION": "0"
                }):
            with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), \
                 patch("vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
                return run_inner()
    else:
        # Run standard inference
        # We still set some env vars to ensure consistent behavior if needed
        # but for baseline we want it as standard as possible.
        # However, to match the disagg run's potential jax settings:
        with patch.dict(os.environ, {
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_XLA_CHECK_RECOMPILATION": "0"
        }):
            return run_inner()


def test_disaggregated_serving_correctness(test_prompts, sampling_params):
    """
    Test that disaggregated serving produces consistent results compared to a baseline.
    """
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # Use a smaller subset of prompts for correctness testing
    small_prompts = test_prompts[:20]
    sampling_params.max_tokens = 16

    # Run baseline (standard execution)
    # We use tensor_parallel_size=4 to match the disagg resources if we assume
    # the user has enough chips, or if we are just mocking.
    # Since the original test used tp=4, we stick to it.
    print("Running Baseline Inference...")
    baseline_outputs = _run_inference(model_name=model_name,
                                      test_prompts=small_prompts,
                                      sampling_params=sampling_params,
                                      tensor_parallel_size=4,
                                      is_disagg=False)

    # Run disaggregated inference
    os.system(f"rm -rf {vllm_envs.VLLM_XLA_CACHE_PATH}/*")
    print("Running Disaggregated Inference...")

    disagg_outputs = _run_inference(model_name=model_name,
                                    test_prompts=small_prompts,
                                    sampling_params=sampling_params,
                                    tensor_parallel_size=4,
                                    is_disagg=True,
                                    prefill_slices="4",
                                    decode_slices="4")

    # Compare outputs
    assert len(baseline_outputs) == len(disagg_outputs)

    text_matches = 0
    text_mismatches = 0
    token_mismatches = 0

    for i, (baseline,
            disagg) in enumerate(zip(baseline_outputs, disagg_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        disagg_text = disagg.outputs[0].text.strip()

        # Check text output
        if baseline_text == disagg_text:
            text_matches += 1
        else:
            text_mismatches += 1
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Disagg:   {disagg_text}")

        # Check log probabilities (tokens) if available
        baseline_logprobs = baseline.outputs[0].logprobs
        disagg_logprobs = disagg.outputs[0].logprobs

        if baseline_logprobs is not None and disagg_logprobs is not None:
            assert len(baseline_logprobs) == len(disagg_logprobs), \
                f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(disagg_logprobs)}"

            for token_idx, (base_lp, disagg_lp) in enumerate(
                    zip(baseline_logprobs, disagg_logprobs)):
                if base_lp and disagg_lp:
                    # Compare the top token IDs
                    base_top_token = list(base_lp.keys())[0]
                    disagg_top_token = list(disagg_lp.keys())[0]

                    if base_top_token != disagg_top_token:
                        token_mismatches += 1
                        print(
                            f"Token mismatch in prompt {i}, token {token_idx}:"
                        )
                        print(f"  Baseline: {base_top_token}")
                        print(f"  Disagg:   {disagg_top_token}")

    print("✓ Correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Token mismatches in logprobs: {token_mismatches}")
    assert text_mismatches <= 5, f"Found {text_mismatches} text mismatches"
    assert token_mismatches <= 40, f"Found {token_mismatches} token mismatches"
