# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


# MODEL_IMPL_TYPE=vllm vllm serve Qwen/Qwen2.5-32B  --seed 42  
#   --disable-log-requests  --tensor-parallel-size 8 --max-model-len 2048 
#   --gpu-memory-utilization 0.96 --no-enable-prefix-caching --max-num-seqs 256 
#   --max-num-batched-tokens 4096

# python3 ./benchmarks/benchmark_serving.py 
#   --model Qwen/Qwen2.5-32B --dataset-name sonnet 
#   --dataset-path benchmarks/sonnet_4x.txt --sonnet-input-len 1800 
#   --sonnet-output-len 128 --ignore_eos

@dataclass
class TestConfig:
    """Configuration for SP test runs."""
    max_model_len: int = 2048 
    max_num_batched_tokens: int = 4096 
    max_num_seqs: int = 256 
    num_prompts: int = 16 

    @classmethod
    def for_correctness(cls) -> "TestConfig":
        return cls()

    @classmethod
    def for_performance(cls) -> "TestConfig":
        return cls()


@dataclass
class InferenceConfig:
    """Configuration for a single inference run."""
    model_name: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float = 0.96
    compilation_config: Optional[CompilationConfig] = None
    async_scheduling: bool = False
    additional_config: dict = field(default_factory=dict)
    kv_cache_dtype: str = "auto"


@pytest.fixture(autouse=True)
def setup_new_model_design():
    os.environ['MODEL_IMPL_TYPE'] = 'vllm'


def generate_test_prompts(num_prompts: int = 256) -> list[str]:
    base_texts = [
        # having a long prompt to trigger a edge case.
        "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Nine for Mortal Men doomed to die, One for the Dark Lord on his dark throne In the Land of Mordor where the Shadows lie. One Ring to rule them all, One Ring to find them, One Ring to bring them all and in the darkness bind them In the Land of Mordor where the Shadows lie.",
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
    return [
        f"Prompt {i}: {base_texts[i % len(base_texts)]}"
        for i in range(num_prompts)
    ]

@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
    return SamplingParams(
        temperature=0.0,
        max_tokens=16,
        ignore_eos=True,
        logprobs=1,
        seed=42
    )


def _run_inference(
    config: InferenceConfig,
    test_prompts: list[str],
    sampling_params: SamplingParams,
) -> tuple[list, float]:
    """Run inference with the given configuration."""
    llm = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
        compilation_config=config.compilation_config,
        async_scheduling=config.async_scheduling,
        additional_config=config.additional_config,
        kv_cache_dtype=config.kv_cache_dtype,
    )

    start_time = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed_time = time.time() - start_time

    del llm
    time.sleep(10)  # Wait for TPUs to be released
    return outputs, elapsed_time

import numpy as np
def _check_correctness(test_name: str, baseline_outputs: list,
                       sp_outputs: list):
    """Verify outputs match between baseline and sequence parallel runs."""
    assert len(baseline_outputs) == len(sp_outputs)

    text_matches = 0
    logprob_matches = 0
    total_compared_logprobs = 0
    max_logprob_diff = 0.0

    for i, (baseline, sp_result) in enumerate(zip(baseline_outputs,
                                                  sp_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        dp_text = dp_resp_resultsult.outputs[0].text.strip()

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
            print(f"  Baseline:          {baseline_text}")
            print(f"  Sequence Parallel: {dp_text}")
            print(f"  Match percent: {match_percent:.2%}")

        # Compare log probabilities
        baseline_logprobs = baseline.outputs[0].logprobs
        dp_logprobs = sp_result.outputs[0].logprobs

        if baseline_logprobs is None or dp_logprobs is None:
            continue

        assert len(baseline_logprobs) == len(dp_logprobs), (
            f"Logprobs length mismatch: {len(baseline_logprobs)} vs {len(dp_logprobs)}"
        )

        for token_idx, (base_lp,
                        sp_lp) in enumerate(zip(baseline_logprobs,
                                                dp_logprobs)):
            if not (base_lp and sp_lp):
                continue

            base_top_token = list(base_lp.keys())[0]
            dp_top_token = list(sp_lp.keys())[0]

            # Only compare logprobs if tokens match
            if base_top_token != dp_top_token:
                continue

            base_logprob_val = base_lp[base_top_token].logprob
            dp_logprob_val = sp_lp[dp_top_token].logprob
            diff = abs(base_logprob_val - dp_logprob_val)
            max_logprob_diff = max(max_logprob_diff, diff)
            total_compared_logprobs += 1

            if diff < 0.1:
                logprob_matches += 1
            else:
                print(f"  Logprob mismatch in prompt {i}, token {token_idx}: "
                      f"Baseline={base_logprob_val}, SP={dp_logprob_val}, "
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



def _check_performance(
    test_name: str,
    baseline_time: float,
    sp_time: float,
    num_prompts: int,
    min_speedup: float,
):
    """Verify sequence parallelism provides expected speedup."""
    speedup = baseline_time / sp_time if sp_time > 0 else 0

    print(f"✓ {test_name} performance test results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Sequence parallel time: {sp_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(
        f"  Sequence parallel throughput: {num_prompts/sp_time:.2f} prompts/s")

    assert speedup >= min_speedup, (
        f"Sequence parallelism did not provide expected speedup "
        f"({min_speedup:.2f}x): {speedup:.2f}x")


def _test_sequence_parallelism(
    sampling_params: SamplingParams,
    check_correctness: bool = True,
    check_performance: bool = True,
):
    """Correctness and performance test for sequence parallelism."""
    cfg = TestConfig.for_performance(
    ) if check_performance else TestConfig.for_correctness()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    tensor_parallel_size = 8
    model_name = "Qwen/Qwen2.5-32B"

    # Run with sequence parallelism
    sp_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
        compilation_config=CompilationConfig(pass_config={"enable_sp": True}),
    )
    sp_outputs, sp_time = _run_inference(sp_config, test_prompts,
                                         sampling_params)

    # Run baseline (no sequence parallelism)
    baseline_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )
    baseline_outputs, baseline_time = _run_inference(baseline_config,
                                                     test_prompts,
                                                     sampling_params)

    if check_correctness:
        _check_correctness("Sequence parallelism", baseline_outputs,
                           sp_outputs)

    if check_performance:
        _check_performance(
            "Sequence parallelism",
            baseline_time,
            sp_time,
            len(test_prompts),
            min_speedup=1.1,
        )


def test_sp_correctness(sampling_params: SamplingParams):
    _test_sequence_parallelism(
        sampling_params=sampling_params,
        check_correctness=True,
        check_performance=False,
    )


# def test_sp_performance(sampling_params: SamplingParams):
#     _test_sequence_parallelism(
#         sampling_params=sampling_params,
#         check_correctness=False,
#         check_performance=True,
#     )
