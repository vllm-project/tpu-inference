# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import dataclass, field

import pytest
from vllm import LLM, SamplingParams


@dataclass
class TestConfig:
    """Configuration for DP test runs."""
    max_model_len: int = 512
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 16
    num_prompts: int = 16

    @classmethod
    def for_correctness(cls) -> "TestConfig":
        return cls()

    @classmethod
    def for_performance(cls) -> "TestConfig":
        return cls(
            max_model_len=1024,
            max_num_batched_tokens=2048,
            max_num_seqs=2048,
            num_prompts=2048,
        )


@dataclass
class InferenceConfig:
    """Configuration for a single inference run."""
    model_name: str
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    async_scheduling: bool = False
    max_model_len: int = 512
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 16
    additional_config: dict = field(default_factory=dict)
    gpu_memory_utilization: float = 0.80


@pytest.fixture(autouse=True)
def setup_new_model_design():
    os.environ['NEW_MODEL_DESIGN'] = '1'


def generate_test_prompts(num_prompts: int = 256) -> list[str]:
    base_text = (
        "The rapid advancement of artificial intelligence has transformed "
        "numerous industries and continues to reshape our understanding of "
        "technology's potential. Machine learning algorithms have become "
        "increasingly sophisticated, enabling computers to perform tasks "
        "that were once thought to require human intelligence. From natural "
        "language processing to computer vision, AI systems are now capable "
        "of understanding context, recognizing patterns, and making decisions "
        "with remarkable accuracy. ")
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


def _run_inference(
    config: InferenceConfig,
    test_prompts: list[str],
    sampling_params: SamplingParams,
) -> tuple[list, float]:
    """Run inference with the given configuration."""
    llm = LLM(
        model=config.model_name,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        additional_config=config.additional_config,
        async_scheduling=config.async_scheduling,
    )

    start_time = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed_time = time.time() - start_time

    del llm
    time.sleep(10)
    return outputs, elapsed_time


def _check_performance(
    test_name: str,
    baseline_time: float,
    dp_time: float,
    num_prompts: int,
    min_speedup: float,
):
    """Verify data parallelism provides expected speedup."""
    speedup = baseline_time / dp_time if dp_time > 0 else 0

    print(f"✓ {test_name} performance test results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Data parallel time: {dp_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(f"  Data parallel throughput: {num_prompts/dp_time:.2f} prompts/s")

    assert speedup >= min_speedup, (
        f"Data parallelism did not provide expected speedup "
        f"({min_speedup:.2f}x): {speedup:.2f}x")


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


def _test_attention_data_parallelism(
    sampling_params: SamplingParams,
    check_correctness: bool = True,
    check_performance: bool = True,
):
    """Correctness and performance test for attention DP."""
    os.environ['MODEL_IMPL_TYPE'] = "vllm"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    cfg = TestConfig.for_performance(
    ) if check_performance else TestConfig.for_correctness()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    # Run with attn_dp=2 tp=2
    dp_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=4,
        async_scheduling=False,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens // 2,
        max_num_seqs=cfg.max_num_seqs // 2,
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": 1
                }
            }
        },
    )
    dp_outputs, dp_time = _run_inference(dp_config, test_prompts,
                                         sampling_params)

    # Run baseline (tp=2)
    baseline_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=2,
        data_parallel_size=1,
        async_scheduling=False,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )
    baseline_outputs, baseline_time = _run_inference(baseline_config,
                                                     test_prompts,
                                                     sampling_params)

    if check_correctness:
        _check_correctness("Attention data parallelism", baseline_outputs,
                           dp_outputs)

    if check_performance:
        _check_performance(
            "Attention data parallelism",
            baseline_time,
            dp_time,
            len(test_prompts),
            min_speedup=0.6,
        )


def _test_data_parallelism(
    sampling_params: SamplingParams,
    check_correctness: bool = True,
    check_performance: bool = True,
):
    """Correctness and performance test for model DP."""
    os.environ['MODEL_IMPL_TYPE'] = "flax_nnx"
    model_name = "meta-llama/Meta-Llama-3-8B"

    cfg = TestConfig.for_performance(
    ) if check_performance else TestConfig.for_correctness()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    # Run with data parallelism (dp=2, tp=1)
    dp_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        data_parallel_size=2,
        async_scheduling=True,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens // 2,
        max_num_seqs=cfg.max_num_seqs // 2,
    )
    dp_outputs, dp_time = _run_inference(dp_config, test_prompts,
                                         sampling_params)

    # Run baseline (tp=1)
    baseline_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        data_parallel_size=1,
        async_scheduling=True,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )
    baseline_outputs, baseline_time = _run_inference(baseline_config,
                                                     test_prompts,
                                                     sampling_params)

    if check_correctness:
        _check_correctness("Data parallelism", baseline_outputs, dp_outputs)

    if check_performance:
        _check_performance(
            "Data parallelism",
            baseline_time,
            dp_time,
            len(test_prompts),
            min_speedup=1.1,
        )


def test_dp_correctness(sampling_params: SamplingParams):
    """Test data parallelism correctness without compilation."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    _test_data_parallelism(sampling_params,
                           check_correctness=True,
                           check_performance=False)
    _test_attention_data_parallelism(sampling_params,
                                     check_correctness=True,
                                     check_performance=False)


def test_dp_performance(sampling_params: SamplingParams):
    """Test data parallelism performance with compilation."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    _test_data_parallelism(sampling_params,
                           check_correctness=False,
                           check_performance=True)
    _test_attention_data_parallelism(sampling_params,
                                     check_correctness=False,
                                     check_performance=True)
