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
    base_text = (
        "The rapid advancement of artificial intelligence has transformed "
        "numerous industries and continues to reshape our understanding of "
        "technology's potential ") * 100
    return [
        f"Prompt {i}: {base_text} What are your thoughts on this topic?"
        for i in range(num_prompts)
    ]

@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
    return SamplingParams(
        temperature=0.0,
        max_tokens=128,
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


def _check_correctness(test_name: str, baseline_outputs: list,
                       sp_outputs: list):
    """Verify outputs match between baseline and sequence parallel runs."""
    assert len(baseline_outputs) == len(sp_outputs)

    for i, (baseline, sp_result) in enumerate(zip(baseline_outputs,
                                                  sp_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        sp_text = sp_result.outputs[0].text.strip()

        assert baseline_text == sp_text, (
            f"Text mismatch found in prompt {i}:\n"
            f"  Baseline: {baseline_text}\n"
            f"  Sequence Parallel: {sp_text}")

    print(f"✓ {test_name} correctness test passed for {len(baseline_outputs)} "
          "prompts.")


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


def test_sp_performance(sampling_params: SamplingParams):
    _test_sequence_parallelism(
        sampling_params=sampling_params,
        check_correctness=False,
        check_performance=True,
    )
