# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import dataclass, field

import pytest
from vllm import LLM, SamplingParams


@dataclass
class TestConfig:
    """Configuration for TP test runs."""
    max_model_len: int = 512
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 16
    num_prompts: int = 16

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
    pipeline_parallel_size: int = 1
    max_model_len: int = 512
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 16
    additional_config: dict = field(default_factory=dict)
    gpu_memory_utilization: float = 0.80
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = False


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
        pipeline_parallel_size=config.pipeline_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        additional_config=config.additional_config,
        kv_cache_dtype=config.kv_cache_dtype,
        enable_prefix_caching=config.enable_prefix_caching,
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
    tp_time: float,
    num_prompts: int,
    min_speedup: float,
):
    """Verify tensor parallelism provides expected speedup."""
    speedup = baseline_time / tp_time if tp_time > 0 else 0

    print(f"✓ {test_name} performance test results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Tensor parallel time: {tp_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(f"  Tensor parallel throughput: {num_prompts/tp_time:.2f} prompts/s")

    assert speedup >= min_speedup, (
        f"Tensor parallelism did not provide expected speedup "
        f"({min_speedup:.2f}x): {speedup:.2f}x")


def _test_tensor_parallelism_performance(
    sampling_params: SamplingParams,
    model_name: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int = 1,
    additional_config: dict | None = None,
    min_speedup: float = 1.05,
):
    """Performance test for tensor parallelism."""
    cfg = TestConfig.for_performance()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    tp_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens //
        tensor_parallel_size,
        max_num_seqs=cfg.max_num_seqs // tensor_parallel_size,
        additional_config=additional_config or {},
    )
    _, tp_time = _run_inference(tp_config, test_prompts, sampling_params)

    baseline_config = InferenceConfig(
        model_name=model_name,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )
    _, baseline_time = _run_inference(baseline_config, test_prompts,
                                      sampling_params)

    _check_performance(
        "Tensor parallelism",
        baseline_time,
        tp_time,
        len(test_prompts),
        min_speedup=min_speedup,
    )


def test_tp_performance_jax_model(sampling_params: SamplingParams):
    """Performance test for tensor parallelism on JAX models."""
    os.environ['MODEL_IMPL_TYPE'] = 'flax_nnx'
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    _test_tensor_parallelism_performance(
        sampling_params=sampling_params,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        min_speedup=1.05,
    )


def test_tp_performance_vllm_model(sampling_params: SamplingParams):
    """Performance test for tensor parallelism on vLLM models."""
    os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    _test_tensor_parallelism_performance(
        sampling_params=sampling_params,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        min_speedup=1.05,
    )
