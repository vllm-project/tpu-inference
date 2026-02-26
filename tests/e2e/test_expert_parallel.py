# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict, dataclass

import pytest
from vllm import LLM, EngineArgs, SamplingParams


@dataclass
class TestConfig:
    """Configuration for EP test runs."""
    max_model_len: int = 512
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 16
    num_prompts: int = 16

    @classmethod
    def for_performance(cls) -> "TestConfig":
        return cls(
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=512,
            num_prompts=512,
        )


@dataclass
class InferenceConfig:
    """Configuration for a single inference run."""
    model_name: str
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    max_model_len: int = 512
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.95


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
    engine_args = EngineArgs(
        model=config.model_name,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=1,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        enable_prefix_caching=False,
        kv_cache_dtype="auto",
        enable_expert_parallel=config.enable_expert_parallel,
    )

    engine_args_dict = asdict(engine_args)
    llm = LLM(**engine_args_dict)

    start_time = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed_time = time.time() - start_time

    del llm
    time.sleep(10)
    return outputs, elapsed_time


def _check_performance(
    test_name: str,
    baseline_time: float,
    ep_time: float,
    num_prompts: int,
    min_speedup: float,
):
    """Verify expert parallelism provides expected speedup."""
    speedup = baseline_time / ep_time if ep_time > 0 else 0

    print(f"✓ {test_name} performance test results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Expert parallel time: {ep_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(f"  Expert parallel throughput: {num_prompts/ep_time:.2f} prompts/s")

    assert speedup >= min_speedup, (
        f"Expert parallelism did not provide expected speedup "
        f"({min_speedup:.2f}x): {speedup:.2f}x")


def _test_expert_parallelism_performance(
    sampling_params: SamplingParams,
    use_fused_kernel: bool,
    model_name: str | None = None,
):
    """Performance test for expert parallelism."""
    if model_name is None:
        model_name = os.environ.get("EP_MODEL_NAME", "Qwen/Qwen1.5-MoE-A2.7B")

    cfg = TestConfig.for_performance()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    if use_fused_kernel:
        os.environ['USE_MOE_EP_KERNEL'] = '1'

    try:
        # Run EP (TP=4 + EP)
        ep_config = InferenceConfig(
            model_name=model_name,
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            max_model_len=cfg.max_model_len,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.max_num_seqs,
        )
        _, ep_time = _run_inference(ep_config, test_prompts, sampling_params)

        # Run baseline (TP=1)
        baseline_config = InferenceConfig(
            model_name=model_name,
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            max_model_len=cfg.max_model_len,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.max_num_seqs,
        )
        _, baseline_time = _run_inference(baseline_config, test_prompts,
                                          sampling_params)

        kernel_name = "EP Fused" if use_fused_kernel else "EP GMM"
        _check_performance(
            f"Expert parallelism ({kernel_name})",
            baseline_time,
            ep_time,
            len(test_prompts),
            min_speedup=0.6,
        )
    finally:
        if use_fused_kernel:
            del os.environ['USE_MOE_EP_KERNEL']


def test_ep_fused_performance(sampling_params: SamplingParams):
    """Test expert parallelism performance with fused MoE EP kernel."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    _test_expert_parallelism_performance(sampling_params,
                                         use_fused_kernel=True)


def test_ep_gmm_performance(sampling_params: SamplingParams):
    """Test expert parallelism performance with GMM kernel.

    Uses OLMoE-1B-7B (64 experts, power-of-2) instead of Qwen2MoE
    (60 experts) because the GMM EP kernel requires num_tokens*topk
    to be divisible by the tile size, which only works when
    num_experts_per_shard is a power of 2.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    gmm_model = os.environ.get("EP_GMM_MODEL_NAME", "allenai/OLMoE-1B-7B-0924")
    _test_expert_parallelism_performance(sampling_params,
                                         use_fused_kernel=False,
                                         model_name=gmm_model)
