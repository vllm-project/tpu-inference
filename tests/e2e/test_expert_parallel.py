# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict, dataclass

import pytest
from vllm import LLM, EngineArgs, SamplingParams

# Hardcoded baseline times (seconds) for 512 prompts on TPU v7x-8.
# Measured on 2026-03-20 with vllm LKG aa84e43c + tpu-inference main.
# Tests pass if EP time is within REGRESSION_THRESHOLD of these baselines.
EP_FUSED_BASELINE_TIME = 3.40
EP_GMM_BASELINE_TIME = 2.07
REGRESSION_THRESHOLD = 0.15  # 15% regression tolerance


@dataclass
class TestConfig:
    """Configuration for EP test runs."""
    max_model_len: int = 512
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 512
    num_prompts: int = 512


@dataclass
class InferenceConfig:
    """Configuration for a single inference run."""
    model_name: str
    tensor_parallel_size: int = 4
    enable_expert_parallel: bool = True
    max_model_len: int = 512
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 512
    gpu_memory_utilization: float = 0.95


def generate_test_prompts(num_prompts: int = 512) -> list[str]:
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

    # Warmup
    llm.generate(test_prompts[:8], sampling_params)

    start_time = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed_time = time.time() - start_time

    del llm
    time.sleep(10)
    return outputs, elapsed_time


def _check_no_regression(
    test_name: str,
    baseline_time: float,
    actual_time: float,
    num_prompts: int,
    threshold: float = REGRESSION_THRESHOLD,
):
    """Verify EP time has not regressed beyond threshold vs hardcoded baseline."""
    max_allowed = baseline_time * (1 + threshold)
    regression_pct = ((actual_time - baseline_time) / baseline_time) * 100

    print(f"\n{test_name} performance results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Actual time:   {actual_time:.2f}s")
    print(
        f"  Max allowed:   {max_allowed:.2f}s (baseline + {threshold*100:.0f}%)"
    )
    print(f"  Delta:         {regression_pct:+.1f}%")
    print(f"  Throughput:    {num_prompts/actual_time:.2f} prompts/s")

    assert actual_time <= max_allowed, (
        f"{test_name} regressed by {regression_pct:.1f}% "
        f"(actual {actual_time:.2f}s > allowed {max_allowed:.2f}s)")


def test_ep_fused_performance(sampling_params: SamplingParams):
    """Test EP with fused MoE kernel does not regress vs hardcoded baseline."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'
    os.environ['USE_MOE_EP_KERNEL'] = '1'

    model_name = os.environ.get("EP_MODEL_NAME", "Qwen/Qwen1.5-MoE-A2.7B")
    cfg = TestConfig()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    try:
        ep_config = InferenceConfig(
            model_name=model_name,
            max_model_len=cfg.max_model_len,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.max_num_seqs,
        )
        _, ep_time = _run_inference(ep_config, test_prompts, sampling_params)

        _check_no_regression(
            "EP Fused",
            EP_FUSED_BASELINE_TIME,
            ep_time,
            cfg.num_prompts,
        )
    finally:
        del os.environ['USE_MOE_EP_KERNEL']


def test_ep_gmm_performance(sampling_params: SamplingParams):
    """Test EP with GMM kernel does not regress vs hardcoded baseline.

    Uses OLMoE-1B-7B (64 experts, power-of-2) instead of Qwen2MoE
    (60 experts) because the GMM EP kernel requires num_tokens*topk
    to be divisible by the tile size, which only works when
    num_experts_per_shard is a power of 2.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    model_name = os.environ.get("EP_GMM_MODEL_NAME",
                                "allenai/OLMoE-1B-7B-0924")
    cfg = TestConfig()
    test_prompts = generate_test_prompts(cfg.num_prompts)

    ep_config = InferenceConfig(
        model_name=model_name,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )
    _, ep_time = _run_inference(ep_config, test_prompts, sampling_params)

    _check_no_regression(
        "EP GMM",
        EP_GMM_BASELINE_TIME,
        ep_time,
        cfg.num_prompts,
    )
