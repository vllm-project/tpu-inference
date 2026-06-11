# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os
import random
import time
from dataclasses import dataclass, field
from typing import Union

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture(autouse=True)
def setup_new_model_design():
    os.environ['NEW_MODEL_DESIGN'] = '1'


@dataclass
class InferenceConfig:
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B"
    impl_type: str = "vllm"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    max_model_len: int = 256
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.85
    additional_config: dict = field(default_factory=dict)


def _run_inference_and_return_outputs(
        config: InferenceConfig,
        prompts: list[str],
        sampling_params: Union[SamplingParams, list[SamplingParams]],
        enable_export: bool = True):
    os.environ['MODEL_IMPL_TYPE'] = config.impl_type

    llm = LLM(
        model=config.model_name,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        enable_expert_parallel=config.enable_expert_parallel,
        additional_config=config.additional_config,
        async_scheduling=False,
    )

    if enable_export:
        llm.llm_engine.vllm_config.model_config.enable_return_routed_experts = True

    try:
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        latency = time.time() - start_time
        return outputs, latency
    finally:
        del llm
        gc.collect()
        time.sleep(15)


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
    assert text_match_rate >= 0.7, f"Text match rate {text_match_rate:.2%} is too low"

    if total_compared_logprobs > 0:
        assert logprob_match_rate >= 0.9, f"Logprob match rate {logprob_match_rate:.2%} is too low"


def _check_performance(
    test_name: str,
    baseline_time: float,
    target_time: float,
    num_prompts: int,
    min_speedup: float,
):
    """Verify continue decoding provides expected performance speedup."""
    speedup = baseline_time / target_time if target_time > 0 else 0

    print(f"✓ {test_name} performance test results:")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Baseline time: {baseline_time:.2f}s")
    print(f"  Target time: {target_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {num_prompts/baseline_time:.2f} prompts/s")
    print(f"  Target throughput: {num_prompts/target_time:.2f} prompts/s")

    assert speedup >= min_speedup, (
        f"Continue decode did not provide expected speedup "
        f"({min_speedup:.2f}x): {speedup:.2f}x")


def generate_test_prompts(num_prompts: int = 4) -> list[str]:
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
        f"Prompt {i}: {base_text} Write an essay on this topic."
        for i in range(num_prompts)
    ]


@pytest.fixture(params=[0.0, 1.0])
def sampling_params(request):
    return SamplingParams(
        temperature=request.param,
        max_tokens=16,
        ignore_eos=True,
        logprobs=1,
        seed=42,
    )


# Scenarios 1-4: Correctness on MMLU-equivalent topologies with Llama-3.1-8B-Instruct
@pytest.mark.parametrize(
    "matrix_case",
    [
        {
            "id": "correctness_non_DP_JAX",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "impl": "flax_nnx",
            "tp": 8,
            "dp": 1,
            "max_model_len": 1024,
            "max_num_batched_tokens": 8192,
        },
        {
            "id": "correctness_DP_JAX",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "impl": "flax_nnx",
            "tp": 1,
            "dp": 8,
            "max_model_len": 1024,
            "max_num_batched_tokens": 8192,
        },
        {
            "id": "correctness_non_DP_torchax",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "impl": "vllm",
            "tp": 8,
            "dp": 1,
            "max_model_len": 1024,
            "max_num_batched_tokens": 8192,
        },
        {
            "id": "correctness_DP_torchax",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "impl": "vllm",
            "tp": 1,
            "dp": 8,
            "max_model_len": 1024,
            "max_num_batched_tokens": 8192,
        },
    ],
    ids=lambda x: x["id"],
)
def test_continue_decode_correctness_matrix(matrix_case, sampling_params):
    """Verify continue_decode correctness across benchmarking script execution scenarios."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    config = InferenceConfig(
        model_name=matrix_case["model"],
        impl_type=matrix_case["impl"],
        tensor_parallel_size=matrix_case["tp"],
        data_parallel_size=matrix_case["dp"],
        max_model_len=matrix_case["max_model_len"],
        max_num_batched_tokens=matrix_case["max_num_batched_tokens"],
        additional_config={},
    )

    prompts = generate_test_prompts(4)

    # 1. Generate ground-truth outputs using baseline single-step decoding
    config.additional_config = {"enable_continue_decode": False}
    baseline_outputs, _ = _run_inference_and_return_outputs(config,
                                                            prompts,
                                                            sampling_params,
                                                            enable_export=True)

    # 2. Generate outputs using optimized multi-step continue decode
    config.additional_config = {"enable_continue_decode": True}
    continue_decode_outputs, _ = _run_inference_and_return_outputs(
        config, prompts, sampling_params, enable_export=True)

    # 3. Call shared correctness verification logic matching test_data_parallel.py
    _check_correctness(f"Continue decode ({matrix_case['id']})",
                       baseline_outputs, continue_decode_outputs)

    # 5. Verify expert routing metadata for MoE architectures
    if "MoE" in config.model_name:
        for cd_out in continue_decode_outputs:
            cd_req = cd_out.outputs[0]
            assert cd_req.routed_experts is not None, "routed_experts tensor must be populated for MoE models when enabled."
            assert len(
                cd_req.routed_experts.shape
            ) == 3, f"Expected 3D expert tensor, got shape {cd_req.routed_experts.shape}"

            # Verify alignment length: prompt tokens + generated tokens - 1
            num_prompt_tokens = len(cd_out.prompt_token_ids)
            num_gen_tokens = len(cd_req.token_ids)
            expected_tokens_dim = num_prompt_tokens + num_gen_tokens - 1
            actual_tokens_dim = cd_req.routed_experts.shape[0]

            assert actual_tokens_dim == expected_tokens_dim, (
                f"Exported expert IDs token dimension mismatch. Expected {expected_tokens_dim} "
                f"(prompt len {num_prompt_tokens} + gen len {num_gen_tokens} - 1), got {actual_tokens_dim}"
            )


# Scenarios 5-6: Performance evaluation on MMLU workloads with Qwen/Qwen1.5-MoE-A2.7B
@pytest.mark.parametrize(
    "perf_case",
    [
        {
            "id": "performance_DP_torchax_random_length",
            "impl": "vllm",
            "tp": 4,
            "dp": 2,
            "random_len": True,
            "max_model_len": 1024,
            "max_num_batched_tokens": 2048,
        },
        {
            "id": "performance_DP_torchax_nonrandom_length",
            "impl": "vllm",
            "tp": 4,
            "dp": 2,
            "random_len": False,
            "max_model_len": 1024,
            "max_num_batched_tokens": 2048,
        },
    ],
    ids=lambda x: x["id"],
)
def test_continue_decode_performance_overhead(perf_case):
    """Verify that continue_decode performance overhead matches benchmark scenarios on MMLU workloads."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    config = InferenceConfig(
        model_name="Qwen/Qwen1.5-MoE-A2.7B",
        impl_type=perf_case["impl"],
        tensor_parallel_size=perf_case["tp"],
        data_parallel_size=perf_case["dp"],
        max_model_len=perf_case.get("max_model_len", 2048),
        max_num_batched_tokens=perf_case.get("max_num_batched_tokens", 2048),
        max_num_seqs=256,
        additional_config={},
    )

    prompts = generate_test_prompts(4)

    # Configure sampling params based on random length scenario requirement on MMLU prompts
    if perf_case["random_len"]:
        # Simulate --random-range-ratio=0.8 on target output generation lengths matching default random dataset base length (128)
        base_len = 128
        ratio = 0.8
        low = max(int(base_len * (1 - ratio)), 1)
        high = int(base_len * (1 + ratio))
        # Set fixed seed for repeatable testing
        random.seed(42)
        sampling_params = [
            SamplingParams(temperature=0.0,
                           max_tokens=random.randint(low, high),
                           ignore_eos=True) for _ in prompts
        ]
    else:
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=128,
                                         ignore_eos=True)

    # Measure baseline execution without continue decode (standard step-by-step decoding)
    config.additional_config = {"enable_continue_decode": False}
    _, latency_baseline = _run_inference_and_return_outputs(
        config, prompts, sampling_params, enable_export=False)

    # Measure execution with continue decode enabled
    config.additional_config = {"enable_continue_decode": True}
    _, latency_continue_decode = _run_inference_and_return_outputs(
        config, prompts, sampling_params, enable_export=False)

    _check_performance(
        f"Continue decode ({perf_case['id']})",
        latency_baseline,
        latency_continue_decode,
        len(prompts),
        min_speedup=1.05,
    )
