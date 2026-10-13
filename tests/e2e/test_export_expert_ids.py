# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import dataclass, field

import pytest
from vllm import LLM, SamplingParams


@dataclass
class InferenceConfig:
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    max_model_len: int = 256
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.85
    additional_config: dict = field(default_factory=dict)


def _run_inference_and_return_outputs(config: InferenceConfig,
                                      prompts: list[str],
                                      sampling_params: SamplingParams,
                                      enable_export: bool = True):
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
        time.sleep(10)


@pytest.fixture
def test_prompts():
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the future of computing.",
        "Deep learning architectures leverage parallel hardware acceleration.",
        "Mixture of Experts routes tokens dynamically to specialized feedforward layers.",
    ]


@pytest.fixture
def sampling_params():
    return SamplingParams(
        temperature=0.0,
        max_tokens=16,
        ignore_eos=True,
    )


@pytest.mark.parametrize(
    "matrix_case",
    [
        {
            "id": "single_device",
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "ep": False
        },
        {
            "id": "tensor_parallel",
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "ep": False
        },
        {
            "id": "pipeline_parallel",
            "tp": 1,
            "pp": 2,
            "dp": 1,
            "ep": False
        },
        {
            "id": "data_parallel",
            "tp": 1,
            "pp": 1,
            "dp": 2,
            "ep": False
        },
        {
            "id": "expert_parallel",
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "ep": True
        },
    ],
    ids=lambda x: x["id"],
)
def test_export_expert_ids_correctness_matrix(matrix_case, test_prompts,
                                              sampling_params):
    """Verify expert ID export correctness across comprehensive execution matrix configurations."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    config = InferenceConfig(
        tensor_parallel_size=matrix_case["tp"],
        pipeline_parallel_size=matrix_case["pp"],
        data_parallel_size=matrix_case["dp"],
        enable_expert_parallel=matrix_case["ep"],
        additional_config={"enable_continue_decode": True},
    )

    outputs, _ = _run_inference_and_return_outputs(config,
                                                   test_prompts,
                                                   sampling_params,
                                                   enable_export=True)

    assert len(outputs) == len(test_prompts)
    for output in outputs:
        req_output = output.outputs[0]
        assert req_output.routed_experts is not None, "routed_experts tensor must be populated when enabled."
        assert len(
            req_output.routed_experts.shape
        ) == 3, f"Expected 3D expert tensor, got shape {req_output.routed_experts.shape}"

        # Verify alignment length: prompt tokens + generated tokens - 1
        num_prompt_tokens = len(output.prompt_token_ids)
        num_gen_tokens = len(req_output.token_ids)
        expected_tokens_dim = num_prompt_tokens + num_gen_tokens - 1
        actual_tokens_dim = req_output.routed_experts.shape[0]

        assert actual_tokens_dim == expected_tokens_dim, (
            f"Exported expert IDs token dimension mismatch. Expected {expected_tokens_dim} "
            f"(prompt len {num_prompt_tokens} + gen len {num_gen_tokens} - 1), got {actual_tokens_dim}"
        )


def test_export_expert_ids_performance_overhead(test_prompts, sampling_params):
    """Verify that exporting expert IDs adds minimal latency overhead compared to baseline execution."""
    os.environ['SKIP_JAX_PRECOMPILE'] = '0'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '1'

    config = InferenceConfig(
        tensor_parallel_size=2,
        additional_config={"enable_continue_decode": True},
    )

    # Measure baseline execution without exporting expert IDs
    _, latency_baseline = _run_inference_and_return_outputs(
        config, test_prompts, sampling_params, enable_export=False)

    # Measure execution with export enabled
    _, latency_export = _run_inference_and_return_outputs(config,
                                                          test_prompts,
                                                          sampling_params,
                                                          enable_export=True)

    overhead = (latency_export - latency_baseline
                ) / latency_baseline if latency_baseline > 0 else 0
    print("✓ Export expert IDs performance validation:")
    print(f"  Baseline latency: {latency_baseline:.3f}s")
    print(f"  Export latency:   {latency_export:.3f}s")
    print(f"  Overhead ratio:   {overhead:.1%}")

    # Set conservative upper bound threshold for latency overhead to prevent regression
    assert overhead < 0.50, f"Expert ID export performance overhead ({overhead:.1%}) exceeds 50% threshold limit."
