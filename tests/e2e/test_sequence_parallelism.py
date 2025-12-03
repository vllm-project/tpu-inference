# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import logging
import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.config import CompilationConfig


@pytest.fixture(autouse=True)
def setup_new_model_design():
    os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'


@pytest.fixture
def test_prompts():
    """Simple test prompts for data parallelism testing."""
    return [
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


@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
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
                               additional_config: dict = {},
                               kv_cache_dtype: str = "auto",
                               enable_prefix_caching: bool = False,
                               async_scheduling: bool = False) -> list:
    """Helper function to run inference with specified configuration."""

    # Create LLM args using parser-based approach similar to offline_inference.py
    compilation_config = CompilationConfig(pass_config={
        "enable_sequence_parallelism": True,
    }, )
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=128,
        compilation_config=compilation_config,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.98,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        enable_prefix_caching=enable_prefix_caching,
        additional_config=additional_config,
        kv_cache_dtype=kv_cache_dtype,
        async_scheduling=async_scheduling,
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


def test_model_sequence_parallelism(
    caplog: pytest.LogCaptureFixture,
    test_prompts: list,
    sampling_params: SamplingParams,
):
    logger_name = "vllm.tpu_inference.layers.vllm.quantization.common"
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    original_propagate = logger.propagate

    # Create an in-memory stream to capture log output
    log_capture_string = io.StringIO()
    # Create a handler that writes to our in-memory stream
    capture_handler = logging.StreamHandler(log_capture_string)

    test_model = "Qwen/Qwen2.5-32B"

    try:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(capture_handler)

        outputs = _run_inference_with_config(
            model_name=test_model,
            test_prompts=test_prompts,
            sampling_params=sampling_params,
            tensor_parallel_size=8,
            async_scheduling=True,
        )

        # Verify we got outputs for all prompts
        assert len(outputs) == len(test_prompts)

        # Verify each output has generated text
        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text.strip()) > 0
            print(f"Output: {output.outputs[0].text.strip()}")

        log_contents = log_capture_string.getvalue()
        print(f'xw32 {log_contents=}')
        print(
            f"✓ Model sequence parallelism test passed with {len(outputs)} outputs"
        )
    finally:
        # --- IMPORTANT: Clean up and restore the logger's original state ---
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        logger.removeHandler(capture_handler)
