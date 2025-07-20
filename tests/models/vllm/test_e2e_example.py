# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

# Test data
PROMPTS = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]

EXPECTED_ANSWERS = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]

N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
SAMPLING_PARAMS = SamplingParams(temperature=0, top_p=1.0, n=N, max_tokens=16)


@pytest.fixture(autouse=True)
def set_tpu_backend_env():
    """Set TPU_BACKEND_TYPE=torchax for all tests in this module."""
    with patch.dict(os.environ, {"TPU_BACKEND_TYPE": "torchax"}):
        yield


@pytest.mark.parametrize("use_spmd", [False, True])
def test_e2e_inference(use_spmd):
    """Test end-to-end inference with and without SPMD mode."""
    from vllm import LLM

    # Base LLM arguments
    llm_args = {
        "model": "Qwen/Qwen2-1.5B-Instruct",
        "max_num_batched_tokens": 64,
        "max_num_seqs": 4,
        "max_model_len": 128,
        "enforce_eager": True,
    }

    # Configure SPMD-specific settings
    if use_spmd:
        with patch.dict(os.environ, {"VLLM_XLA_USE_SPMD": "1"}):
            # Can only hardcode the number of chips for now.
            # calling xr.global_runtime_device_count() before init SPMD env in
            # torch_xla will mess up the distributed env.
            llm_args["tensor_parallel_size"] = 8
            # Use Llama, for num_kv_heads = 8.
            llm_args["model"] = "meta-llama/Llama-3.1-8B-Instruct"

            # Set `enforce_eager=True` to avoid ahead-of-time compilation.
            # In real workloads, `enforce_eager` should be `False`.
            llm = LLM(**llm_args)
            outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    else:
        # Non-SPMD mode
        llm = LLM(**llm_args)
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)

    # Verify outputs
    assert len(outputs) == len(PROMPTS)
    assert len(outputs) == len(EXPECTED_ANSWERS)

    for output, expected_answer in zip(outputs, EXPECTED_ANSWERS):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        # Log the results for debugging
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Expected to start with: {expected_answer!r}")
        print("-" * 50)

        # Verify the generated text starts with expected answer
        assert generated_text.startswith(expected_answer), (
            f"Generated text '{generated_text}' does not start with "
            f"expected answer '{expected_answer}'")
