# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture(autouse=True)
def setup_new_model_design():
    os.environ['NEW_MODEL_DESIGN'] = '1'


@pytest.mark.skip(reason="Requires vLLM with PR 33195/33438 merged")
def test_sleep_level_0_enqueue_wait():
    """Test that sleep(0) pauses scheduling, enqueue() adds requests,
    and wake_up() + wait_for_completion() processes them correctly.

    This validates the core flow used by `vllm bench iterations`:
      1. sleep(level=0) to pause the scheduler
      2. enqueue() to add requests while paused
      3. wake_up(tags=["scheduling"]) to resume scheduling
      4. wait_for_completion() to block until all requests finish
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        ignore_eos=True,
    )

    llm = LLM(
        model=model_name,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=128,
    )

    prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water boils at",
    ]

    # Step 1: Pause the scheduler with sleep(level=0)
    llm.sleep(level=0)

    # Step 2: Enqueue requests while scheduler is paused
    request_ids = llm.enqueue(prompts, sampling_params)

    # Step 3: Resume scheduling
    llm.wake_up(tags=["scheduling"])

    # Step 4: Wait for all requests to complete
    outputs = llm.wait_for_completion()

    # Verify results
    assert len(outputs) == len(prompts), (
        f"Expected {len(prompts)} outputs, got {len(outputs)}"
    )

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        assert len(generated_text) > 0, (
            f"Prompt {i} produced empty output"
        )
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {generated_text}")
        print()

    del llm
