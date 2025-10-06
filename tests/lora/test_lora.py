# https://github.com/vllm-project/vllm/blob/ed10f3cea199a7a1f3532fbe367f5c5479a6cae9/tests/tpu/lora/test_lora.py
import pytest
import vllm
from vllm.lora.request import LoRARequest

# This file contains tests to ensure that LoRA works correctly on the TPU
# backend. We use a series of custom trained adapters for Qwen2.5-3B-Instruct
# for this. The adapters are:
# Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_x_adapter, where x ranges
# from 1 to 4.

# These adapters are trained using a standard huggingface peft training script,
# where all the inputs are "What is 1+1? \n" and all the outputs are "x". We run
# 100 training iterations with a training batch size of 100.


@pytest.fixture(scope="function", autouse=True)
def use_v1_only(monkeypatch: pytest.MonkeyPatch):
    """
    Since Multi-LoRA is only supported on the v1 TPU backend, set VLLM_USE_V1=1
    for all tests in this file
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        yield


def setup_vllm(num_loras: int) -> vllm.LLM:
    return vllm.LLM(model="Qwen/Qwen2.5-3B-Instruct",
                    max_model_len=256,
                    max_num_seqs=8,
                    enable_lora=True,
                    max_loras=num_loras,
                    max_lora_rank=8)


def test_single_lora():
    """
    This test ensures we can run a single LoRA adapter on the TPU backend.
    We run "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_2_adapter" which
    will force Qwen2.5-3B-Instruct to claim 1+1=2.
    """

    llm = setup_vllm(1)

    prompt = "What is 1+1? \n"

    lora_request = LoRARequest(
        "lora_adapter_2", 2,
        "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_2_adapter")
    output = llm.generate(prompt,
                          sampling_params=vllm.SamplingParams(max_tokens=16,
                                                              temperature=0),
                          lora_request=lora_request)[0].outputs[0].text

    answer = output.strip()[0]

    assert answer.isdigit()
    assert int(answer) == 2


def test_lora_hotswapping():
    """
    This test ensures we can run multiple LoRA adapters on the TPU backend, even
    if we only have space to store 1.

    We run "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_x_adapter" which
    will force Qwen2.5-3B-Instruct to claim 1+1=x, for a range of x.
    """

    lora_name_template = \
        "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    lora_requests = [
        LoRARequest(f"lora_adapter_{i}", i, lora_name_template.format(i))
        for i in range(1, 5)
    ]

    llm = setup_vllm(1)

    prompt = "What is 1+1? \n"

    for i, req in enumerate(lora_requests):
        output = llm.generate(prompt,
                              sampling_params=vllm.SamplingParams(
                                  max_tokens=16, temperature=0),
                              lora_request=req)[0].outputs[0].text
        answer = output.strip()[0]

        assert answer.isdigit()
        assert int(answer) == i + 1, f"Expected {i + 1}, got {answer}"


def test_multi_lora():
    """
    This test ensures we can run multiple LoRA adapters on the TPU backend, when
    we have enough space to store all of them.

    We run "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_x_adapter" which
    will force Qwen2.5-3B-Instruct to claim 1+1=x, for a range of x.
    """
    lora_name_template = \
        "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    lora_requests = [
        LoRARequest(f"lora_adapter_{i}", i, lora_name_template.format(i))
        for i in range(1, 5)
    ]

    llm = setup_vllm(4)

    prompt = "What is 1+1? \n"

    for i, req in enumerate(lora_requests):
        output = llm.generate(prompt,
                              sampling_params=vllm.SamplingParams(
                                  max_tokens=16, temperature=0),
                              lora_request=req)[0].outputs[0].text

        answer = output.strip()[0]

        assert answer.isdigit()
        assert int(
            output.strip()
            [0]) == i + 1, f"Expected {i + 1}, got {int(output.strip()[0])}"
