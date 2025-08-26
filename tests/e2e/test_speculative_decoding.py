from __future__ import annotations

import random
import string

import pytest
from vllm import LLM, SamplingParams


def get_test_prompts():
    num_prompts = 100
    prompts = []

    for _ in range(num_prompts):
        w = random.choice(list(string.ascii_lowercase))
        prompts.append(
            f"Keep repeating: {w} {w} {w} {w} {w} {w} {w} {w} {w} {w}")

    return prompts


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0,
                          max_tokens=10,
                          ignore_eos=False,
                          repetition_penalty=1,
                          frequency_penalty=0,
                          presence_penalty=0,
                          min_p=0,
                          logprobs=None)


@pytest.fixture
def model_name():
    return "Qwen/Qwen2.5-0.5B-Instruct"


def test_ngram_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context():
        test_prompts = get_test_prompts()

        ref_llm = LLM(model=model_name, max_model_len=1024)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)

        del ref_llm
        import time

        # Waiting for TPUs to be released.
        time.sleep(10)

        spec_llm = LLM(model=model_name,
                       speculative_config={
                           "method": "ngram",
                           "prompt_lookup_max": 5,
                           "prompt_lookup_min": 3,
                           "num_speculative_tokens": 3,
                       },
                       max_model_len=1024,
                       max_num_seqs=4)
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        assert misses == 0
        del spec_llm
