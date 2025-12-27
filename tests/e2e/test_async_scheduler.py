# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import random
import string
import time

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0,
                          max_tokens=120,
                          ignore_eos=True,
                          repetition_penalty=1,
                          frequency_penalty=0,
                          presence_penalty=0,
                          min_p=0,
                          logprobs=None)


@pytest.fixture
def model_name():
    return "Qwen/Qwen2.5-1.5B-Instruct"


def get_test_prompts():
    """
    Generates a list of prompts with a specific word count,

    Args:
        num_prompts: The number of prompts to generate.
        input_len_words: The total number of words for each prompt.

    Returns:
        A list of strings with number of prompts = num_prompts and
        The total number of words for each prompt = input_len_words.
    """
    num_prompts = 500
    input_len_words = 120
    prompts = []

    # For example w = 's'
    # The generated prompt will be Keep repeating: s s s ...
    num_repetitions = input_len_words
    prefix = "Keep repeating: "

    for _ in range(num_prompts):
        # 1. Pick a random lowercase letter
        w = random.choice(list(string.ascii_lowercase))

        # 2. Create the string of repeated words
        #    This will have (num_repetitions) words
        repeating_part = " ".join([w] * num_repetitions)

        # 3. Combine with the prefix (if any)
        print(f"{prefix}{repeating_part}")
        prompts.append(f"{prefix}{repeating_part}")

    return prompts


def _test_performance_helper(monkeypatch: pytest.MonkeyPatch,
                             sampling_config: SamplingParams, model_name: str,
                             min_speedup: float):
    '''
    Helper function to test async scheduler decoding performance.
    Compares timing between reference LLM and async LLM using Qwen2.5-1.5B.
    '''

    with monkeypatch.context():
        # Use a smaller set of prompts for performance testing
        test_prompts = get_test_prompts()  # num_prompts=100, input_len=120

        # Test reference LLM timing
        ref_llm = LLM(model=model_name,
                      max_model_len=800,
                      max_num_seqs=24,
                      max_num_batched_tokens=512,
                      enable_prefix_caching=False,
                      async_scheduling=0)

        start_time = time.time()
        _ = ref_llm.generate(test_prompts, sampling_config)
        ref_time = time.time() - start_time

        del ref_llm
        # Waiting for TPUs to be released
        time.sleep(10)

        # # Test async LLM timing with max_num_seqs=256
        async_llm = LLM(model=model_name,
                        max_model_len=800,
                        max_num_seqs=24,
                        max_num_batched_tokens=512,
                        enable_prefix_caching=False,
                        async_scheduling=1)

        start_time = time.time()
        _ = async_llm.generate(test_prompts, sampling_config)
        async_time = time.time() - start_time

        del async_llm
        # # Waiting for TPUs to be released
        time.sleep(10)

        speedup = ref_time / async_time
        print(f"Reference LLM time: {ref_time:.2f}s")
        print(f"Async LLM time: {async_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        assert speedup >= min_speedup, f"Expected at least {min_speedup}x speedup for async scheduler, got {speedup:.2f}x"


def test_performance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Test that async scheduler decoding provides significant performance improvement.
    Compares timing between reference LLM and async LLM using Qwen2.5-1.5B.
    Expects async_llm to be at least 1.3x faster than ref_llm.
    '''
    min_speed_up = 1.3
    _test_performance_helper(monkeypatch, sampling_config, model_name,
                             min_speed_up)


def _test_correctness_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Helper function to test async scheduler correctness.
    Compare the outputs of a original LLM and a async LLM
    should be the same when using async scheduler decoding.

    Known Edge Case (KV Cache Swapping):
    Under this case, though the temperature is set to 0,
    the output is still slightly different everytime.
    This is an expected behaviour as the normal scheduler also
    behaves the same and hence, it is difficult to design a test
    for such scenario.
    '''
    with monkeypatch.context():
        test_prompts = get_test_prompts()

        ref_llm = LLM(model=model_name,
                      max_model_len=1024,
                      max_num_seqs=100,
                      async_scheduling=0)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)

        del ref_llm

        # Waiting for TPUs to be released.
        time.sleep(10)

        async_llm = LLM(model=model_name,
                        max_model_len=1024,
                        max_num_seqs=100,
                        async_scheduling=1)
        async_outputs = async_llm.generate(test_prompts, sampling_config)

        matches = 0
        misses = 0
        for ref_output, async_output in zip(ref_outputs, async_outputs):
            if ref_output.outputs[0].text == async_output.outputs[0].text:
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"async_output: {async_output.outputs[0].text}")
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"async_output: {async_output.outputs[0].text}")

        assert misses == 0
        del async_outputs

        # Waiting for TPUs to be released.
        time.sleep(10)


def test_async_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a async LLM
    should be the same when using async scheduler.
    '''

    _test_correctness_helper(monkeypatch, sampling_config, model_name)
