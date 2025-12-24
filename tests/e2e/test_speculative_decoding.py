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

import os
import random
import string
import time

import pytest
from vllm import LLM, SamplingParams


# TODO (Qiliang Cui): remove this when XLA fixes the recursive jit call issue.
def _is_v7x():
    # jax.devices() will hang so use IS_FOR_V7X to indicate the version.
    return os.environ.get("IS_FOR_V7X", "false") == "true"


def _get_tensor_parallel_size():
    # Work around an XLA issue.
    if _is_v7x():
        return 2
    return 1


def get_ngram_test_prompts():
    num_prompts = 100
    prompts = []

    for _ in range(num_prompts):
        w = random.choice(list(string.ascii_lowercase))
        prompts.append(
            f"Keep repeating: {w} {w} {w} {w} {w} {w} {w} {w} {w} {w}")

    return prompts


def get_eagle3_test_prompts():
    num_prompts = 100
    prompts = []

    for _ in range(num_prompts):
        prompts.append(
            "Predict the continuation of this sequence: 1 2 3 4 5 6 7 8")

    return prompts


def get_test_prompts(speculative_config: dict):
    if speculative_config['method'] == 'ngram':
        return get_ngram_test_prompts()
    elif speculative_config['method'] == 'eagle3':
        return get_eagle3_test_prompts()
    else:
        raise NotImplementedError(
            f"{speculative_config['method']} is not supported yet.")


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0,
                          max_tokens=32,
                          ignore_eos=True,
                          repetition_penalty=1,
                          frequency_penalty=0,
                          presence_penalty=0,
                          min_p=0,
                          logprobs=None)


@pytest.fixture
def model_name():
    return "Qwen/Qwen2.5-0.5B-Instruct"


# TODO(pooyam): run vLLM engine with InProcClient (`VLLM_ENABLE_V1_MULTIPROCESSING = 0`) mode to avoid TPU contention among processes.
def _test_correctness_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
    speculative_config: dict,
):
    '''
    Helper function to test ngram correctness.
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context():
        test_prompts = get_test_prompts(speculative_config)

        ref_llm = LLM(model=model_name,
                      max_model_len=1024,
                      max_num_seqs=4,
                      tensor_parallel_size=_get_tensor_parallel_size())
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)

        del ref_llm

        # Waiting for TPUs to be released.
        time.sleep(10)

        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       max_model_len=1024,
                       max_num_seqs=4,
                       tensor_parallel_size=_get_tensor_parallel_size())
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

        # Waiting for TPUs to be released.
        time.sleep(10)


def test_ngram_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding with greedy sampling.
    '''
    _test_correctness_helper(
        monkeypatch, sampling_config, model_name, {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        })


def test_ngram_correctness_random(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding with random sampling.
    '''
    # Modify sampling config for random sampling
    sampling_config.temperature = 0.01
    sampling_config.top_p = 0.9
    sampling_config.top_k = 5

    _test_correctness_helper(
        monkeypatch, sampling_config, model_name, {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        })


def _test_performance_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    speculative_config: dict,
    min_speedup: float,
):
    '''
    Helper function to test speculative decoding performance.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    '''
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    with monkeypatch.context():
        # Use a smaller set of prompts for performance testing
        test_prompts = get_test_prompts(speculative_config)

        # Test reference LLM timing
        ref_llm = LLM(model=model_name,
                      max_model_len=1024,
                      max_num_seqs=1,
                      enable_prefix_caching=False,
                      tensor_parallel_size=_get_tensor_parallel_size())

        start_time = time.time()
        _ = ref_llm.generate(test_prompts, sampling_config)
        ref_time = time.time() - start_time

        del ref_llm

        # Waiting for TPUs to be released
        time.sleep(10)

        # Test speculative LLM timing with max_num_seqs=1
        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       max_model_len=1024,
                       max_num_seqs=1,
                       tensor_parallel_size=_get_tensor_parallel_size(),
                       enable_prefix_caching=False)

        start_time = time.time()
        _ = spec_llm.generate(test_prompts, sampling_config)
        spec_time = time.time() - start_time

        del spec_llm
        # Waiting for TPUs to be released
        time.sleep(10)

        speedup = ref_time / spec_time
        print(f"Reference LLM time: {ref_time:.2f}s")
        print(f"Speculative LLM time: {spec_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        # TODO(pooyam): Make this tighter once we have better performance.
        assert speedup >= min_speedup, f"Expected at least {min_speedup}x speedup for {speculative_config['method']}, got {speedup:.2f}x"


def test_ngram_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    '''
    Test that speculative decoding provides significant performance improvement.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    Expects spec_llm to be at least 3.x faster than ref_llm.
    '''
    _test_performance_helper(
        monkeypatch, sampling_config, {
            "method": "ngram",
            "prompt_lookup_max": 2,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 4,
        }, 1.2 if _is_v7x() else 3.0)


def test_ngram_performance_random(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    '''
    Test that speculative decoding provides significant performance improvement.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    Expects spec_llm to be at least 3.x faster than ref_llm.
    '''
    sampling_config.temperature = 0.01
    sampling_config.top_p = 0.9
    sampling_config.top_k = 5

    _test_performance_helper(
        monkeypatch, sampling_config, {
            "method": "ngram",
            "prompt_lookup_max": 2,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 4,
        }, 1.5 if _is_v7x() else 3.0)


def test_eagle3_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle-3 speculative decoding.
    '''
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    _test_correctness_helper(
        monkeypatch, sampling_config, model_name, {
            'model': "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 3,
            "method": "eagle3",
            "draft_tensor_parallel_size": 1
        })


def test_eagle3_performance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    '''
    Test that speculative decoding provides significant performance improvement.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    Expects spec_llm to be at least 1.8 faster than ref_llm.
    '''
    _test_performance_helper(
        monkeypatch, sampling_config, {
            "method": "eagle3",
            "model": "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": 1
        }, 1.2 if _is_v7x() else 1.8)
