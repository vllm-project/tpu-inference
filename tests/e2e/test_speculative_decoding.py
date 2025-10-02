from __future__ import annotations

import random
import string
import time

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
def _test_ngram_correctness_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Helper function to test ngram correctness.
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context():
        test_prompts = get_test_prompts()

        ref_llm = LLM(model=model_name, max_model_len=1024, max_num_seqs=4)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)

        del ref_llm

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
    _test_ngram_correctness_helper(monkeypatch, sampling_config, model_name)


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

    _test_ngram_correctness_helper(monkeypatch, sampling_config, model_name)


def _test_ngram_performance_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    min_speedup: float,
):
    '''
    Helper function to test ngram performance.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    '''
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    with monkeypatch.context():
        # Use a smaller set of prompts for performance testing
        test_prompts = get_test_prompts()

        # Test reference LLM timing
        ref_llm = LLM(model=model_name,
                      max_model_len=1024,
                      max_num_seqs=1,
                      enable_prefix_caching=False)

        start_time = time.time()
        _ = ref_llm.generate(test_prompts, sampling_config)
        ref_time = time.time() - start_time

        del ref_llm

        # Waiting for TPUs to be released
        time.sleep(10)

        # Test speculative LLM timing with max_num_seqs=1
        spec_llm = LLM(model=model_name,
                       speculative_config={
                           "method": "ngram",
                           "prompt_lookup_max": 2,
                           "prompt_lookup_min": 2,
                           "num_speculative_tokens": 4,
                       },
                       max_model_len=1024,
                       max_num_seqs=1,
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
        assert speedup >= min_speedup, f"Expected at least {min_speedup}x speedup, got {speedup:.2f}x"


def test_ngram_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    '''
    Test that speculative decoding provides significant performance improvement.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    Expects spec_llm to be at least 3.x faster than ref_llm.
    '''
    _test_ngram_performance_helper(monkeypatch, sampling_config, 3.0)


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

    _test_ngram_performance_helper(monkeypatch, sampling_config, 3.0)


# TODO(pooyam): Make this rigorous once EAGLE-3 is working correctly.
# For now, it's just an e2e test to make sure code runs without error.
def test_eagle3_performance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    with monkeypatch.context():
        monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "1")
        monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")

        # Use a smaller set of prompts for performance testing
        test_prompts = get_test_prompts()[:30]

        # Test speculative LLM timing with max_num_seqs=1
        spec_llm = LLM(model=model_name,
                       speculative_config={
                           "method": "eagle3",
                           "model": "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
                           "num_speculative_tokens": 3,
                           "draft_tensor_parallel_size": 1
                       },
                       max_model_len=1024,
                       max_num_seqs=2,
                       enable_prefix_caching=False)

        spec_llm.generate(test_prompts, sampling_config)
