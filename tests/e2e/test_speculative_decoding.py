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
from vllm.v1.metrics.reader import Counter


# TODO (Qiliang Cui): remove this when XLA fixes the recursive jit call issue.
def _is_v7x():
    # jax.devices() will hang so use TPU_VERSION to indicate the version.
    return os.environ.get("TPU_VERSION", "tpu6e") == "tpu7x"


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
    elif speculative_config['method'] in ('eagle3', 'qwen3_next_mtp', 'mtp'):
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
def _get_baseline_results(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
    test_prompts: list,
    max_num_seqs: int = 4,
    async_scheduling: bool = False,
    enable_dp_attention: bool = False,
    extra_kwargs: dict | None = None,
):
    '''
    Generate reference outputs from a non-speculative LLM, to be compared
    against the speculative LLM outputs in _test_correctness_helper. The caller
    must pass the same test_prompts to both so the comparison lines up.
    '''
    with monkeypatch.context():
        kwargs = {
            "max_model_len": 1024,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "model_loader_extra_config": {
                "enable_weights_track": False
            },
            "async_scheduling": async_scheduling,
        }
        if enable_dp_attention:
            os.environ["NEW_MODEL_DESIGN"] = "1"
            kwargs["additional_config"] = {
                "sharding": {
                    "sharding_strategy": {
                        "enable_dp_attention": True,
                        "attn_dp_size": _get_tensor_parallel_size()
                    }
                }
            }
        else:
            os.environ["NEW_MODEL_DESIGN"] = "0"
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)

        del ref_llm

        # Waiting for TPUs to be released.
        time.sleep(10)
        return ref_outputs


# TODO(pooyam): run vLLM engine with InProcClient (`VLLM_ENABLE_V1_MULTIPROCESSING = 0`) mode to avoid TPU contention among processes.
def _test_correctness_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
    speculative_config: dict,
    test_prompts: list,
    ref_outputs: list,
    max_num_seqs: int = 4,
    async_scheduling: bool = False,
    enable_dp_attention: bool = False,
    extra_kwargs: dict | None = None,
):
    '''
    Helper function to test ngram correctness.
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context():
        kwargs = {
            "max_model_len": 1024,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "model_loader_extra_config": {
                "enable_weights_track": False
            },
            "async_scheduling": async_scheduling,
        }
        if enable_dp_attention:
            os.environ["NEW_MODEL_DESIGN"] = "1"
            kwargs["additional_config"] = {
                "sharding": {
                    "sharding_strategy": {
                        "enable_dp_attention": True,
                        "attn_dp_size": _get_tensor_parallel_size()
                    }
                }
            }
        else:
            os.environ["NEW_MODEL_DESIGN"] = "0"
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        if sampling_config.logprobs is not None:
            for spec_output in spec_outputs:
                completion = spec_output.outputs[0]
                assert completion.logprobs is not None, "Logprobs should not be None"
                assert len(completion.logprobs) == len(completion.token_ids), (
                    f"Length mismatch: len(logprobs)={len(completion.logprobs)} vs "
                    f"len(token_ids)={len(completion.token_ids)}")

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
        print(
            f"All {matches} outputs match between reference LLM and speculative LLM."
        )
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
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3,
    }
    test_prompts = get_test_prompts(speculative_config)

    ref_outputs = _get_baseline_results(monkeypatch, sampling_config,
                                        model_name, test_prompts)

    _test_correctness_helper(monkeypatch,
                             sampling_config,
                             model_name,
                             speculative_config,
                             test_prompts,
                             ref_outputs=ref_outputs)


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

    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3,
    }
    test_prompts = get_test_prompts(speculative_config)

    ref_outputs = _get_baseline_results(monkeypatch, sampling_config,
                                        model_name, test_prompts)

    _test_correctness_helper(monkeypatch,
                             sampling_config,
                             model_name,
                             speculative_config,
                             test_prompts,
                             ref_outputs=ref_outputs)


def _test_performance_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    speculative_config: dict,
    min_acceptance_rate: float,
    max_num_seqs: int = 1,
    async_scheduling: bool = False,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    enable_dp_attention: bool = False,
    extra_kwargs: dict | None = None,
):
    '''
    Helper function to test speculative decoding performance.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    '''
    with monkeypatch.context():
        # Use a smaller set of prompts for performance testing
        test_prompts = get_test_prompts(speculative_config)

        kwargs = {
            "max_model_len": 1024,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "enable_prefix_caching": False,
            "model_loader_extra_config": {
                "enable_weights_track": False
            },
            "disable_log_stats": False,
            "async_scheduling": async_scheduling,
        }
        if enable_dp_attention:
            os.environ["NEW_MODEL_DESIGN"] = "1"
            kwargs["additional_config"] = {
                "sharding": {
                    "sharding_strategy": {
                        "enable_dp_attention": True,
                        "attn_dp_size": _get_tensor_parallel_size()
                    }
                }
            }
        else:
            os.environ["NEW_MODEL_DESIGN"] = "0"
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)

        spec_llm.generate(test_prompts, sampling_config)

        metrics = spec_llm.get_metrics()
        num_draft_tokens = num_accepted_tokens = 0
        acceptance_rate = 0.0
        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        if num_draft_tokens > 0:
            acceptance_rate = num_accepted_tokens / num_draft_tokens
            print(f"Acceptance rate: {acceptance_rate:.2%}")
            print("num_accepted_tokens:" + str(num_accepted_tokens))
            print("num_draft_tokens:" + str(num_draft_tokens))

        del spec_llm
        # Waiting for TPUs to be released
        time.sleep(30)

        assert acceptance_rate >= min_acceptance_rate, f"Expected at least {min_acceptance_rate:.2%} acceptance rate for {speculative_config['method']}, got {acceptance_rate:.2%}"


def test_ngram_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    '''
    Test that speculative decoding provides significant performance improvement.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    '''
    _test_performance_helper(monkeypatch,
                             sampling_config, {
                                 "method": "ngram",
                                 "prompt_lookup_max": 2,
                                 "prompt_lookup_min": 2,
                                 "num_speculative_tokens": 4,
                             },
                             min_acceptance_rate=0.85)


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

    _test_performance_helper(monkeypatch,
                             sampling_config, {
                                 "method": "ngram",
                                 "prompt_lookup_max": 2,
                                 "prompt_lookup_min": 2,
                                 "num_speculative_tokens": 4,
                             },
                             min_acceptance_rate=0.85)


@pytest.fixture(scope="module")
def eagle3_baseline():
    '''
    Compute the eagle3 reference prompts and baseline outputs once and share
    them across all parametrized test_eagle3_correctness cases. The baseline is
    non-speculative and greedy, so its outputs are deterministic regardless of
    async_scheduling or enable_dp_attention.

    Note: mirrors the default `sampling_config` fixture; that fixture is
    function-scoped (and mutated by other tests) so it can't be injected here.
    '''
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    sampling_config = SamplingParams(temperature=0,
                                     max_tokens=32,
                                     ignore_eos=True,
                                     repetition_penalty=1,
                                     frequency_penalty=0,
                                     presence_penalty=0,
                                     min_p=0,
                                     logprobs=None)
    test_prompts = get_eagle3_test_prompts()
    with pytest.MonkeyPatch.context() as mp:
        ref_outputs = _get_baseline_results(mp,
                                            sampling_config,
                                            model_name,
                                            test_prompts,
                                            max_num_seqs=10)
    return test_prompts, ref_outputs


@pytest.mark.parametrize(
    "async_scheduling, enable_dp_attention",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_eagle3_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    async_scheduling: bool,
    enable_dp_attention: bool,
    eagle3_baseline: tuple,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle-3 speculative decoding.
    '''
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    speculative_config = {
        'model': "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "method": "eagle3",
        "draft_tensor_parallel_size": 1
    }
    test_prompts, ref_outputs = eagle3_baseline

    _test_correctness_helper(monkeypatch,
                             sampling_config,
                             model_name,
                             speculative_config,
                             test_prompts,
                             ref_outputs=ref_outputs,
                             max_num_seqs=10,
                             async_scheduling=async_scheduling,
                             enable_dp_attention=enable_dp_attention)


@pytest.mark.parametrize(
    "max_num_seqs,async_scheduling, enable_dp_attention",
    [(1, False, False), (20, True, False), (20, True, True)],
)
def test_eagle3_performance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
    enable_dp_attention: bool,
):
    '''
    Test that speculative decoding provides significant performance improvement.
    Compares timing between reference LLM and speculative LLM using Llama 3 8B.
    Expects spec_llm to be at least 1.8 faster than ref_llm.
    '''
    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    _test_performance_helper(
        monkeypatch,
        sampling_config, {
            "method": "eagle3",
            "model": "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": 1
        },
        min_acceptance_rate=0.75,
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
        enable_dp_attention=enable_dp_attention,
        model_name='meta-llama/Llama-3.1-8B-Instruct')


@pytest.fixture(scope="module")
def mtp_baseline():
    '''
    Compute the mtp reference prompts, baseline outputs and the LLM extra_kwargs
    once and share them across all parametrized test_mtp_correctness cases. The
    baseline is non-speculative and greedy, so its outputs are deterministic
    regardless of async_scheduling. extra_kwargs is returned so the spec LLM in
    the test reuses the exact same config.

    Note: mirrors the default `sampling_config` fixture; that fixture is
    function-scoped (and mutated by other tests) so it can't be injected here.
    '''
    model_name = "Qwen/Qwen3.5-4B"
    sampling_config = SamplingParams(temperature=0,
                                     max_tokens=32,
                                     ignore_eos=True,
                                     repetition_penalty=1,
                                     frequency_penalty=0,
                                     presence_penalty=0,
                                     min_p=0,
                                     logprobs=None)
    extra_kwargs = {
        "seed": 42,
        "max_model_len": 128,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.90,
    }
    test_prompts = get_eagle3_test_prompts()
    with pytest.MonkeyPatch.context() as mp:
        ref_outputs = _get_baseline_results(
            mp,
            sampling_config,
            model_name,
            test_prompts,
            max_num_seqs=10,
            extra_kwargs=extra_kwargs,
        )
    return test_prompts, ref_outputs, extra_kwargs


@pytest.mark.parametrize(
    "async_scheduling, enable_dp_attention",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    async_scheduling: bool,
    enable_dp_attention: bool,
    mtp_baseline: tuple,
):
    '''
    Compare the outputs of an original LLM and a speculative LLM;
    they should be the same when using MTP speculative decoding.
    '''
    model_name = "Qwen/Qwen3.5-4B"
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "vllm")

    speculative_config = {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    test_prompts, ref_outputs, extra_kwargs = mtp_baseline

    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        model_name,
        speculative_config=speculative_config,
        test_prompts=test_prompts,
        ref_outputs=ref_outputs,
        max_num_seqs=10,
        async_scheduling=async_scheduling,
        enable_dp_attention=enable_dp_attention,
        extra_kwargs=extra_kwargs,
    )


@pytest.mark.parametrize(
    "max_num_seqs,async_scheduling, enable_dp_attention",
    [(1, False, False), (20, True, False), (20, True, True)],
)
def test_mtp_performance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
    enable_dp_attention: bool,
):
    '''
    Test that MTP speculative decoding achieves the expected acceptance rate.
    '''
    model_name = "Qwen/Qwen3.5-4B"
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "vllm")

    extra_kwargs = {
        "seed": 42,
        "max_model_len": 128,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.90,
    }

    _test_performance_helper(
        monkeypatch,
        sampling_config,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
        min_acceptance_rate=0.99,
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
        enable_dp_attention=enable_dp_attention,
        model_name=model_name,
        extra_kwargs=extra_kwargs,
    )


def test_eagle3_logprobs_correctness(monkeypatch: pytest.MonkeyPatch, ):
    """Reproduction test for speculative decoding with logprobs enabled."""
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    speculative_config = {
        'model': "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "method": "eagle3",
        "draft_tensor_parallel_size": 1
    }

    sampling_config = SamplingParams(temperature=0,
                                     max_tokens=8,
                                     ignore_eos=True,
                                     repetition_penalty=1,
                                     frequency_penalty=0,
                                     presence_penalty=0,
                                     min_p=0,
                                     logprobs=1)
    test_prompts = [
        "Predict the continuation of this sequence: 1 2 3 4 5 6 7 8"
    ]

    # Get baseline outputs with logprobs enabled
    ref_outputs = _get_baseline_results(monkeypatch,
                                        sampling_config,
                                        model_name,
                                        test_prompts,
                                        max_num_seqs=2)

    # Get speculative decoding outputs with logprobs enabled
    _test_correctness_helper(monkeypatch,
                             sampling_config,
                             model_name,
                             speculative_config,
                             test_prompts,
                             ref_outputs=ref_outputs,
                             max_num_seqs=2,
                             async_scheduling=True,
                             enable_dp_attention=False)


def test_eagle3_logprobs_correctness_random(monkeypatch: pytest.MonkeyPatch, ):
    """Reproduction test for speculative decoding with logprobs enabled and temperature > 0."""
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    speculative_config = {
        'model': "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "method": "eagle3",
        "draft_tensor_parallel_size": 1
    }

    sampling_config = SamplingParams(temperature=0.8,
                                     top_p=0.9,
                                     max_tokens=8,
                                     ignore_eos=True,
                                     repetition_penalty=1,
                                     frequency_penalty=0,
                                     presence_penalty=0,
                                     min_p=0,
                                     logprobs=1)
    test_prompts = [
        "Predict the continuation of this sequence: 1 2 3 4 5 6 7 8"
    ]

    # Get baseline outputs with logprobs enabled
    ref_outputs = _get_baseline_results(monkeypatch,
                                        sampling_config,
                                        model_name,
                                        test_prompts,
                                        max_num_seqs=2)

    # Get speculative decoding outputs with logprobs enabled
    _test_correctness_helper(monkeypatch,
                             sampling_config,
                             model_name,
                             speculative_config,
                             test_prompts,
                             ref_outputs=ref_outputs,
                             max_num_seqs=2,
                             async_scheduling=True,
                             enable_dp_attention=False)


@pytest.fixture(scope="module")
def gemma4_mtp_baseline():
    '''
    Compute the gemma4 mtp reference prompts, baseline outputs and the LLM extra_kwargs
    once and share them across all parameterized test_gemma4_mtp_correctness cases.
    '''
    model_name = "google/gemma-4-E2B-it"
    sampling_config = SamplingParams(temperature=0,
                                     max_tokens=32,
                                     ignore_eos=True,
                                     repetition_penalty=1,
                                     frequency_penalty=0,
                                     presence_penalty=0,
                                     min_p=0,
                                     logprobs=None)

    extra_kwargs = {
        "seed": 42,
        "max_model_len": 128,
        "max_num_batched_tokens": 1024,
        "block_size": 256,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.8,
    }
    test_prompts = get_eagle3_test_prompts()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_IMPL_TYPE", "flax_nnx")
        ref_outputs = _get_baseline_results(
            mp,
            sampling_config,
            model_name,
            test_prompts,
            max_num_seqs=10,
            extra_kwargs=extra_kwargs,
        )
    return test_prompts, ref_outputs, extra_kwargs


@pytest.mark.parametrize(
    "async_scheduling, enable_dp_attention",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_gemma4_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    async_scheduling: bool,
    enable_dp_attention: bool,
    gemma4_mtp_baseline: tuple,
):
    '''
    Compare the outputs of an original LLM and a speculative LLM;
    they should be the same when using Gemma-4 MTP speculative decoding.
    '''
    model_name = "google/gemma-4-E2B-it"
    monkeypatch.setenv("MODEL_IMPL_TYPE", "flax_nnx")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "flax_nnx")

    speculative_config = {
        "model": "google/gemma-4-E2B-it-assistant",
        "num_speculative_tokens": 4,
    }
    test_prompts, ref_outputs, extra_kwargs = gemma4_mtp_baseline

    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        model_name,
        speculative_config=speculative_config,
        test_prompts=test_prompts,
        ref_outputs=ref_outputs,
        max_num_seqs=10,
        async_scheduling=async_scheduling,
        enable_dp_attention=enable_dp_attention,
        extra_kwargs=extra_kwargs,
    )


@pytest.mark.parametrize(
    "max_num_seqs,async_scheduling, enable_dp_attention",
    [(1, False, False), (20, True, False), (20, True, True)],
)
def test_gemma4_mtp_performance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
    enable_dp_attention: bool,
):
    '''
    Test that Gemma-4 MTP speculative decoding achieves the expected acceptance rate.
    '''
    model_name = "google/gemma-4-E2B-it"
    monkeypatch.setenv("MODEL_IMPL_TYPE", "flax_nnx")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "flax_nnx")

    extra_kwargs = {
        "seed": 42,
        "max_model_len": 128,
        "max_num_batched_tokens": 1024,
        "block_size": 256,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.8,
    }

    _test_performance_helper(
        monkeypatch,
        sampling_config,
        speculative_config={
            "model": "google/gemma-4-E2B-it-assistant",
            "num_speculative_tokens": 4,
            "method": "mtp",
        },
        min_acceptance_rate=0.80,
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
        enable_dp_attention=enable_dp_attention,
        model_name=model_name,
        extra_kwargs=extra_kwargs,
    )
