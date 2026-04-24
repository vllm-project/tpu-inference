# Copyright 2026 Google LLC
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

import itertools

import pytest
from vllm import SamplingParams
from vllm.config import KVTransferConfig

from tests.offload.tpu_offload_accuracy_test import (
    _test_kv_cache_cpu_offloading_accuracy, read_prompt_from_file)


@pytest.fixture
def sampling_config():
    """deterministic sampling config"""
    return SamplingParams(temperature=0.0,
                          max_tokens=1,
                          seed=42,
                          ignore_eos=True)


@pytest.fixture
def kv_transfer_config():
    """use TPUOffloadConnector"""
    return KVTransferConfig(
        kv_connector="TPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_module_path="tpu_inference.offload.tpu_offload_connector",
    )


def test_kv_cache_cpu_offloading_performance_larger_than_cpu_ram(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
):
    decode_saves = ["0"]
    skip_precompile = ["0"]
    batched_saves = ["0"]
    prompts = [read_prompt_from_file("large_prompt.txt")]

    # Define the prefill latency improvement (in seconds)
    # NOTE(jcgu): This value is specific to model, prompt, TPU (v7 etc.,
    # please update it when running on a different setup.
    latency_diff = 0.150

    for decode_save, _skip_precompile, batched_save in itertools.product(
            decode_saves, skip_precompile, batched_saves):
        pass1_time, pass2_time = _test_kv_cache_cpu_offloading_accuracy(
            monkeypatch,
            sampling_config,
            kv_transfer_config,
            _skip_precompile,
            decode_save,
            batched_save,
            "10",  # TPU_OFFLOAD_NUM_CPU_CHUNKS
            prompts,
        )

        print("\nPerformance Results:")
        print(f"  Pass 1 Time: {pass1_time * 1000:.2f} ms")
        print(f"  Pass 2 Time: {pass2_time * 1000:.2f} ms")
        print(f"  Ratio (Pass2/Pass1): {pass2_time / pass1_time:.4f}")

        assert pass2_time < pass1_time - latency_diff, (
            f"Second pass was not fast enough! "
            f"Pass 1: {pass1_time:.4f}s, Pass 2: {pass2_time:.4f}s ")
