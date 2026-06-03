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

import pytest
from vllm import LLM, SamplingParams

LLM.__repr__ = lambda self: "LLM"


@pytest.fixture(scope="function")
def llm_enabled():
    engine = LLM(
        model="Qwen/Qwen1.5-MoE-A2.7B",
        load_format="dummy",
        trust_remote_code=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_prefix_caching=False,
        kv_cache_dtype="auto",
        enable_expert_parallel=False,
        enable_return_routed_experts=True,
    )
    yield engine
    del engine
    import gc
    gc.collect()
    import time
    time.sleep(10)


@pytest.fixture(scope="function")
def llm_disabled():
    engine = LLM(
        model="Qwen/Qwen1.5-MoE-A2.7B",
        load_format="dummy",
        trust_remote_code=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_prefix_caching=False,
        kv_cache_dtype="auto",
        enable_expert_parallel=False,
        enable_return_routed_experts=False,
    )
    yield engine
    del engine
    import gc
    gc.collect()
    import time
    time.sleep(10)


class TestMoEExpertIds:
    """Verify that MoE routed experts are successfully returned when enabled,
    and not returned when disabled.
    """

    def test_moe_expert_ids_returned_when_enabled(self, llm_enabled: LLM):
        prompt = "The capital of France is"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        outputs = llm_enabled.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Verify that routed_experts is populated and has correct shape
        assert output.routed_experts is not None, (
            "MoE models must populate routed_experts when enabled")
        assert len(output.routed_experts.shape) == 3, (
            f"Expected 3D expert shape, got {output.routed_experts.shape}")

        # Verify that the token dimension has size P + G - 1
        P = len(outputs[0].prompt_token_ids)
        G = len(output.token_ids)
        expected_len = P + G - 1
        actual_len = output.routed_experts.shape[0]
        assert actual_len == expected_len, (
            f"Expected expert 0-th dim to be P + G - 1 ({expected_len}), "
            f"got {actual_len}")

    def test_moe_expert_ids_not_returned_when_disabled(self,
                                                       llm_disabled: LLM):
        prompt = "The capital of France is"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        outputs = llm_disabled.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Verify that routed_experts is None
        assert output.routed_experts is None, (
            "MoE models must not populate routed_experts when disabled")
