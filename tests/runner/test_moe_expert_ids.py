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

import numpy as np
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
    engine.llm_engine.engine_core.shutdown()
    import gc
    gc.collect()


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
    engine.llm_engine.engine_core.shutdown()
    import gc
    gc.collect()


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

    def test_moe_expert_ids_batch_isolation(self, llm_enabled: LLM):
        """Verifies that the returned expert IDs for batched requests
        match their reference runs when executed in isolation.
        """
        prompt_a = "The capital of France is"
        prompt_b = "Explain quantum computing in simple terms:"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)

        # 1. Run Request A isolated
        outputs_a = llm_enabled.generate([prompt_a], sampling_params)
        routed_experts_a = outputs_a[0].outputs[0].routed_experts
        assert routed_experts_a is not None

        # 2. Run Request B isolated
        outputs_b = llm_enabled.generate([prompt_b], sampling_params)
        routed_experts_b = outputs_b[0].outputs[0].routed_experts
        assert routed_experts_b is not None

        # 3. Run Request A & B together in a batch
        outputs_batched = llm_enabled.generate([prompt_a, prompt_b],
                                               sampling_params)

        # Verify batched outputs match their isolated runs
        routed_experts_batched_a = outputs_batched[0].outputs[0].routed_experts
        routed_experts_batched_b = outputs_batched[1].outputs[0].routed_experts

        assert routed_experts_batched_a is not None
        assert routed_experts_batched_b is not None

        # Verify shapes are matching
        assert routed_experts_batched_a.shape == routed_experts_a.shape
        assert routed_experts_batched_b.shape == routed_experts_b.shape

        # Verify values are identical (batch isolation test)
        np.testing.assert_array_equal(routed_experts_batched_a,
                                      routed_experts_a)
        np.testing.assert_array_equal(routed_experts_batched_b,
                                      routed_experts_b)
