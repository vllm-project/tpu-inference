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
        # Routed-experts output requires prefix caching (vLLM #45635).
        enable_prefix_caching=True,
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


@pytest.fixture(scope="function")
def llm_enabled_sync():
    # async_scheduling=False routes generation through the sync sampler path
    # (_sample_from_logits), which reconstructs routed experts differently than
    # the async get_output path. continue_decode requires async_scheduling=False,
    # so this also covers that use case. The default llm_enabled fixture runs on
    # the async path, so this fixture is the only coverage of the sync path.
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
        # Routed-experts output requires prefix caching (vLLM #45635).
        enable_prefix_caching=True,
        kv_cache_dtype="auto",
        enable_expert_parallel=False,
        enable_return_routed_experts=True,
        async_scheduling=False,
        additional_config={
            "enable_continue_decode": True,
            "max_decode_steps": 4
        },
    )
    yield engine
    engine.llm_engine.engine_core.shutdown()
    import gc
    gc.collect()


def _shrink_moe_eagle3(hf_config):
    hf_config.hidden_size = 512
    hf_config.intermediate_size = 512
    hf_config.num_attention_heads = 4
    hf_config.num_key_value_heads = 2
    hf_config.head_dim = 128
    hf_config.vocab_size = 151936
    if getattr(hf_config, "model_type", None) == "llama":
        hf_config.num_hidden_layers = 1
        hf_config.target_hidden_size = 512
        hf_config.draft_vocab_size = 151936
        hf_config.eagle_aux_hidden_state_layer_ids = [1, 2, 3]
    else:
        hf_config.num_hidden_layers = 4
        hf_config.moe_intermediate_size = 256
        hf_config.num_experts = 8
        hf_config.num_experts_per_tok = 2
    return hf_config


@pytest.fixture(scope="function")
def llm_enabled_async_spec_decode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "vllm")
    engine = LLM(
        model="Qwen/Qwen3-30B-A3B",
        load_format="dummy",
        hf_overrides=_shrink_moe_eagle3,
        trust_remote_code=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_prefix_caching=True,
        kv_cache_dtype="auto",
        enable_expert_parallel=False,
        enable_return_routed_experts=True,
        async_scheduling=True,
        speculative_config={
            "method": "eagle3",
            "model": "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
        },
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

    def test_moe_expert_ids_sync_path_prompt_populated(self,
                                                       llm_enabled_sync: LLM):
        """Covers the sync sampler path (async_scheduling=False, used by
        continue_decode). The slot start for routed experts must be the
        request's pre-step computed-token count; deriving it as
        num_computed_tokens - num_tokens underflowed to a negative value on this
        path and crashed the engine on the first prefill. The prompt's routed
        experts must be populated (not zero-filled)."""
        prompt = "The capital of France is"
        sampling_params = SamplingParams(temperature=0, max_tokens=8)
        output = llm_enabled_sync.generate([prompt], sampling_params)[0]

        routed_experts = output.outputs[0].routed_experts
        assert routed_experts is not None, (
            "routed_experts must be populated on the sync path")
        re_arr = np.asarray(routed_experts)
        num_prompt_tokens = len(output.prompt_token_ids)
        # The prompt slice must not be all-zero (the slot bug zero-filled it).
        prompt_slice = re_arr[:num_prompt_tokens]
        assert not (prompt_slice == 0).all(), (
            "prompt routed experts are all zero on the sync path -- slot "
            "start_pos is misaligned with the scheduler-side read")

    def test_moe_expert_ids_async_spec_decode_with_rejections(
        self,
        llm_enabled_async_spec_decode: LLM,
    ):
        """Verifies routed_experts slot tracking under async_scheduling=True +
        speculative decoding when draft tokens are proposed and rejected across
        consecutive decode steps.

        Under async scheduling, step N+1 is scheduled before step N's rejected
        draft tokens are subtracted from scheduler_output's num_computed_tokens.
        Without tracking state.emit_cursor / in_decode in TPUAuxOutputWorker,
        reconstruct_slots writes decode expert rows to the un-reconciled offset
        and inflates routed_experts.shape[0] beyond P + G - 1.
        """
        prompt = (
            "Keep repeating the sequence: alpha beta gamma alpha beta gamma "
            "alpha beta gamma alpha beta gamma")
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
        )
        outputs = llm_enabled_async_spec_decode.generate([prompt],
                                                         sampling_params)
        req_out = outputs[0]
        comp_out = req_out.outputs[0]

        assert comp_out.routed_experts is not None, (
            "routed_experts must be populated under async speculative decoding")
        re_arr = np.asarray(comp_out.routed_experts)
        assert re_arr.ndim == 3, (
            f"Expected 3D routed_experts tensor, got {re_arr.shape}")

        P = len(req_out.prompt_token_ids)
        G = len(comp_out.token_ids)
        expected_len = P + G - 1
        assert re_arr.shape[0] == expected_len, (
            f"Expected routed_experts token dim P + G - 1 ({expected_len}), "
            f"got {re_arr.shape[0]} (un-reconciled spec-decode token_start "
            f"shifts emit_cursor on draft rejections)")
        assert np.all(re_arr >= 0), (
            "routed_experts contains negative/uninitialized slot values")
        assert not (re_arr[P:] == 0).all(), (
            "decode routed_experts slice is all zero after spec-decode steps")

        # Second turn sharing the prompt prefix (hits prefix cache + spec decode)
        follow_up = prompt + " delta epsilon"
        outputs_2 = llm_enabled_async_spec_decode.generate([follow_up],
                                                           sampling_params)
        req_out_2 = outputs_2[0]
        comp_out_2 = req_out_2.outputs[0]
        assert comp_out_2.routed_experts is not None
        re_arr_2 = np.asarray(comp_out_2.routed_experts)
        P2 = len(req_out_2.prompt_token_ids)
        G2 = len(comp_out_2.token_ids)
        assert re_arr_2.shape[0] == P2 + G2 - 1
        np.testing.assert_array_equal(re_arr_2[:P], re_arr[:P])


