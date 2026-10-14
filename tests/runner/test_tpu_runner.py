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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.config.multimodal import BaseDummyOptions

from tpu_inference.models.common.interface import (ModelInterface,
                                                   MultiModalInterface)
from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestTPUJaxRunner:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=MagicMock()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            speculative_config = SpeculativeConfig(
                model='ngram',
                num_speculative_tokens=5,
                prompt_lookup_max=4,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=speculative_config,
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)

    def test_get_supported_tasks_runner(self):
        """Test get_supported_tasks for generate runner type."""
        supported_tasks = self.runner.get_supported_tasks()
        assert supported_tasks == ("generate", )

    def test_get_input_ids_embeds(self):
        """Tests _get_input_ids_embeds for both multimodal and text-only models."""
        # 1. ===== Setup =====
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = [jnp.ones((10, 128))]
        dummy_is_mm_embed = jnp.array([False, True, True], dtype=jnp.bool_)
        dummy_final_embeds = jnp.ones((3, 128))

        # Mock the embedding function
        self.mock_get_input_embed_fn = MagicMock()
        self.runner.embed_input_ids_fn = self.mock_get_input_embed_fn
        self.mock_get_input_embed_fn.return_value = dummy_final_embeds
        self.runner.state_leaves = MagicMock()

        # 2. ===== Act & Assert (Multimodal) =====
        self.runner.is_multimodal_model = True

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds, dummy_is_mm_embed)

        assert input_ids_res is None
        np.testing.assert_array_equal(np.asarray(inputs_embeds_res),
                                      np.asarray(dummy_final_embeds))
        self.mock_get_input_embed_fn.assert_called_once_with(
            self.runner.state_leaves,
            dummy_input_ids,
            dummy_mm_embeds,
            is_multimodal=dummy_is_mm_embed)

        # 3. ===== Act & Assert (Multimodal w/o mm embeds) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = True

        # Without mm_embeds in the current scheduled tokens
        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, None, None)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

        # 4. ===== Act & Assert (Text-only) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = False

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds, dummy_is_mm_embed)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_hybrid_kvcache(self, mock_sampling_metadata):
        # create hybrid kv cache config
        # 20 layers, 10 full attn + 10 sw attn
        self._create_mock_hybrid_kv_cache_config()

        # Mock scheduler output.
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10
        scheduler_output.num_scheduled_tokens = {'req1': 10}
        scheduler_output.scheduled_spec_decode_tokens = {}
        scheduler_output.grammar_bitmask = None

        # Mock input_batch
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.max_decode_tokens = 1
        self.runner.input_batch.num_reqs = 1
        self.runner.input_batch.req_ids = ['req1']
        self.runner.input_batch.req_id_to_index = {'req1': 0}
        self.runner.input_batch.num_computed_tokens_cpu = np.array([10])
        self.runner.input_batch.token_ids_cpu = np.random.randint(
            0, 1000, (8, 64), dtype=np.int32)
        # Concrete numpy array so `.copy()` returns a real ndarray
        # (otherwise the surrounding `device_array` introspection on
        # `MagicMock` recurses on every `dtype` access).
        self.runner.input_batch.mamba_state_indices_cpu = np.zeros(
            self.runner.max_num_reqs, dtype=np.int32)

        # Mock block tables
        # there will be 2 block tables since there are 2 kv cache groups
        mock_block_table = MagicMock()
        mock_block_table.max_num_blocks_per_req = 8
        mock_block_table.get_cpu_tensor.return_value = np.zeros((1, 8),
                                                                dtype=np.int32)
        self.runner.input_batch.block_table = [
            mock_block_table, mock_block_table
        ]

        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance

        output = self.runner._prepare_inputs(scheduler_output)
        assert len(output) == 12
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs, req_ids_dp, padded_num_scheduled_tokens_per_dp_rank, tokens_indices_selector, shared_attention_metadata = output
        # assert it will create attention metadata for each layer.
        assert isinstance(attention_metadata, dict)
        assert len(attention_metadata) == 20

    def _create_mock_hybrid_kv_cache_config(self):
        mock_kv_cache_config = MagicMock()
        mock_kv_cache_group1 = MagicMock()
        mock_kv_cache_group1.layer_names = [f'layer.{i}' for i in range(10)]
        mock_kv_cache_group2 = MagicMock()
        mock_kv_cache_group2.layer_names = [
            f'layer.{i}' for i in range(10, 20)
        ]
        mock_kv_cache_config.kv_cache_groups = [
            mock_kv_cache_group1, mock_kv_cache_group2
        ]
        mock_kv_cache_config.has_mamba_layers = False
        self.runner.kv_cache_config = mock_kv_cache_config
        self.runner.use_hybrid_kvcache = True

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    def test_execute_continue_decode_with_routed_experts(
            self, mock_device_get, mock_continue_decode):
        """_execute_continue_decode() should retrieve and format MoE expert indices correctly."""
        runner = MagicMock()
        runner.scheduler_config.async_scheduling = False
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.dp_size = 1
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.input_batch.num_reqs = 2
        runner.input_batch.req_ids = ["req1", "req2"]
        runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1}
        runner.input_batch.num_tokens_no_spec = [10, 20]
        runner.input_batch.token_ids_cpu = np.zeros((8, 512), dtype=np.int32)
        runner.requests = {"req1": MagicMock(), "req2": MagicMock()}
        runner._get_min_remaining_slots.return_value = 30
        runner.vllm_config.additional_config = {"max_decode_steps": 5}
        runner.static_max_decode_steps = 5
        runner.model_config.get_vocab_size.return_value = 1000
        runner.model_config.hf_config = MagicMock(eos_token_id=999,
                                                  pad_token_id=0)
        runner.eos_token_id = 999
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {}
        runner.block_size = 16
        # Single KV-cache group, so routing slots key off group 0. Must be a
        # real int: MagicMock.__index__() returns 1, which would index past
        # the one-group block_ids below.
        runner.routed_experts_attn_gid = 0
        runner.requests["req1"].num_computed_tokens = 0
        runner.requests["req1"].block_ids = [[10]]
        runner.requests["req2"].num_computed_tokens = 0
        runner.requests["req2"].block_ids = [[20]]

        # Mock continue_decode output
        # Unpacks: generated_tokens, final_kv_caches, final_state, final_rng, all_expert_indices
        mock_generated_tokens = MagicMock()
        mock_all_expert_indices = MagicMock()
        mock_final_state = MagicMock()
        mock_continue_decode.return_value = (
            mock_generated_tokens,
            MagicMock(),  # kv_caches
            mock_final_state,  # final_state
            MagicMock(),  # final_rng
            mock_all_expert_indices,
            None)

        # continue_decode now returns fixed-size stacked buffers; the caller
        # trims them with final_state.step_counter via a single device_get of
        # (generated_tokens, all_expert_indices, step_counter).
        mock_tokens_cpu = np.zeros((5, 8), dtype=np.int32)
        mock_experts_cpu = np.arange(5 * 3 * 8 * 2,
                                     dtype=np.int32).reshape(5, 3, 8, 2)

        def device_get_side_effect(arg):
            if isinstance(arg, tuple) and len(arg) == 2:
                tokens_arg, _ = arg
                if tokens_arg is mock_generated_tokens:
                    return mock_tokens_cpu, np.int32(5)
            elif arg is mock_all_expert_indices:
                return mock_experts_cpu
            return arg

        mock_device_get.side_effect = device_get_side_effect

        # Setup scheduler output
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req1": 1, "req2": 1}

        runner._prepare_inputs.return_value = (
            np.zeros(8, dtype=np.int32),  # input_ids
            None,
            None,
            None,
            np.array([0, 1, -1, -1, -1, -1, -1, -1], dtype=np.int32),
            None,
            None,
            None,
            None,
            None,
            None,
            None)

        # Execute target method
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        TPUModelRunner._execute_continue_decode(runner, scheduler_output)

        mock_continue_decode.assert_called_once()
        assert mock_continue_decode.call_args.kwargs["max_decode_steps"] == 5
        assert mock_continue_decode.call_args.kwargs[
            "static_max_decode_steps"] == 5

        # Verify routed experts are formatted correctly:
        # routing_data: (num_reqs * actual_steps, num_layers, top_k) -> (10, 3, 2)
        # slot_mapping: (num_reqs * actual_steps,) -> (10,)
        output_runner_output = runner._continue_decode_output
        assert output_runner_output is not None
        assert output_runner_output.routed_experts is not None
        assert output_runner_output.routed_experts.routing_data.shape == (10,
                                                                          3, 2)
        assert output_runner_output.routed_experts.slot_mapping.shape == (10, )
        # req1 slots: block 10 * 16 + [0..4] = [160..164]
        # req2 slots: block 20 * 16 + [0..4] = [320..324]
        expected_slots = np.concatenate([
            np.arange(160, 165, dtype=np.int32),
            np.arange(320, 325, dtype=np.int32)
        ])
        np.testing.assert_array_equal(
            output_runner_output.routed_experts.slot_mapping, expected_slots)

        # Verify scheduler_output.num_scheduled_tokens is mutated to actual_steps = 5
        assert scheduler_output.num_scheduled_tokens["req1"] == 5
        assert scheduler_output.num_scheduled_tokens["req2"] == 5

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    def test_execute_continue_decode_standard(self, mock_device_get,
                                              mock_continue_decode):
        """_execute_continue_decode() should compute steps, slice/trim generated tokens, and update cpu state."""
        runner = MagicMock()
        runner.scheduler_config.async_scheduling = False
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.dp_size = 1
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.input_batch.num_reqs = 3
        runner.input_batch.req_ids = ["req1", "req2", "req3"]
        runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1, "req3": 2}

        # Initialize token_ids_cpu buffer
        runner.input_batch.token_ids_cpu = np.zeros((8, 512), dtype=np.int32)

        req_mock1 = MagicMock(output_token_ids=[1, 2, 3])
        req_mock2 = MagicMock(output_token_ids=[4])
        req_mock3 = MagicMock(output_token_ids=[5, 6])
        runner.requests = {
            "req1": req_mock1,
            "req2": req_mock2,
            "req3": req_mock3
        }

        # Setup remaining slots to check capping logic
        # req1 has 15 slots left, req2 has 5 slots left, req3 has 20 slots left.
        # min_remaining = 5. max_decode_steps should be capped at min(10, 5) = 5.
        runner._get_min_remaining_slots.return_value = 5
        runner.vllm_config.additional_config = {"max_decode_steps": 10}
        runner.static_max_decode_steps = 10

        runner.model_config.get_vocab_size.return_value = 1000
        runner.model_config.hf_config = MagicMock(eos_token_id=999,
                                                  pad_token_id=0)
        runner.eos_token_id = 999
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {}
        runner.continue_decode_eos_check_interval = 1

        # Mock continue_decode output
        mock_generated_tokens = MagicMock()
        mock_final_state = MagicMock()
        mock_continue_decode.return_value = (
            mock_generated_tokens,
            MagicMock(),  # kv_caches
            mock_final_state,  # final_state
            MagicMock(),  # final_rng
            None,  # all_expert_indices
            None,
        )

        # Mock jax.device_get output for generated tokens
        # Shape (max_decode_steps, batch_size) -> (5, 8)
        # We will simulate:
        # req1: generates [101, 102, 999, 0, 0] (hits EOS at step 2)
        # req2: generates [201, 202, 203, 204, 999] (hits EOS at step 4)
        # req3: generates [301, 302, 303, 304, 305] (no EOS)
        mock_tokens_cpu = np.zeros((5, 8), dtype=np.int32)
        mock_tokens_cpu[:, 0] = [101, 102, 999, 0, 0]
        mock_tokens_cpu[:, 1] = [201, 202, 203, 204, 999]
        mock_tokens_cpu[:, 2] = [301, 302, 303, 304, 305]

        # New contract: caller does a single device_get of
        # (generated_tokens, all_expert_indices, step_counter) and trims to
        # step_counter (= 5 here, so all rows are kept).
        def device_get_side_effect(arg):
            if isinstance(arg, tuple) and len(arg) == 2:
                tokens_arg, _ = arg
                if tokens_arg is mock_generated_tokens:
                    return mock_tokens_cpu, np.int32(5)
            return arg

        mock_device_get.side_effect = device_get_side_effect

        # Setup scheduler output
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {
            "req1": 1,
            "req2": 1,
            "req3": 1
        }

        # Mock attn_metadata.seq_lens_cpu
        mock_seq_lens_cpu = np.array([10, 20, 30, 0, 0, 0, 0, 0])
        runner.input_batch.num_tokens = mock_seq_lens_cpu.copy()
        runner.input_batch.num_tokens_no_spec = mock_seq_lens_cpu.copy()

        attn_metadata = MagicMock()
        attn_metadata.seq_lens_cpu = mock_seq_lens_cpu
        runner._prepare_inputs.return_value = (
            np.zeros(8, dtype=np.int32),  # input_ids
            None,
            attn_metadata,
            None,
            np.array([0, 1, 2, -1, -1, -1, -1, -1], dtype=np.int32),
            None,
            None,
            None,
            None,
            None,
            None,
            None)

        # Execute target method
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        TPUModelRunner._execute_continue_decode(runner, scheduler_output)

        # 1. Verify step capping logic: max_decode_steps = min(10, 5) = 5.
        mock_continue_decode.assert_called_once()
        assert mock_continue_decode.call_args.kwargs["max_decode_steps"] == 5
        assert mock_continue_decode.call_args.kwargs[
            "static_max_decode_steps"] == 10
        called_init_state = mock_continue_decode.call_args.kwargs["init_state"]
        expected_active_mask = np.array(
            [True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(called_init_state.active_mask,
                                      expected_active_mask)

        # 2. Verify generated tokens are trimmed at EOS and placed in output
        output = runner._continue_decode_output
        assert output is not None

        # req1: [101, 102, 999] (length 3)
        assert output.sampled_token_ids[0] == [101, 102, 999]
        # req2: [201, 202, 203, 204, 999] (length 5)
        assert output.sampled_token_ids[1] == [201, 202, 203, 204, 999]
        # req3: [301, 302, 303, 304, 305] (length 5)
        assert output.sampled_token_ids[2] == [301, 302, 303, 304, 305]

        # 3. Verify CPU token_ids_cpu buffer is updated correctly
        # req1 starts at 10. next 3 tokens written.
        np.testing.assert_array_equal(
            runner.input_batch.token_ids_cpu[0, 10:13], [101, 102, 999])
        # req2 starts at 20. next 5 tokens written.
        np.testing.assert_array_equal(
            runner.input_batch.token_ids_cpu[1, 20:25],
            [201, 202, 203, 204, 999])
        # req3 starts at 30. next 5 tokens written.
        np.testing.assert_array_equal(
            runner.input_batch.token_ids_cpu[2, 30:35],
            [301, 302, 303, 304, 305])

        # 4. Verify request output_token_ids are extended
        assert req_mock1.output_token_ids == [1, 2, 3, 101, 102, 999]
        assert req_mock2.output_token_ids == [4, 201, 202, 203, 204, 999]
        assert req_mock3.output_token_ids == [5, 6, 301, 302, 303, 304, 305]

        # 5. Verify attention metadata sequence lengths are advanced
        # req1: 10 -> 13
        assert attn_metadata.seq_lens_cpu[0] == 13
        # req2: 20 -> 25
        assert attn_metadata.seq_lens_cpu[1] == 25
        # req3: 30 -> 35
        assert attn_metadata.seq_lens_cpu[2] == 35

        # 6. Verify default continue_decode_eos_check_interval parameter passed to continue_decode
        assert mock_continue_decode.call_args.kwargs[
            "continue_decode_eos_check_interval"] == 1

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    def test_execute_continue_decode_grouped_attention_metadata(
            self, mock_device_get, mock_continue_decode):
        """_execute_continue_decode() should accept GroupedAttentionMetadata for hybrid GDN models."""
        from tpu_inference.layers.common.attention_metadata import (
            AttentionMetadata, GroupedAttentionMetadata)

        runner = MagicMock()
        runner.scheduler_config.async_scheduling = False
        runner.max_num_reqs = 4
        runner.max_model_len = 512
        runner.dp_size = 1
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.input_batch.num_reqs = 2
        runner.input_batch.req_ids = ["req1", "req2"]
        runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1}
        runner.input_batch.token_ids_cpu = np.zeros((4, 512), dtype=np.int32)
        runner.input_batch.num_tokens = np.array([10, 20, 0, 0],
                                                 dtype=np.int32)
        runner.input_batch.num_tokens_no_spec = np.array([10, 20, 0, 0],
                                                         dtype=np.int32)

        req_mock1 = MagicMock(output_token_ids=[1, 2])
        req_mock2 = MagicMock(output_token_ids=[3])
        runner.requests = {"req1": req_mock1, "req2": req_mock2}

        runner._get_min_remaining_slots.return_value = 3
        runner.vllm_config.additional_config = {"max_decode_steps": 4}
        runner.static_max_decode_steps = 4
        runner.eos_token_id = (999, )
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {
            "layers.0.attn": 0,
            "layers.1.gdn": 1,
        }
        runner.continue_decode_eos_check_interval = 1

        g0 = AttentionMetadata(
            input_positions=jnp.array([9, 19, 0, 0], dtype=jnp.int32),
            block_tables=jnp.zeros((4, 16), dtype=jnp.int32),
            seq_lens=jnp.array([10, 20, 0, 0], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 1, 2, 2, 2], dtype=jnp.int32),
            request_distribution=jnp.array([2, 0, 2], dtype=jnp.int32),
            mamba_state_indices=jnp.array([2, 0, 1, 3], dtype=jnp.int32),
            padded_num_reqs=4,
            is_decode=True,
        )
        g1 = AttentionMetadata(
            input_positions=g0.input_positions,
            block_tables=jnp.zeros((4, 2), dtype=jnp.int32),
            seq_lens=g0.seq_lens,
            query_start_loc=g0.query_start_loc,
            request_distribution=g0.request_distribution,
            mamba_state_indices=g0.mamba_state_indices,
            padded_num_reqs=4,
            is_decode=True,
        )
        grouped_attn = GroupedAttentionMetadata(
            groups=(g0, g1),
            layer_names_per_group=(("layers.0.attn", ), ("layers.1.gdn", )),
        )

        mock_generated_tokens = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.step_counter = jnp.array(3, dtype=jnp.int32)
        mock_continue_decode.return_value = (
            mock_generated_tokens,
            MagicMock(),
            mock_final_state,
            MagicMock(),
            None,
            None,
        )

        mock_tokens_cpu = np.zeros((3, 4), dtype=np.int32)
        mock_tokens_cpu[:, 0] = [101, 102, 103]
        mock_tokens_cpu[:, 1] = [201, 999, 0]

        def device_get_side_effect(arg):
            if isinstance(arg, tuple) and len(arg) == 2:
                tokens_arg, _ = arg
                if tokens_arg is mock_generated_tokens:
                    return mock_tokens_cpu, np.int32(3)
            return arg

        mock_device_get.side_effect = device_get_side_effect

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req1": 1, "req2": 1}

        runner._prepare_inputs.return_value = (
            np.zeros(4, dtype=np.int32),
            None,
            grouped_attn,
            None,
            np.array([0, 1, -1, -1], dtype=np.int32),
            None,
            None,
            4,
            None,
            None,
            None,
            None,
        )

        TPUModelRunner._execute_continue_decode(runner, scheduler_output)

        mock_continue_decode.assert_called_once()
        called_init_state = mock_continue_decode.call_args.kwargs["init_state"]
        assert isinstance(called_init_state.attn_metadata,
                          GroupedAttentionMetadata)
        assert called_init_state.attn_metadata.padded_num_reqs == 4
        output = runner._continue_decode_output
        assert output.sampled_token_ids[0] == [101, 102, 103]
        assert output.sampled_token_ids[1] == [201, 999]
        assert scheduler_output.num_scheduled_tokens["req1"] == 3
        assert scheduler_output.num_scheduled_tokens["req2"] == 2

    def test_get_min_remaining_slots_mamba_align_sync_and_async(self):
        """_get_min_remaining_slots() should cap steps to not cross mamba_block_size boundaries in align mode."""
        self.runner.max_model_len = 512
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.num_tokens = np.array([13, 20, 0, 0],
                                                      dtype=np.int32)

        # 1. Without mamba align mode: capped only by max_model_len (512 - 20 = 492)
        self.runner.kv_cache_config = MagicMock(has_mamba_layers=False)
        self.runner.cache_config.mamba_cache_mode = "none"
        self.runner.cache_config.mamba_block_size = None
        self.runner.scheduler_config.async_scheduling = False
        assert self.runner._get_min_remaining_slots() == 492

        # 2. Sync scheduling with mamba_cache_mode="align", mamba_block_size=16:
        # num_tokens=13 -> mamba_remaining = 16 - 12 = 4 (stops at token 16)
        # num_tokens=20 -> mamba_remaining = 16 - 3 = 13 (stops at token 32)
        self.runner.kv_cache_config = MagicMock(has_mamba_layers=True)
        self.runner.cache_config.mamba_cache_mode = "align"
        self.runner.cache_config.mamba_block_size = 16
        self.runner.scheduler_config.async_scheduling = False
        assert self.runner._get_min_remaining_slots() == 4

        # At exact block boundary token (seq_len=16 -> last token of block 0):
        # mamba_remaining = 16 - ((16 - 1) % 16) = 1
        self.runner.input_batch.num_reqs = 1
        self.runner.input_batch.num_tokens = np.array([16, 0], dtype=np.int32)
        assert self.runner._get_min_remaining_slots() == 1

        # First token of block 1 (seq_len=17): mamba_remaining = 16
        self.runner.input_batch.num_tokens = np.array([17, 0], dtype=np.int32)
        assert self.runner._get_min_remaining_slots() == 16

        # 3. Async scheduling with mamba_cache_mode="align", mamba_block_size=16:
        # When mamba_remaining > 1, caps at max(1, mamba_remaining - 1) so step k
        # stops 1 token before the boundary and in-flight step k+1 executes the
        # single boundary token using the current mamba block.
        self.runner.scheduler_config.async_scheduling = True
        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.num_tokens = np.array([13, 20, 0, 0],
                                                      dtype=np.int32)
        assert self.runner._get_min_remaining_slots() == 3

        # When step k+1 arrives at the boundary token (seq_len=16, mamba_remaining=1):
        # max(1, 1 - 1) = 1 step
        self.runner.input_batch.num_reqs = 1
        self.runner.input_batch.num_tokens = np.array([16, 0], dtype=np.int32)
        assert self.runner._get_min_remaining_slots() == 1

        # First token of block 1 in async mode (seq_len=17, mamba_remaining=16):
        # max(1, 16 - 1) = 15 steps (stops at token 31)
        self.runner.input_batch.num_tokens = np.array([17, 0], dtype=np.int32)
        assert self.runner._get_min_remaining_slots() == 15

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    def test_execute_continue_decode_eos_check_interval_config(
            self, mock_device_get, mock_continue_decode):
        """_execute_continue_decode() should pass continue_decode_eos_check_interval (sourced from the CONTINUE_DECODE_EOS_CHECK_INTERVAL env var) through to continue_decode."""
        runner = MagicMock()
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.dp_size = 1
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.scheduler_config.async_scheduling = False
        runner.vllm_config.additional_config = {
            "enable_continue_decode": True,
            "max_decode_steps": 5,
        }
        runner.continue_decode_eos_check_interval = 5
        runner.static_max_decode_steps = 5
        runner.input_batch.num_reqs = 1
        runner.input_batch.req_ids = ["req1"]
        runner.input_batch.req_id_to_index = {"req1": 0}
        runner.input_batch.token_ids_cpu = np.zeros((8, 512), dtype=np.int32)
        runner.requests = {"req1": MagicMock(output_token_ids=[])}
        runner._get_min_remaining_slots.return_value = 5
        runner.model_config.get_vocab_size.return_value = 1000
        runner.eos_token_id = 999
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {}

        mock_generated_tokens = MagicMock()
        mock_final_state = MagicMock()
        mock_continue_decode.return_value = (
            mock_generated_tokens,
            MagicMock(),
            mock_final_state,
            MagicMock(),
            None,
            None,
        )

        mock_tokens_cpu = np.zeros((5, 8), dtype=np.int32)
        mock_tokens_cpu[:, 0] = [101, 999, 0, 0, 0]

        def device_get_side_effect(arg):
            if isinstance(arg, tuple) and len(arg) == 2:
                return mock_tokens_cpu, np.int32(5)
            return arg

        mock_device_get.side_effect = device_get_side_effect

        mock_seq_lens_cpu = np.array([10, 0, 0, 0, 0, 0, 0, 0])
        runner.input_batch.num_tokens = mock_seq_lens_cpu.copy()
        runner.input_batch.num_tokens_no_spec = mock_seq_lens_cpu.copy()

        attn_metadata = MagicMock()
        attn_metadata.seq_lens_cpu = mock_seq_lens_cpu
        runner._prepare_inputs.return_value = (
            np.zeros(8, dtype=np.int32),
            None,
            attn_metadata,
            None,
            np.array([0, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        from tpu_inference.runner.tpu_runner import TPUModelRunner
        TPUModelRunner._execute_continue_decode(
            runner, MagicMock(num_scheduled_tokens={"req1": 1}))

        mock_continue_decode.assert_called_once()
        assert mock_continue_decode.call_args.kwargs[
            "continue_decode_eos_check_interval"] == 5

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    def test_execute_continue_decode_sharded(self, mock_device_get,
                                             mock_continue_decode):
        """_execute_continue_decode() should realign generated tokens correctly when dp_size > 1."""
        runner = MagicMock()
        runner.scheduler_config.async_scheduling = False
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.dp_size = 2
        runner.vllm_config.parallel_config.data_parallel_size = 2
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.input_batch.num_reqs = 3
        runner.input_batch.req_ids = ["req1", "req2", "req3"]
        # req1 and req2 are in Rank 0 (slots 0, 1)
        # req3 is in Rank 1 (slot 0)
        runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1, "req3": 2}

        # Initialize token_ids_cpu buffer
        runner.input_batch.token_ids_cpu = np.zeros((8, 512), dtype=np.int32)

        req_mock1 = MagicMock(output_token_ids=[1, 2, 3])
        req_mock2 = MagicMock(output_token_ids=[4])
        req_mock3 = MagicMock(output_token_ids=[5, 6])
        runner.requests = {
            "req1": req_mock1,
            "req2": req_mock2,
            "req3": req_mock3
        }

        runner._get_min_remaining_slots.return_value = 5
        runner.vllm_config.additional_config = {"max_decode_steps": 10}
        runner.static_max_decode_steps = 10

        runner.model_config.get_vocab_size.return_value = 1000
        runner.model_config.hf_config = MagicMock(eos_token_id=999,
                                                  pad_token_id=0)
        runner.eos_token_id = 999
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {}

        # Mock continue_decode output
        mock_generated_tokens = MagicMock()
        mock_final_state = MagicMock()
        mock_continue_decode.return_value = (
            mock_generated_tokens,
            MagicMock(),  # kv_caches
            mock_final_state,  # final_state
            MagicMock(),  # final_rng
            None,  # all_expert_indices
            None,
        )

        # Mock jax.device_get output for generated tokens
        # Shape (max_decode_steps, padded_total_num_scheduled_tokens) -> (5, 8)
        mock_tokens_cpu = np.zeros((5, 8), dtype=np.int32)
        # Rank 0 (slots 0, 1, 2, 3)
        mock_tokens_cpu[:, 0] = [101, 102, 999, 0, 0]  # req1
        mock_tokens_cpu[:, 1] = [201, 202, 203, 204, 999]  # req2
        # Rank 1 (slots 0, 1, 2, 3)
        mock_tokens_cpu[:, 4] = [301, 302, 303, 304, 305]  # req3

        # New contract: caller does a single device_get of
        # (generated_tokens, all_expert_indices, step_counter) and trims to
        # step_counter (= 5 here, so all rows are kept).
        def device_get_side_effect(arg):
            if isinstance(arg, tuple) and len(arg) == 2:
                tokens_arg, _ = arg
                if tokens_arg is mock_generated_tokens:
                    return mock_tokens_cpu, np.int32(5)
            return arg

        mock_device_get.side_effect = device_get_side_effect

        # Setup scheduler output
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {
            "req1": 1,
            "req2": 1,
            "req3": 1
        }

        # Mock attn_metadata.seq_lens_cpu
        mock_seq_lens_cpu = np.array([10, 20, 30, 0, 0, 0, 0, 0])
        runner.input_batch.num_tokens = mock_seq_lens_cpu.copy()
        runner.input_batch.num_tokens_no_spec = mock_seq_lens_cpu.copy()

        attn_metadata = MagicMock()
        attn_metadata.seq_lens_cpu = mock_seq_lens_cpu

        # logits_indices: Rank 0 has 2 active, Rank 1 has 1 active.
        # padded_num_reqs_per_dp_rank = 4
        # logits_indices = [0, 1, -1, -1,  0, -1, -1, -1]
        # logits_indices_selector = [0, 1, 4]
        runner._prepare_inputs.return_value = (
            np.zeros(8, dtype=np.int32),  # input_ids
            None,
            attn_metadata,
            None,
            np.array([0, 1, -1, -1, 0, -1, -1, -1],
                     dtype=np.int32),  # logits_indices
            None,
            [0, 1, 4],  # logits_indices_selector
            None,
            None,
            None,
            [0, 1, 4],
            None)

        # Execute target method
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        TPUModelRunner._execute_continue_decode(runner, scheduler_output)

        # Verify generated tokens are trimmed at EOS and placed in output
        output = runner._continue_decode_output
        assert output is not None
        assert output.req_id_to_index == {"req1": 0, "req2": 1, "req3": 2}

        # req1: [101, 102, 999] (length 3)
        assert output.sampled_token_ids[0] == [101, 102, 999]
        # req2: [201, 202, 203, 204, 999] (length 5)
        assert output.sampled_token_ids[1] == [201, 202, 203, 204, 999]
        # req3: [301, 302, 303, 304, 305] (length 5)
        assert output.sampled_token_ids[2] == [301, 302, 303, 304, 305]

        # Verify CPU token_ids_cpu buffer is updated correctly
        # req1 starts at 10. next 3 tokens written.
        np.testing.assert_array_equal(
            runner.input_batch.token_ids_cpu[0, 10:13], [101, 102, 999])
        # req2 starts at 20. next 5 tokens written.
        np.testing.assert_array_equal(
            runner.input_batch.token_ids_cpu[1, 20:25],
            [201, 202, 203, 204, 999])
        # req3 starts at 30. next 5 tokens written.
        np.testing.assert_array_equal(
            runner.input_batch.token_ids_cpu[2, 30:35],
            [301, 302, 303, 304, 305])

        # Verify request output_token_ids are extended
        assert req_mock1.output_token_ids == [1, 2, 3, 101, 102, 999]
        assert req_mock2.output_token_ids == [4, 201, 202, 203, 204, 999]
        assert req_mock3.output_token_ids == [5, 6, 301, 302, 303, 304, 305]

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    def test_execute_continue_decode_async(self, mock_device_get,
                                           mock_continue_decode):
        """_execute_continue_decode() under async scheduling should handle pre_async_results, extract next_tokens_in_tpu, and return AsyncTPUModelRunnerOutput."""
        runner = MagicMock()
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.dp_size = 1
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.is_moe_model = False
        runner.scheduler_config.async_scheduling = True
        runner.input_batch.num_reqs = 2
        runner.input_batch.req_ids = ["req1", "req2"]
        runner.input_batch.req_id_to_index = {"req1": 0, "req2": 1}

        runner.input_batch.token_ids_cpu = np.zeros((8, 512), dtype=np.int32)
        runner.input_batch.num_tokens = np.array([10, 20, 0, 0, 0, 0, 0, 0],
                                                 dtype=np.int32)
        runner.input_batch.num_tokens_no_spec = np.array(
            [10, 20, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        runner.input_batch.num_computed_tokens_cpu = np.array(
            [10, 20, 0, 0, 0, 0, 0, 0], dtype=np.int32)

        req_mock1 = MagicMock(output_token_ids=[1, 2, 3],
                              num_computed_tokens=10)
        req_mock2 = MagicMock(output_token_ids=[4], num_computed_tokens=20)
        runner.requests = {
            "req1": req_mock1,
            "req2": req_mock2,
        }

        runner._get_min_remaining_slots.return_value = 5
        runner.vllm_config.additional_config = {"max_decode_steps": 5}
        runner.static_max_decode_steps = 5
        runner.continue_decode_eos_check_interval = 5
        runner.model_config.get_vocab_size.return_value = 1000
        runner.model_config.hf_config = MagicMock(eos_token_id=999,
                                                  pad_token_id=0)
        runner.eos_token_id = 999
        runner.pad_token_id = 0
        runner.layer_name_to_kvcache_index = {}

        mock_generated_tokens = MagicMock()
        mock_generated_tokens.shape = (5, 2)
        mock_final_state = MagicMock()
        mock_final_state.step_counter = jnp.array(5, dtype=jnp.int32)
        mock_continue_decode.return_value = (
            mock_generated_tokens,
            MagicMock(),  # kv_caches
            mock_final_state,
            MagicMock(),  # final_rng
            None,
            None,
        )

        mock_tokens_cpu = np.zeros((5, 2), dtype=np.int32)
        mock_tokens_cpu[:, 0] = [101, 102, 999, 0, 0]
        mock_tokens_cpu[:, 1] = [201, 202, 203, 204, 205]

        def device_get_side_effect(arg):
            if isinstance(arg, tuple) and len(arg) == 2:
                return mock_tokens_cpu, np.int32(3)
            return arg

        mock_device_get.side_effect = device_get_side_effect

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req1": 1, "req2": 1}

        attn_metadata = MagicMock()
        attn_metadata.seq_lens_cpu = np.array([10, 20, 0, 0, 0, 0, 0, 0])
        runner._prepare_inputs.return_value = (
            np.zeros(8, dtype=np.int32), None, attn_metadata, None,
            np.array([0, 1, -1, -1, -1, -1, -1, -1],
                     dtype=np.int32), None, None, None, None, None, None, None)

        from tpu_inference.runner.tpu_runner import TPUModelRunner
        result = TPUModelRunner._execute_continue_decode(
            runner, scheduler_output)

        assert result is None
        assert runner._pre_async_results is not None
        assert runner._pre_async_results.is_continue_decode is True
        assert runner._continue_decode_output is not None

        assert mock_continue_decode.call_args.kwargs[
            "continue_decode_eos_check_interval"] == runner.continue_decode_eos_check_interval
        # Verify output resolution via get_output()
        async_out = runner._continue_decode_output
        model_runner_output = async_out.get_output()
        assert model_runner_output.sampled_token_ids[0] == [101, 102, 999]
        assert model_runner_output.sampled_token_ids[1] == [201, 202, 203]

    @patch('tpu_inference.runner.tpu_runner.continue_decode')
    @patch('jax.device_get')
    @patch('jax.copy_to_host_async', side_effect=lambda x: x)
    def test_execute_continue_decode_async_multistep_mamba_boundary(
            self, mock_copy_async, mock_device_get, mock_continue_decode):
        """Multi-step async continue_decode with GDN mamba_cache_mode='align'
        must accurately track token_ids_cpu, num_tokens, num_computed_tokens_cpu,
        and output_token_ids across update_states and never cross a mamba_block_size
        boundary."""
        self.runner.scheduler_config.async_scheduling = True
        self.runner.speculative_config = None
        self.runner.enable_continue_decode = True
        self.runner.static_max_decode_steps = 5
        self.runner.continue_decode_eos_check_interval = 1
        self.runner.eos_token_id = 999
        self.runner.pad_token_id = 0
        self.runner.max_model_len = 512
        self.runner.kv_cache_config = MagicMock(has_mamba_layers=True)
        self.runner.cache_config.mamba_cache_mode = "align"
        self.runner.cache_config.mamba_block_size = 16
        self.runner.layer_name_to_kvcache_index = {}
        self.runner.model = MagicMock()

        # Setup 1 request in input_batch:
        # 10 prompt tokens [10..19], plus 3 output tokens already generated [20, 21, 22]
        # so before Step 1, token_ids_cpu[0, :13] = [10..22], num_tokens = 13,
        # num_computed_tokens_cpu = 12 (token at index 12 is the input to Step 1).
        req_id = "req1"
        req_state = MagicMock()
        req_state.req_id = req_id
        req_state.prompt_token_ids = list(range(10, 20))
        req_state.num_prompt_tokens = 10
        req_state.output_token_ids = [20, 21, 22]
        req_state.num_computed_tokens = 12
        req_state.block_ids = ([1], [1])
        req_state.mrope_positions = None
        req_state.mrope_position_delta = None
        self.runner.requests = {req_id: req_state}

        ib = self.runner.input_batch
        ib.num_reqs = 1
        ib._req_ids = [req_id]
        ib.req_id_to_index = {req_id: 0}
        ib.num_prompt_tokens[0] = 10
        ib.token_ids_cpu[0, :13] = np.arange(10, 23, dtype=np.int32)
        ib.num_tokens[0] = 13
        ib.num_tokens_no_spec[0] = 13
        ib.num_computed_tokens_cpu[0] = 12
        ib.request_distribution = [1, 1, 1]
        ib.has_mamba_layers = False
        self.runner._pre_async_results = None

        captured_steps = []

        def fake_prepare_inputs(sched_out):
            idx = 0
            pos = int(ib.num_computed_tokens_cpu[idx])
            tok = int(ib.token_ids_cpu[idx, pos])
            seq_len = pos + int(sched_out.num_scheduled_tokens[req_id])
            captured_steps.append((pos, tok, seq_len, int(ib.num_tokens[idx])))
            attn_meta = MagicMock()
            attn_meta.seq_lens_cpu = np.array([seq_len], dtype=np.int32)
            return (
                jnp.array([tok] + [0] * 7, dtype=jnp.int32),
                jnp.array([pos] + [0] * 7, dtype=jnp.int32),
                attn_meta,
                MagicMock(),
                jnp.array([0] + [-1] * 7, dtype=jnp.int32),
                None,
                None,
                8,
                {
                    0: [req_id]
                },
                8,
                None,
                MagicMock(),
            )

        self.runner._prepare_inputs = fake_prepare_inputs

        # --- Step 1: starts at num_tokens=13 (pos=12, seq_len=13) ---
        # mamba_remaining = 16 - ((13 - 1) % 16) = 4 -> async cap = max(1, 4 - 1) = 3
        # Generates 3 tokens: [101, 102, 103] (for seq_len = 13, 14, 15, stopping 1 before 16)
        step1_tokens = np.zeros((5, 8), dtype=np.int32)
        step1_tokens[:3, 0] = [101, 102, 103]
        step1_state = MagicMock()
        step1_state.step_counter = jnp.array(3, dtype=jnp.int32)
        mock_continue_decode.return_value = (
            step1_tokens,
            MagicMock(),
            step1_state,
            MagicMock(),
            None,
            None,
        )
        sched_out1 = MagicMock()
        sched_out1.num_scheduled_tokens = {req_id: 1}
        self.runner._execute_continue_decode(sched_out1)

        assert int(
            mock_continue_decode.call_args.kwargs["max_decode_steps"]) == 3
        assert captured_steps[-1] == (12, 22, 13, 13)
        # Placeholders added at end of Step 1: 13 + 3 = 16
        assert ib.num_tokens[0] == 16
        assert len(req_state.output_token_ids) == 6

        # --- Host schedules Step 2 while Step 1 is in flight ---
        # Host only knows 1 token was scheduled in Step 1:
        # num_computed_tokens = 13, num_output_tokens = 4 (3 prior + 1 placeholder)
        cached_reqs2 = MagicMock()
        cached_reqs2.req_ids = [req_id]
        cached_reqs2.num_computed_tokens = [13]
        cached_reqs2.new_block_ids = [None]
        cached_reqs2.resumed_req_ids = set()
        cached_reqs2.num_output_tokens = [4]
        sched_out2 = MagicMock()
        sched_out2.finished_req_ids = set()
        sched_out2.free_encoder_mm_hashes = []
        sched_out2.num_scheduled_tokens = {req_id: 1}
        sched_out2.total_num_scheduled_tokens = 1
        sched_out2.scheduled_new_reqs = []
        sched_out2.scheduled_cached_reqs = cached_reqs2
        sched_out2.scheduled_spec_decode_tokens = {}
        sched_out2.preempted_req_ids = set()

        self.runner.persistent_batch_manager.update_states(sched_out2, None)
        # update_states truncated num_tokens to 10 + 4 = 14 (off from 13!)
        assert ib.num_tokens_no_spec[0] == 14
        assert ib.num_computed_tokens_cpu[0] == 13

        # --- Step 2: _modify_prev_results resolves Step 1 ([101, 102, 103]) ---
        # Must write [101, 102, 103] at 13:16, set num_tokens=16, num_computed_tokens_cpu=15,
        # and run 1 boundary step (seq_len=16, mamba_remaining=1 -> async cap=1).
        mock_device_get.side_effect = lambda arg: (step1_tokens, np.int32(3))
        step2_tokens = np.zeros((5, 8), dtype=np.int32)
        step2_tokens[0, 0] = 104
        step2_state = MagicMock()
        step2_state.step_counter = jnp.array(1, dtype=jnp.int32)
        mock_continue_decode.return_value = (
            step2_tokens,
            MagicMock(),
            step2_state,
            MagicMock(),
            None,
            None,
        )
        self.runner._execute_continue_decode(sched_out2)

        assert int(
            mock_continue_decode.call_args.kwargs["max_decode_steps"]) == 1
        # Verify _prepare_inputs saw pos=15, tok=103, seq_len=16, num_tokens=16
        assert captured_steps[-1] == (15, 103, 16, 16)
        np.testing.assert_array_equal(ib.token_ids_cpu[0, 13:16],
                                      [101, 102, 103])
        assert sched_out1.num_scheduled_tokens[req_id] == 3

        # --- Host schedules Step 3 after Step 1 output returned (Step 2 in flight) ---
        cached_reqs3 = MagicMock()
        cached_reqs3.req_ids = [req_id]
        cached_reqs3.num_computed_tokens = [16]
        cached_reqs3.new_block_ids = [None]
        cached_reqs3.resumed_req_ids = set()
        cached_reqs3.num_output_tokens = [7]
        sched_out3 = MagicMock()
        sched_out3.finished_req_ids = set()
        sched_out3.free_encoder_mm_hashes = []
        sched_out3.num_scheduled_tokens = {req_id: 1}
        sched_out3.total_num_scheduled_tokens = 1
        sched_out3.scheduled_new_reqs = []
        sched_out3.scheduled_cached_reqs = cached_reqs3
        sched_out3.scheduled_spec_decode_tokens = {}
        sched_out3.preempted_req_ids = set()

        self.runner.persistent_batch_manager.update_states(sched_out3, None)

        # --- Step 3: _modify_prev_results resolves Step 2 ([104] at index 16) ---
        # Starts at num_tokens=17 (pos=16, seq_len=17, first token of mamba block 1).
        # mamba_remaining = 16 -> async cap = 15 -> capped by static_max_decode_steps=5 to 5.
        mock_device_get.side_effect = lambda arg: (step2_tokens, np.int32(1))
        step3_tokens = np.zeros((5, 8), dtype=np.int32)
        step3_tokens[:5, 0] = [105, 106, 107, 108, 109]
        step3_state = MagicMock()
        step3_state.step_counter = jnp.array(5, dtype=jnp.int32)
        mock_continue_decode.return_value = (
            step3_tokens,
            MagicMock(),
            step3_state,
            MagicMock(),
            None,
            None,
        )
        self.runner._execute_continue_decode(sched_out3)

        assert int(
            mock_continue_decode.call_args.kwargs["max_decode_steps"]) == 5
        assert captured_steps[-1] == (16, 104, 17, 17)
        assert ib.token_ids_cpu[0, 16] == 104
        assert sched_out2.num_scheduled_tokens[req_id] == 1
        assert req_state.output_token_ids[:7] == [
            20, 21, 22, 101, 102, 103, 104
        ]

    def test_mesh_devices_sorting(self):

        class MockDevice:

            def __init__(self, coords, core_on_chip):
                self.coords = coords
                self.core_on_chip = core_on_chip
                self.device_kind = "unknown"
                self.platform = "tpu"

            def __repr__(self):
                return (f"MockDevice(coords={self.coords}, "
                        f"core_on_chip={self.core_on_chip})")

        device_0 = MockDevice((1, 0, 0), 0)
        device_1 = MockDevice((0, 0, 1), 0)
        device_2 = MockDevice((0, 0, 0), 1)
        device_3 = MockDevice((0, 0, 0), 0)
        device_4 = MockDevice((1, 0, 0), 1)
        device_5 = MockDevice((0, 0, 1), 1)
        device_6 = MockDevice((1, 1, 0), 0)
        device_7 = MockDevice((1, 1, 0), 1)

        unordered_devices = [
            device_0, device_1, device_2, device_3, device_4, device_5,
            device_6, device_7
        ]
        expected_sorted_devices = [
            device_3, device_2, device_0, device_4, device_6, device_7,
            device_1, device_5
        ]

        model_config = ModelConfig(tokenizer_mode="auto",
                                   trust_remote_code=False,
                                   seed=0,
                                   dtype='bfloat16')
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        scheduler_config = SchedulerConfig(max_num_seqs=16,
                                           max_model_len=1024,
                                           is_encoder_decoder=False)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        speculative_config = SpeculativeConfig(
            model='ngram',
            num_speculative_tokens=5,
            prompt_lookup_max=4,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            speculative_config=speculative_config,
            observability_config={},
            additional_config={},
        )

        mock_sharding_config = MagicMock()
        mock_sharding_config.model_dp_size = 8
        mock_sharding_config.total_dp_size = 8
        mock_sharding_config.attn_dp_size = 1
        mock_sharding_config.attn_dp_expert_size = 1
        mock_sharding_config.expert_size = 1
        mock_sharding_config.tp_size = 1
        mock_sharding_config.decode_cp_size = 1
        mock_sharding_config.prefill_cp_size = 1
        vllm_config.sharding_config = mock_sharding_config
        mesh_shape = (8, 1, 1, 1, 1, 1, 1)

        # Case A: envs.NEW_MODEL_DESIGN = True and envs.TPU_MESH_SORT_BY_COORDS = True
        with patch('tpu_inference.runner.tpu_runner.envs.NEW_MODEL_DESIGN', True), \
             patch('tpu_inference.runner.tpu_runner.envs.TPU_MESH_SORT_BY_COORDS', True):
            runner_a = TPUModelRunner(vllm_config, devices=unordered_devices)
            assert runner_a.mesh.devices.shape == mesh_shape
            flat_results_a = runner_a.mesh.devices.flatten().tolist()
            assert flat_results_a == expected_sorted_devices

        # Case B: envs.NEW_MODEL_DESIGN = True and envs.TPU_MESH_SORT_BY_COORDS = False
        # Assume create_device_mesh returns a dummy array
        dummy_mesh_array = np.array(unordered_devices).reshape(mesh_shape)

        with patch('tpu_inference.runner.tpu_runner.envs.NEW_MODEL_DESIGN', True), \
             patch('tpu_inference.runner.tpu_runner.envs.TPU_MESH_SORT_BY_COORDS', False), \
             patch('tpu_inference.runner.tpu_runner.mesh_utils.create_device_mesh', return_value=dummy_mesh_array):
            runner_b = TPUModelRunner(vllm_config, devices=unordered_devices)
            assert runner_b.mesh.devices.shape == mesh_shape
            flat_results_b = runner_b.mesh.devices.flatten().tolist()
            assert flat_results_b == unordered_devices


class TestTPUJaxRunnerMultimodalModelLoadedForTextOnly:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        # Setup the runner with the model_config.is_multimodal_model set to True but get_model returning None for embed_multimodal_fn and embed_input_ids_fn.
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=self._model_get_model()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh), \
             patch('jax.device_put', side_effect=lambda x, *args, **kwargs: x):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            # Set multimodal_config to not None, such that the is_multimodal_model property of model_config is True.
            model_config.multimodal_config = MagicMock()

            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self):
        mock_multimodal_fns = MultiModalInterface(
            precompile_vision_encoder_fn=None,
            embed_multimodal_fn=None,
            embed_input_ids_fn=None,
            get_mrope_input_positions_fn=None)
        return ModelInterface(
            model_fn=MagicMock(),
            compute_logits_fn=MagicMock(),
            pooler_fn=MagicMock(),
            combine_hidden_states_fn=MagicMock(),
            multimodal_fns=mock_multimodal_fns,
            state=MagicMock(),
            state_leaves=MagicMock(),
            lora_manager=None,
            model=None,
        )

    def test_is_multimodal_model(self):
        # Precondition: make sure the model_config claims the model supports MM.
        assert self.runner.model_config.is_multimodal_model

        # Precondition: load the model and returns embed_multimodal_fn as None.
        assert self.runner.embed_multimodal_fn is None

        assert not self.runner.is_multimodal_model

        self.runner.embed_input_ids_fn = MagicMock()
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = [jnp.ones((10, 128))]
        dummy_is_mm_embed = jnp.array([False, True, True], dtype=jnp.bool_)
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds,
                                              dummy_is_mm_embed)
        self.runner.embed_input_ids_fn.assert_not_called()


class TestTPUJaxRunnerDisableMM:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))

    def _model_get_model(self):
        mock_multimodal_fns = MultiModalInterface(
            precompile_vision_encoder_fn=None,
            embed_multimodal_fn=MagicMock(),
            embed_input_ids_fn=MagicMock(),
            get_mrope_input_positions_fn=None)
        return ModelInterface(
            model_fn=MagicMock(),
            compute_logits_fn=MagicMock(),
            pooler_fn=MagicMock(),
            combine_hidden_states_fn=MagicMock(),
            multimodal_fns=mock_multimodal_fns,
            state=MagicMock(),
            state_leaves=MagicMock(),
            lora_manager=None,
            model=None,
        )

    @pytest.mark.parametrize(
        "limit_per_prompt, still_mm_after_loading_model",
        [
            ({
                "image": BaseDummyOptions(count=0),
                "video": BaseDummyOptions(count=0)
            }, False),
            ({
                "video": BaseDummyOptions(count=0)
            }, False),
            ({
                "image": BaseDummyOptions(count=0),
                "video": BaseDummyOptions(count=1)
            }, True),
            (
                {
                    # Empty limit means no limit, which should not disable MM.
                },
                True)
        ])
    def test_multimodal_model_loading_with_limits(
            self, limit_per_prompt, still_mm_after_loading_model):
        """Test that "--limit-mm-per-prompt" config can disable multi-modality for a multi-modal model.

        If an user *explicitly* sets the limit for all modalities to 0, then we can safely disable multi-modality even if the model itself claims to be multimodal.
        """
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=self._model_get_model()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh), \
             patch('jax.device_put', side_effect=lambda x, *args, **kwargs: x):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            model_config.multimodal_config = MagicMock()
            model_config.multimodal_config.limit_per_prompt = limit_per_prompt

            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                observability_config={},
                additional_config={},
            )

            runner = TPUModelRunner(vllm_config, devices=self.mock_devices)
            # Precondition: make sure the model_config claims the model supports MM.
            assert runner.model_config.is_multimodal_model
            runner.load_model()

            assert runner.is_multimodal_model == still_mm_after_loading_model

    @patch('tpu_inference.runner.tpu_runner.set_forward_context')
    def test_execute_model_continue_decode_structured_output_guard(
            self, mock_set_forward_context):
        """_execute_model() should bypass continue decode if structured output is requested."""
        runner = MagicMock()
        runner.enable_continue_decode = True
        runner.input_batch.num_reqs = 2
        runner.input_batch.request_distribution = [2, 0, 2]  # decode only
        runner._prepare_inputs.return_value = (MagicMock(), MagicMock(),
                                               MagicMock(), MagicMock(),
                                               MagicMock(), None, None, 2, {},
                                               2, None, MagicMock())
        runner._get_input_ids_embeds.return_value = (MagicMock(), None)
        runner.is_multimodal_model = False
        runner.compute_logits_fn = MagicMock()
        runner.state_leaves = MagicMock()
        runner.mesh = MagicMock()
        runner.lora_utils.extract_lora_metadata.return_value = None
        runner.input_batch.num_prompt_logprobs = 0
        runner.model_fn.return_value = (MagicMock(), MagicMock(), None, None)
        runner._select_from_array_fn = MagicMock()
        runner.is_pooling_model = False
        runner.is_last_rank = True
        runner.speculative_config = None
        runner.execute_model_state = None

        # 1. Batch has structured output requests -> continue decode MUST be bypassed
        scheduler_output_structured = MagicMock()
        scheduler_output_structured.has_structured_output_requests = True
        scheduler_output_structured.finished_req_ids = []
        TPUModelRunner._execute_model(runner, scheduler_output_structured)
        runner._execute_continue_decode.assert_not_called()

        # 2. Batch does NOT have structured output requests -> continue decode called
        scheduler_output_normal = MagicMock()
        scheduler_output_normal.has_structured_output_requests = False
        scheduler_output_normal.finished_req_ids = []
        TPUModelRunner._execute_model(runner, scheduler_output_normal)
        runner._execute_continue_decode.assert_called_once_with(
            scheduler_output_normal)

        # 3. When async_scheduling is True and previous step was continue_decode,
        # bypassing continue_decode (e.g. structured output or mixed prefill) must
        # still call _modify_prev_results() and clear _pre_async_results before _prepare_inputs.
        runner.scheduler_config.async_scheduling = True
        runner._pre_async_results = MagicMock(is_continue_decode=True)
        call_order = []
        runner._modify_prev_results.side_effect = lambda: call_order.append(
            "modify_prev")
        runner._prepare_inputs.side_effect = lambda *a, **kw: (
            call_order.append("prepare_inputs") or
            (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
             None, None, 2, {}, 2, None, MagicMock()))
        TPUModelRunner._execute_model(runner, scheduler_output_structured)
        assert call_order == ["modify_prev", "prepare_inputs"]
        assert runner._pre_async_results is None
