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
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal.inputs import (MultiModalBatchedField,
                                    MultiModalFeatureSpec, MultiModalFieldElem,
                                    MultiModalKwargsItem, PlaceholderRange)
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from tpu_inference.runner.input_batch import CachedRequestState
from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestMultiModalManager:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        self.mock_rng_key = MagicMock()

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
                swap_space=4,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                worker_use_ray=False,
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

    def test_execute_mm_encoder_single_image(self):
        import torch
        """Tests _execute_mm_encoder with a single request and a single image."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        self.mock_get_mm_embed_fn = MagicMock()
        self.runner.embed_multimodal_fn = self.mock_get_mm_embed_fn

        self.runner.state = MagicMock()
        # Mock scheduler output
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.scheduled_encoder_inputs = {"req-1": [0]}

        # Mock request state
        dummy_pixel_values = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        dummy_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        mm_item = MultiModalKwargsItem.from_elems([
            MultiModalFieldElem("image", "pixel_values", dummy_pixel_values,
                                MultiModalBatchedField()),
            MultiModalFieldElem("image", "image_grid_thw", dummy_grid_thw,
                                MultiModalBatchedField())
        ])

        req_state = CachedRequestState(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[],
            sampling_params=MagicMock(),
            block_ids=(),
            num_computed_tokens=0,
            mm_features=[
                MultiModalFeatureSpec(data=mm_item,
                                      identifier="req-1",
                                      modality="image",
                                      mm_position=PlaceholderRange(offset=0,
                                                                   length=1))
            ],
            lora_request=None,
            pooling_params=None,
            generator=None,
        )
        self.runner.requests = {"req-1": req_state}

        # Mock the return value of the multimodal encoder
        dummy_embedding = jnp.ones((10, 128), dtype=jnp.bfloat16)
        self.mock_get_mm_embed_fn.return_value = (dummy_embedding, )

        # 2. ===== Act =====
        self.runner.mm_manager.execute_mm_encoder(mock_scheduler_output)

        # 3. ===== Assert =====
        # Check if encoder_cache is populated correctly
        assert "req-1" in self.runner.encoder_cache
        cached_embedding = self.runner.encoder_cache["req-1"]
        np.testing.assert_array_equal(np.asarray(cached_embedding),
                                      np.asarray(dummy_embedding))

        # Check if embed_multimodal_fn was called with correct args
        self.mock_get_mm_embed_fn.assert_called_once()
        call_args = self.mock_get_mm_embed_fn.call_args

        # Positional args: (state, image_grid_thw)
        state_arg, grid_arg = call_args.args
        # Keyword args: **batched_mm_inputs
        kwargs_arg = call_args.kwargs

        assert state_arg == self.runner.state
        assert grid_arg == ((1, 1, 1), )
        assert "pixel_values" in kwargs_arg

        # Verify the pixel values tensor passed to the mock
        passed_pixel_values = kwargs_arg['pixel_values']
        assert isinstance(passed_pixel_values, np.ndarray)
        assert passed_pixel_values.dtype == jnp.bfloat16

        # Convert torch tensor for comparison
        expected_pixel_values = dummy_pixel_values.unsqueeze(0).to(
            torch.float32).numpy().astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(passed_pixel_values),
                                      expected_pixel_values)

    def test_execute_mm_encoder_multiple_images(self):
        import torch
        """Tests _execute_mm_encoder with multiple requests and images."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        self.mock_get_mm_embed_fn = MagicMock()
        self.runner.embed_multimodal_fn = self.mock_get_mm_embed_fn

        self.runner.state = MagicMock()
        # Mock scheduler output for two requests
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.scheduled_encoder_inputs = {
            "req-1": [0],
            "req-2": [0]
        }

        # Mock request states
        px_1 = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        grid_1 = torch.tensor([[1, 1, 1]], dtype=torch.int64)

        mm_item_1 = MultiModalKwargsItem.from_elems([
            MultiModalFieldElem("image", "pixel_values", px_1,
                                MultiModalBatchedField()),
            MultiModalFieldElem("image", "image_grid_thw", grid_1,
                                MultiModalBatchedField())
        ])

        req_state_1 = CachedRequestState(
            req_id="req-1",
            prompt_token_ids=[],
            output_token_ids=[],
            sampling_params=MagicMock(),
            block_ids=(),
            num_computed_tokens=0,
            mm_features=[
                MultiModalFeatureSpec(data=mm_item_1,
                                      identifier="req-1",
                                      modality="image",
                                      mm_position=PlaceholderRange(offset=0,
                                                                   length=1))
            ],
            lora_request=None,
            pooling_params=None,
            generator=None)

        px_2 = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        grid_2 = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        mm_item_2 = MultiModalKwargsItem.from_elems([
            MultiModalFieldElem("image", "pixel_values", px_2,
                                MultiModalBatchedField()),
            MultiModalFieldElem("image", "image_grid_thw", grid_2,
                                MultiModalBatchedField())
        ])

        req_state_2 = CachedRequestState(
            req_id="req-2",
            prompt_token_ids=[],
            output_token_ids=[],
            sampling_params=MagicMock(),
            block_ids=(),
            num_computed_tokens=0,
            mm_features=[
                MultiModalFeatureSpec(data=mm_item_2,
                                      identifier="req-2",
                                      modality="image",
                                      mm_position=PlaceholderRange(offset=0,
                                                                   length=1))
            ],
            lora_request=None,
            pooling_params=None,
            generator=None)

        self.runner.requests = {"req-1": req_state_1, "req-2": req_state_2}

        emb_1 = jnp.ones((10, 128), dtype=jnp.bfloat16)
        emb_2 = jnp.ones((20, 128), dtype=jnp.bfloat16) * 2
        self.mock_get_mm_embed_fn.return_value = (emb_1, emb_2)

        # 2. ===== Act =====
        self.runner.mm_manager.execute_mm_encoder(mock_scheduler_output)

        # 3. ===== Assert =====
        assert "req-1" in self.runner.encoder_cache
        np.testing.assert_array_equal(
            np.asarray(self.runner.encoder_cache["req-1"]), np.asarray(emb_1))
        assert "req-2" in self.runner.encoder_cache
        np.testing.assert_array_equal(
            np.asarray(self.runner.encoder_cache["req-2"]), np.asarray(emb_2))

        self.mock_get_mm_embed_fn.assert_called_once()
        call_args = self.mock_get_mm_embed_fn.call_args

        state_arg, grid_arg = call_args.args
        kwargs_arg = call_args.kwargs

        assert state_arg == self.runner.state
        assert grid_arg == ((1, 1, 1), (1, 2, 2))
        assert "pixel_values" in kwargs_arg

        passed_pixel_values = kwargs_arg['pixel_values']
        assert passed_pixel_values.shape == (2, 3, 224, 224)

        expected_pixel_values = torch.stack([px_1, px_2], dim=0).to(
            torch.float32).numpy().astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(passed_pixel_values),
                                      expected_pixel_values)

    def test_gather_mm_embeddings_chunked_prefill(self):
        """Tests _gather_mm_embeddings with chunked prefill scenarios."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        req_id = "req-1"

        # Mock encoder output
        encoder_embedding = jnp.arange(56 * 128, dtype=jnp.bfloat16).reshape(
            (56, 128))
        self.runner.encoder_cache = {req_id: encoder_embedding}

        mock_sampling_params = MagicMock()
        mock_sampling_params.sampling_type = SamplingType.GREEDY
        mock_sampling_params.top_k = -1
        mock_sampling_params.top_p = 1.0
        mock_sampling_params.temperature = 0.0
        mock_sampling_params.min_tokens = 0
        mock_sampling_params.logprobs = None
        mock_sampling_params.logit_bias = None
        mock_sampling_params.allowed_token_ids = set()
        mock_sampling_params.bad_words_token_ids = None
        mock_sampling_params.all_stop_token_ids = set()

        # Mock request state
        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(100)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=([], ),
            num_computed_tokens=0,  # This will be updated per step
            mm_features=[
                MultiModalFeatureSpec(data=None,
                                      identifier=req_id,
                                      modality="image",
                                      mm_position=PlaceholderRange(offset=10,
                                                                   length=56))
            ],
            lora_request=None,
            pooling_params=None,
            generator=None,
        )
        self.runner.requests = {req_id: req_state}
        self.runner.input_batch.add_request(req_state)

        # 2. ===== Act & Assert =====

        # ----- Step 1: First chunk of prefill -----
        req_state.num_computed_tokens = 0
        mock_scheduler_output_1 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_1.num_scheduled_tokens = {req_id: 20}

        gathered_embeds_1 = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output_1, target_pad_len=10)

        expected_embeds_1 = encoder_embedding[0:10]
        assert gathered_embeds_1.shape == expected_embeds_1.shape
        np.testing.assert_array_equal(np.asarray(gathered_embeds_1),
                                      np.asarray(expected_embeds_1))

        # ----- Step 2: Middle chunk of prefill -----
        req_state.num_computed_tokens = 20
        mock_scheduler_output_2 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_2.num_scheduled_tokens = {req_id: 30}

        gathered_embeds_2 = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output_2, target_pad_len=30)

        expected_embeds_2 = encoder_embedding[10:40]
        assert gathered_embeds_2.shape == expected_embeds_2.shape
        np.testing.assert_array_equal(np.asarray(gathered_embeds_2),
                                      np.asarray(expected_embeds_2))

        # ----- Step 3: Last chunk of prefill -----
        req_state.num_computed_tokens = 50
        mock_scheduler_output_3 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_3.num_scheduled_tokens = {req_id: 30}

        gathered_embeds_3 = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output_3, target_pad_len=16)

        expected_embeds_3 = encoder_embedding[40:56]
        assert gathered_embeds_3.shape == expected_embeds_3.shape
        np.testing.assert_array_equal(np.asarray(gathered_embeds_3),
                                      np.asarray(expected_embeds_3))

    def test_calc_mrope_positions(self):
        """Tests the calculation of M-RoPE positions for mixed prompt/completion."""
        # 1. ===== Setup =====
        self.runner.uses_mrope = True
        req_id = "req-1"
        prompt_len = 20
        num_computed = 15
        num_scheduled = 10
        mrope_delta = 100

        # Mock request state with pre-computed mrope positions for the prompt
        mock_mrope_positions = np.arange(3 * prompt_len,
                                         dtype=np.int64).reshape(
                                             3, prompt_len)
        mock_sampling_params = MagicMock()
        mock_sampling_params.sampling_type = SamplingType.GREEDY
        mock_sampling_params.top_k = -1
        mock_sampling_params.top_p = 1.0
        mock_sampling_params.temperature = 0.0
        mock_sampling_params.min_tokens = 0
        mock_sampling_params.logprobs = None
        mock_sampling_params.logit_bias = None
        mock_sampling_params.allowed_token_ids = set()
        mock_sampling_params.bad_words_token_ids = None
        mock_sampling_params.all_stop_token_ids = set()

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(prompt_len)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=([], ),
            num_computed_tokens=num_computed,
            mm_features=[],
            lora_request=None,
            pooling_params=None,
            generator=None,
            mrope_positions=mock_mrope_positions,
            mrope_position_delta=mrope_delta,
        )
        self.runner.requests = {req_id: req_state}
        self.runner.input_batch.add_request(req_state)
        # Manually set num_computed_tokens in the batch as add_request sets it to 0
        self.runner.input_batch.num_computed_tokens_cpu[0] = num_computed

        # Mock scheduler output
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.num_scheduled_tokens = {req_id: num_scheduled}

        # Patch the static method that computes completion positions
        with patch.object(MRotaryEmbedding,
                          "get_next_input_positions_tensor") as mock_get_next:
            # 2. ===== Act =====
            self.runner.mm_manager.calc_mrope_positions(mock_scheduler_output)

            # 3. ===== Assert =====
            # The first 5 positions should be copied from the pre-computed prompt positions
            expected_prompt_part = mock_mrope_positions[:, 15:20]
            actual_prompt_part = self.runner.mrope_positions_cpu[:, 0:5]
            np.testing.assert_array_equal(actual_prompt_part,
                                          expected_prompt_part)

            # The next 5 positions should be computed on-the-fly
            mock_get_next.assert_called_once()
            call_kwargs = mock_get_next.call_args.kwargs
            np.testing.assert_array_equal(call_kwargs["out"],
                                          self.runner.mrope_positions_cpu)
            assert call_kwargs["out_offset"] == 5
            assert call_kwargs["mrope_position_delta"] == mrope_delta
            assert call_kwargs["context_len"] == prompt_len
            assert call_kwargs["num_new_tokens"] == 5
