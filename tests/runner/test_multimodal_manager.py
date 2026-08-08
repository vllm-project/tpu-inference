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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu
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

    def test_execute_mm_encoder_single_image(self):
        import torch
        """Tests _execute_mm_encoder with a single request and a single image."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        self.mock_get_mm_embed_fn = MagicMock()
        self.runner.embed_multimodal_fn = self.mock_get_mm_embed_fn

        self.runner.state_leaves = MagicMock()
        # Mock scheduler output
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.scheduled_encoder_inputs = {"req-1": [0]}

        # Mock request state
        dummy_pixel_values = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        dummy_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        mm_item = MultiModalKwargsItem({
            "pixel_values":
            MultiModalFieldElem(dummy_pixel_values, MultiModalBatchedField()),
            "image_grid_thw":
            MultiModalFieldElem(dummy_grid_thw, MultiModalBatchedField())
        })

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

        # Positional args: (state_leaves,)
        state_arg, = call_args.args
        # Keyword args: **batched_mm_inputs
        kwargs_arg = call_args.kwargs

        assert state_arg == self.runner.state_leaves
        assert "image_grid_thw" in kwargs_arg
        assert "pixel_values" in kwargs_arg

        # Verify the pixel values tensor passed to the mock
        passed_pixel_values = kwargs_arg['pixel_values']
        assert isinstance(passed_pixel_values, torch.Tensor)
        assert passed_pixel_values.shape == (1, 3, 224, 224)
        assert torch.equal(passed_pixel_values[0], dummy_pixel_values)

    def test_execute_mm_encoder_multiple_images(self):
        import torch
        """Tests _execute_mm_encoder with multiple requests and images."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        self.mock_get_mm_embed_fn = MagicMock()
        self.runner.embed_multimodal_fn = self.mock_get_mm_embed_fn

        self.runner.state_leaves = MagicMock()
        # Mock scheduler output for two requests
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.scheduled_encoder_inputs = {
            "req-1": [0],
            "req-2": [0]
        }

        # Mock request states
        px_1 = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        grid_1 = torch.tensor([[1, 1, 1]], dtype=torch.int64)

        mm_item_1 = MultiModalKwargsItem({
            "pixel_values":
            MultiModalFieldElem(px_1, MultiModalBatchedField()),
            "image_grid_thw":
            MultiModalFieldElem(grid_1, MultiModalBatchedField())
        })

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
        mm_item_2 = MultiModalKwargsItem({
            "pixel_values":
            MultiModalFieldElem(px_2, MultiModalBatchedField()),
            "image_grid_thw":
            MultiModalFieldElem(grid_2, MultiModalBatchedField())
        })

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

        state_arg, = call_args.args
        kwargs_arg = call_args.kwargs

        assert state_arg == self.runner.state_leaves
        assert "image_grid_thw" in kwargs_arg
        assert "pixel_values" in kwargs_arg

        passed_pixel_values = kwargs_arg['pixel_values']
        assert isinstance(passed_pixel_values, torch.Tensor)
        assert passed_pixel_values.shape == (2, 3, 224, 224)
        assert torch.equal(passed_pixel_values[0], px_1)
        assert torch.equal(passed_pixel_values[1], px_2)

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
        prompt_token_len = 100
        mm_position = PlaceholderRange(offset=10, length=56)
        is_mm_embed_cpu = np.zeros(prompt_token_len, dtype=np.bool_)
        is_mm_embed_cpu[mm_position.offset:mm_position.offset +
                        mm_position.length] = True
        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(prompt_token_len)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=([], ),
            num_computed_tokens=0,  # This will be updated per step
            mm_features=[
                MultiModalFeatureSpec(data=None,
                                      identifier=req_id,
                                      modality="image",
                                      mm_position=mm_position)
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
        mock_scheduler_output_1.total_num_scheduled_tokens = 20
        mock_scheduler_output_1.num_scheduled_tokens = {req_id: 20}

        gathered_mm_embeds_1, gathered_is_mm_embed_1 = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output_1,
            target_pad_len=mock_scheduler_output_1.total_num_scheduled_tokens,
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=mock_scheduler_output_1.
            total_num_scheduled_tokens)

        assert gathered_mm_embeds_1 is not None
        assert isinstance(gathered_mm_embeds_1, list)
        assert len(gathered_mm_embeds_1) == 1
        assert gathered_is_mm_embed_1 is not None

        expected_embeds_1 = encoder_embedding[0:10]
        gathered_embeds_1 = gathered_mm_embeds_1[0]

        assert gathered_embeds_1.shape == expected_embeds_1.shape
        np.testing.assert_array_equal(np.asarray(gathered_embeds_1),
                                      np.asarray(expected_embeds_1))
        assert gathered_is_mm_embed_1.shape == (20, )
        np.testing.assert_array_equal(np.asarray(gathered_is_mm_embed_1),
                                      is_mm_embed_cpu[:20])

        # ----- Step 2: Middle chunk of prefill -----
        req_state.num_computed_tokens = 20
        mock_scheduler_output_2 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_2.total_num_scheduled_tokens = 30
        mock_scheduler_output_2.num_scheduled_tokens = {req_id: 30}

        gathered_mm_embeds_2, gathered_is_mm_embed_2 = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output_2,
            target_pad_len=mock_scheduler_output_2.total_num_scheduled_tokens,
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=mock_scheduler_output_2.
            total_num_scheduled_tokens)

        assert gathered_mm_embeds_2 is not None
        assert isinstance(gathered_mm_embeds_2, list)
        assert len(gathered_mm_embeds_2) == 1
        assert gathered_is_mm_embed_2 is not None

        expected_embeds_2 = encoder_embedding[10:40]
        gathered_embeds_2 = gathered_mm_embeds_2[0]

        assert gathered_embeds_2.shape == expected_embeds_2.shape
        np.testing.assert_array_equal(np.asarray(gathered_embeds_2),
                                      np.asarray(expected_embeds_2))
        assert gathered_is_mm_embed_2.shape == (30, )
        np.testing.assert_array_equal(np.asarray(gathered_is_mm_embed_2),
                                      is_mm_embed_cpu[20:50])

        # ----- Step 3: Last chunk of prefill -----
        req_state.num_computed_tokens = 50
        mock_scheduler_output_3 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_3.total_num_scheduled_tokens = 30
        mock_scheduler_output_3.num_scheduled_tokens = {req_id: 30}

        gathered_mm_embeds_3, gathered_is_mm_embed_3 = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output_3,
            target_pad_len=mock_scheduler_output_3.total_num_scheduled_tokens,
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=mock_scheduler_output_3.
            total_num_scheduled_tokens)

        assert gathered_mm_embeds_3 is not None
        assert isinstance(gathered_mm_embeds_3, list)
        assert len(gathered_mm_embeds_3) == 1
        assert gathered_is_mm_embed_3 is not None

        expected_embeds_3 = encoder_embedding[40:56]
        gathered_embeds_3 = gathered_mm_embeds_3[0]

        assert gathered_embeds_3.shape == expected_embeds_3.shape
        np.testing.assert_array_equal(np.asarray(gathered_embeds_3),
                                      np.asarray(expected_embeds_3))
        assert gathered_is_mm_embed_3.shape == (30, )
        np.testing.assert_array_equal(np.asarray(gathered_is_mm_embed_3),
                                      is_mm_embed_cpu[50:80])

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
            self.runner.mm_manager.calc_mrope_positions(
                mock_scheduler_output,
                req_ids_dp={0: [req_id]},
                padded_num_scheduled_tokens_per_dp_rank=num_scheduled)

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

    # The test is slow on v6e, causing timeouts in presubmit. See b/513860288.
    @pytest.mark.skipif(not jtu.is_device_tpu_at_least(version=7),
                        reason="Expect TPUv7+")
    def test_gather_mm_embeddings_dp_aware(self):
        """Verifies that with dp_size>1, mm tokens for each request land in
        that request's rank slot of is_mm_embed (offset = rank * padded_per_rank),
        and that mm_embeds is emitted in (rank, then req-within-rank) order so
        a downstream cumsum-based gather aligns."""
        self.runner.is_multimodal_model = True

        # Two requests, one per DP rank, each with its own image embedding.
        # Image placeholder slot is offset=5, length=10 within each request.
        req_id_a, req_id_b = "req-a", "req-b"
        emb_a = jnp.arange(10 * 128, dtype=jnp.bfloat16).reshape((10, 128))
        emb_b = (jnp.arange(10 * 128, dtype=jnp.bfloat16) + 1000.0).reshape(
            (10, 128))
        self.runner.encoder_cache = {req_id_a: emb_a, req_id_b: emb_b}

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

        def _make_req(req_id):
            return CachedRequestState(
                req_id=req_id,
                prompt_token_ids=list(range(20)),
                output_token_ids=[],
                sampling_params=mock_sampling_params,
                block_ids=([], ),
                num_computed_tokens=0,
                mm_features=[
                    MultiModalFeatureSpec(
                        data=None,
                        identifier=req_id,
                        modality="image",
                        mm_position=PlaceholderRange(offset=5, length=10),
                    )
                ],
                lora_request=None,
                pooling_params=None,
                generator=None,
            )

        req_a, req_b = _make_req(req_id_a), _make_req(req_id_b)
        self.runner.requests = {req_id_a: req_a, req_id_b: req_b}
        self.runner.input_batch.add_request(req_a)
        self.runner.input_batch.add_request(req_b)

        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.num_scheduled_tokens = {
            req_id_a: 20,
            req_id_b: 20,
        }

        # padded_per_rank=24 (>20); target_pad_len=48 (= 24 * 2).
        padded_per_rank = 24
        target_pad_len = padded_per_rank * 2
        req_ids_dp = {0: [req_id_a], 1: [req_id_b]}

        mm_embeds, is_mm_embed = self.runner.mm_manager.gather_mm_embeddings(
            mock_scheduler_output,
            target_pad_len=target_pad_len,
            req_ids_dp=req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank=padded_per_rank,
        )

        # Order: rank 0's embed first, rank 1's second.
        assert mm_embeds is not None and len(mm_embeds) == 2
        np.testing.assert_array_equal(np.asarray(mm_embeds[0]),
                                      np.asarray(emb_a))
        np.testing.assert_array_equal(np.asarray(mm_embeds[1]),
                                      np.asarray(emb_b))

        # Within each rank's slot, True bits sit at [5, 15). Padding slots
        # ([20, 24) within each rank) stay False.
        expected = np.zeros(target_pad_len, dtype=np.bool_)
        expected[5:15] = True
        expected[padded_per_rank + 5:padded_per_rank + 15] = True
        np.testing.assert_array_equal(np.asarray(is_mm_embed), expected)
        # Sanity: the cumsum-based downstream gather requires the kth True
        # bit globally to correspond to mm_embeds[k]; this layout satisfies it
        # because rank 0's True bits precede rank 1's.
        assert is_mm_embed.shape == (target_pad_len, )

    # The test is slow on v6e, causing timeouts in presubmit. See b/513860288.
    @pytest.mark.skipif(not jtu.is_device_tpu_at_least(version=7),
                        reason="Expect TPUv7+")
    def test_calc_mrope_positions_dp_aware(self):
        """Verifies that with dp_size>1, each request's mrope_positions are
        written into its rank's slot of mrope_positions_cpu rather than packed
        sequentially into rank 0's slot."""
        self.runner.uses_mrope = True

        req_id_a, req_id_b = "req-a", "req-b"
        prompt_len = 8
        num_scheduled = 8

        # Distinguishable mrope_positions for each request.
        mrope_a = np.arange(3 * prompt_len, dtype=np.int64).reshape(
            (3, prompt_len))
        mrope_b = (np.arange(3 * prompt_len, dtype=np.int64) + 1000).reshape(
            (3, prompt_len))

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

        def _make_req(req_id, mrope):
            return CachedRequestState(
                req_id=req_id,
                prompt_token_ids=list(range(prompt_len)),
                output_token_ids=[],
                sampling_params=mock_sampling_params,
                block_ids=([], ),
                num_computed_tokens=0,
                mm_features=[],
                lora_request=None,
                pooling_params=None,
                generator=None,
                mrope_positions=mrope,
                mrope_position_delta=0,
            )

        req_a = _make_req(req_id_a, mrope_a)
        req_b = _make_req(req_id_b, mrope_b)
        self.runner.requests = {req_id_a: req_a, req_id_b: req_b}
        self.runner.input_batch.add_request(req_a)
        self.runner.input_batch.add_request(req_b)
        # Zero num_computed_tokens for both so the full prompt is scheduled.
        self.runner.input_batch.num_computed_tokens_cpu[0] = 0
        self.runner.input_batch.num_computed_tokens_cpu[1] = 0

        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.num_scheduled_tokens = {
            req_id_a: num_scheduled,
            req_id_b: num_scheduled,
        }

        padded_per_rank = 16  # > num_scheduled, leaves trailing padding slots
        req_ids_dp = {0: [req_id_a], 1: [req_id_b]}

        # Pre-fill mrope_positions_cpu with a sentinel so we can detect that
        # writes only landed in the intended slots.
        self.runner.mrope_positions_cpu[:] = -7
        self.runner.mm_manager.calc_mrope_positions(
            mock_scheduler_output,
            req_ids_dp=req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank=padded_per_rank,
        )

        # Rank 0's slot: [0, 8) holds mrope_a; [8, 16) untouched.
        np.testing.assert_array_equal(self.runner.mrope_positions_cpu[:, 0:8],
                                      mrope_a)
        assert np.all(self.runner.mrope_positions_cpu[:, 8:16] == -7)
        # Rank 1's slot: [16, 24) holds mrope_b; [24, 32) untouched.
        np.testing.assert_array_equal(
            self.runner.mrope_positions_cpu[:, 16:24], mrope_b)
        assert np.all(self.runner.mrope_positions_cpu[:, 24:32] == -7)

    # ---- Vision-encoder temporal chunking (MM_ENCODER_FRAME_CHUNK) ----

    def _set_temporal_patch_size(self, tps: int = 2):
        """Give the runner a minimal hf_config with the vision temporal patch
        size the chunking helper reads to convert frames -> temporal patches."""
        self.runner.model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(temporal_patch_size=tps))

    def _make_video_group(self, t, h, w, feat=5):
        """Build an mm_kwargs_group dict shaped like the one execute_mm_encoder
        passes for a single video item: flat pixels [t*h*w, feat], grid [1,3],
        per-temporal-patch timestamps [t]."""
        import torch
        rows = t * h * w
        pixels = torch.arange(rows * feat,
                              dtype=torch.float32).reshape(rows, feat)
        grid = torch.tensor([[t, h, w]], dtype=torch.int64)
        timestamps = torch.arange(t, dtype=torch.float32)
        return {
            "pixel_values_videos": pixels,
            "video_grid_thw": grid,
            "timestamps": timestamps,
        }, pixels, timestamps

    def test_video_encoder_chunking_splits_and_concats(self):
        """With chunking on, a large video is encoded in temporal chunks and the
        per-chunk outputs are concatenated on the token axis in order."""
        import torch
        self._set_temporal_patch_size(tps=2)
        self.runner.state_leaves = MagicMock()

        t, h, w = 8, 2, 2  # rows_per_temporal_patch = h*w = 4
        group, pixels, timestamps = self._make_video_group(t, h, w)

        H_out = 6  # visual_dim + deepstack levels, packed on hidden axis
        calls = []

        def fake_embed(state_leaves, modality, **kwargs):
            assert state_leaves is self.runner.state_leaves
            assert modality == "video"
            calls.append(kwargs)
            idx = len(calls) - 1
            ct = int(kwargs["video_grid_thw"].tolist()[0][0])
            # one item -> length-1 tuple; mark rows with the chunk index.
            return (jnp.full((ct, H_out), float(idx), dtype=jnp.float32), )

        self.runner.embed_multimodal_fn = fake_embed

        # frame_chunk=4, tps=2 -> chunk_t=2 temporal patches -> 4 chunks.
        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "4"}):
            out = self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "video", 1, group)

        # 4 chunks of 2 temporal patches each.
        assert len(calls) == 4
        rows_per_t = h * w
        for i, kw in enumerate(calls):
            g = kw["video_grid_thw"].tolist()[0]
            assert g == [2, h, w]  # temporal count reduced to the chunk size
            # pixels sliced to this chunk's rows, matching the original slab.
            expected_pixels = pixels[i * 2 * rows_per_t:(i + 1) * 2 *
                                     rows_per_t]
            assert torch.equal(kw["pixel_values_videos"], expected_pixels)
            # timestamps sliced to the chunk's temporal extent.
            assert torch.equal(kw["timestamps"], timestamps[i * 2:(i + 1) * 2])

        # Output: single item, concatenation of the 4 chunk outputs (8 tokens).
        assert isinstance(out, list) and len(out) == 1
        result = np.asarray(out[0])
        assert result.shape == (t, H_out)
        expected = np.concatenate(
            [np.full((2, H_out), float(i)) for i in range(4)], axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_video_encoder_no_chunk_when_disabled(self):
        """MM_ENCODER_FRAME_CHUNK=0 -> single un-chunked call, passthrough."""
        self._set_temporal_patch_size(tps=2)
        self.runner.state_leaves = MagicMock()
        group, _, _ = self._make_video_group(8, 2, 2)

        sentinel = (jnp.ones((32, 6), dtype=jnp.float32), )
        mock = MagicMock(return_value=sentinel)
        self.runner.embed_multimodal_fn = mock

        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "0"}):
            out = self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "video", 1, group)

        mock.assert_called_once()
        assert out is sentinel  # returned as-is, not re-wrapped

    def test_video_encoder_no_chunk_when_fits_one_chunk(self):
        """A video smaller than one chunk is encoded in a single call."""
        self._set_temporal_patch_size(tps=2)
        self.runner.state_leaves = MagicMock()
        group, _, _ = self._make_video_group(4, 2, 2)  # t=4 temporal patches

        mock = MagicMock(return_value=(jnp.ones((16, 6), dtype=jnp.float32), ))
        self.runner.embed_multimodal_fn = mock

        # chunk_t = 100//2 = 50 >= t=4 -> no split.
        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "100"}):
            self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "video", 1, group)

        mock.assert_called_once()

    def test_video_encoder_no_chunk_when_pruning_enabled(self):
        """EVS video-token pruning selects tokens across the whole video, so
        chunking is skipped to stay lossless."""
        self._set_temporal_patch_size(tps=2)
        self.runner.model_config.multimodal_config = SimpleNamespace(
            video_pruning_rate=0.5)
        self.runner.state_leaves = MagicMock()
        group, _, _ = self._make_video_group(8, 2, 2)

        mock = MagicMock(return_value=(jnp.ones((16, 6), dtype=jnp.float32), ))
        self.runner.embed_multimodal_fn = mock

        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "4"}):
            self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "video", 1, group)

        mock.assert_called_once()

    def test_video_encoder_no_chunk_for_image_modality(self):
        """Chunking only applies to video; images are never split."""
        import torch
        self._set_temporal_patch_size(tps=2)
        self.runner.state_leaves = MagicMock()
        group = {
            "pixel_values":
            torch.arange(32 * 5, dtype=torch.float32).reshape(32, 5),
            "image_grid_thw":
            torch.tensor([[8, 2, 2]], dtype=torch.int64),
        }

        mock = MagicMock(return_value=(jnp.ones((32, 6), dtype=jnp.float32), ))
        self.runner.embed_multimodal_fn = mock

        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "4"}):
            self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "image", 1, group)

        mock.assert_called_once()

    def test_video_encoder_no_chunk_multiple_items(self):
        """A batched multi-item group is not chunked (only single video item)."""
        self._set_temporal_patch_size(tps=2)
        self.runner.state_leaves = MagicMock()
        group, _, _ = self._make_video_group(8, 2, 2)

        mock = MagicMock(return_value=(jnp.ones((16, 6), dtype=jnp.float32),
                                       jnp.ones((16, 6), dtype=jnp.float32)))
        self.runner.embed_multimodal_fn = mock

        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "4"}):
            self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "video", 2, group)

        mock.assert_called_once()

    def test_video_encoder_chunking_ragged_last_chunk(self):
        """When t is not a multiple of chunk_t, the last chunk is smaller and
        the concatenated length still matches the full video."""
        import torch
        self._set_temporal_patch_size(tps=2)
        self.runner.state_leaves = MagicMock()

        t, h, w = 7, 1, 3  # 7 temporal patches, chunk_t=2 -> [2,2,2,1]
        group, pixels, timestamps = self._make_video_group(t, h, w)
        H_out = 4
        calls = []

        def fake_embed(state_leaves, modality, **kwargs):
            calls.append(kwargs)
            ct = int(kwargs["video_grid_thw"].tolist()[0][0])
            return (jnp.full((ct, H_out), float(len(calls)),
                             dtype=jnp.float32), )

        self.runner.embed_multimodal_fn = fake_embed

        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK": "4"}):
            out = self.runner.mm_manager._embed_multimodal_maybe_chunked(
                "video", 1, group)

        cts = [int(kw["video_grid_thw"].tolist()[0][0]) for kw in calls]
        assert cts == [2, 2, 2, 1]
        # last chunk pixels/timestamps cover the ragged tail.
        rows_per_t = h * w
        assert torch.equal(calls[-1]["pixel_values_videos"],
                           pixels[6 * rows_per_t:7 * rows_per_t])
        assert torch.equal(calls[-1]["timestamps"], timestamps[6:7])
        assert np.asarray(out[0]).shape == (t, H_out)

    def test_slice_timestamps_variants(self):
        """_slice_timestamps handles 1D/2D tensors, lists, nested lists and
        passes through non-sliceable values."""
        import torch
        slc = self.runner.mm_manager._slice_timestamps

        # 1D tensor
        r = slc(torch.arange(10), 2, 5)
        assert torch.equal(r, torch.tensor([2, 3, 4]))
        # 2D [1, T] tensor -> slice last axis
        r = slc(torch.arange(10).reshape(1, 10), 2, 5)
        assert r.shape == (1, 3)
        assert torch.equal(r, torch.tensor([[2, 3, 4]]))
        # flat list
        assert slc(list(range(10)), 2, 5) == [2, 3, 4]
        # nested single-item list
        assert slc([list(range(10))], 2, 5) == [[2, 3, 4]]
        # non-sliceable -> passthrough
        assert slc(None, 2, 5) is None
