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

from functools import partial
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import \
    Qwen2_5_VLConfig
from vllm.config import (CacheConfig, DeviceConfig, MultiModalConfig,
                         ParallelConfig, SchedulerConfig)

# Import the module itself to allow patching
# Corrected imports for the code under test
from tpu_inference.models.jax.qwen2_5_vl import (
    AttentionMetadata, Qwen2_5_VisionAttention, Qwen2_5_VisionBlock,
    Qwen2_5_VisionMLP, Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionPatchMerger,
    Qwen2_5_VisionRotaryEmbedding, Qwen2_5_VisionTransformer,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLImagePixelInputs, SegmentIds,
    apply_rotary_pos_emb_vision, generate_window_segment_ids)


# --- Configuration Mocking ---
class MockModelConfig:

    def __init__(self, hf_config, dtype):
        self.hf_config = hf_config
        self.dtype = dtype
        self.multimodal_config = MultiModalConfig(
            image_input_type="pixel",
            image_token_id=hf_config.image_token_id,
            image_input_shape=None)
        self.model = "mock_qwen2_5_vl"
        # Add other attributes if needed by the code
        self.tokenizer = "mock_tokenizer"
        self.tokenizer_mode = "auto"
        self.trust_remote_code = True
        self.seed = 0

    def is_multimodal_model(self):
        return True

    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_head_size(self):
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Qwen2.5 VL model."""

    def __init__(self, tie_word_embeddings: bool = False):
        vision_config = {
            "hidden_size": 16,
            "intermediate_size": 32,
            "patch_size": 14,
            "image_size": 28,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "window_size": 28,
            "spatial_merge_size": 2,
            "fullatt_block_indexes": [0],
            "out_hidden_size": 24,
            "depth": 2,
            "hidden_act": "gelu",
            "num_heads": 2,
        }
        hf_config = Qwen2_5_VLConfig(
            vision_config=vision_config,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=32,
            rms_norm_eps=1e-6,
            image_token_id=200000,
            video_token_id=200001,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=32000,
            rope_theta=1000000.0,
        )
        self.model_config = MockModelConfig(hf_config, jnp.bfloat16)
        self.cache_config = MagicMock(spec=CacheConfig)
        self.parallelism_config = MagicMock(spec=ParallelConfig)
        self.scheduler_config = MagicMock(spec=SchedulerConfig)
        self.device_config = MagicMock(spec=DeviceConfig)
        self.load_config = MagicMock()
        self.extra_configs = {}
        self.additional_config = {}


@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with all required axes for testing."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices())
    return Mesh(devices.reshape((len(devices), 1, 1)),
                axis_names=('data', 'attn_dp', 'model'))


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_vllm_config() -> MockVllmConfig:
    return MockVllmConfig()


@pytest.fixture
def rngs(rng: PRNGKey) -> nnx.Rngs:
    return nnx.Rngs(params=rng)


# --- Test Classes ---
class TestUtils:

    def test_apply_rotary_pos_emb_vision(self, rng: PRNGKey):
        B, T, N, H = 1, 10, 2, 8
        x = jax.random.normal(rng, (B, T, N, H))
        rotary_pos_emb = jax.random.normal(rng, (T, H // 2))
        x_rotated = apply_rotary_pos_emb_vision(x, rotary_pos_emb)
        assert x_rotated.shape == (B, T, N, H)

    def test_generate_window_segment_ids(self):
        cu_seqlens = jnp.array([0, 5, 10])
        seq_len = 10
        padded_seq_len = 16
        segment_ids = generate_window_segment_ids(cu_seqlens, seq_len,
                                                  padded_seq_len)
        assert isinstance(segment_ids, SegmentIds)
        assert segment_ids.q.shape == (1, padded_seq_len)
        assert segment_ids.kv.shape == (1, padded_seq_len)
        expected_q = np.array(
            [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(segment_ids.q, expected_q)
        np.testing.assert_array_equal(segment_ids.kv, expected_q)


class TestQwen2_5_VisionMLP:

    def test_forward(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs):
        config = mock_vllm_config.model_config.hf_config.vision_config
        dtype = mock_vllm_config.model_config.dtype
        mlp = Qwen2_5_VisionMLP(config, dtype, rngs)
        x = jnp.ones((5, config.hidden_size), dtype=dtype)
        y = mlp(x)
        assert y.shape == (5, config.hidden_size)
        assert y.dtype == dtype


class TestQwen2_5_VisionAttention:

    @patch('tpu_inference.models.jax.qwen2_5_vl.sharded_flash_attention')
    def test_forward_fullattn(self, mock_flash_attention: MagicMock,
                              mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                              mesh: Mesh, rng: PRNGKey):
        attn_module = Qwen2_5_VisionAttention(
            mock_vllm_config.model_config.hf_config,
            mock_vllm_config.model_config.dtype, rngs, mesh)
        B, T, D = 1, 10, attn_module.hidden_size
        # sharded_flash_attention is a factory, so we mock the returned function
        mock_attn_fn = MagicMock(return_value=jnp.ones((B,
                                                        attn_module.num_heads,
                                                        128,
                                                        attn_module.head_dim)))
        attn_module.flash_attention = mock_attn_fn
        x = jax.random.normal(rng, (T, B, D))
        rotary_pos_emb = jax.random.normal(rng, (T, attn_module.head_dim // 2))
        cu_seqlens = jnp.array([0, 5])

        y_full = attn_module(x,
                             rotary_pos_emb,
                             cu_window_seqlens=cu_seqlens,
                             use_fullattn=True)
        assert y_full.shape == (T, B, D)
        mock_attn_fn.assert_called_once()
        assert mock_attn_fn.call_args[0][3].q.shape == (1, 128)

    @patch('tpu_inference.models.jax.qwen2_5_vl.sharded_flash_attention')
    def test_forward_windowed(self, mock_flash_attention: MagicMock,
                              mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                              mesh: Mesh, rng: PRNGKey):
        attn_module = Qwen2_5_VisionAttention(
            mock_vllm_config.model_config.hf_config,
            mock_vllm_config.model_config.dtype, rngs, mesh)
        B, T, D = 1, 10, attn_module.hidden_size
        mock_attn_fn = MagicMock(return_value=jnp.ones((B,
                                                        attn_module.num_heads,
                                                        128,
                                                        attn_module.head_dim)))
        attn_module.flash_attention = mock_attn_fn
        x = jax.random.normal(rng, (T, B, D))
        rotary_pos_emb = jax.random.normal(rng, (T, attn_module.head_dim // 2))
        cu_window_seqlens = jnp.array([0, 5, 10])

        y_window = attn_module(x,
                               rotary_pos_emb,
                               cu_window_seqlens=cu_window_seqlens,
                               use_fullattn=False)
        assert y_window.shape == (T, B, D)
        mock_attn_fn.assert_called_once()
        assert mock_attn_fn.call_args[0][3].q.shape == (1, 128)

    def test_batch_fail(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                        mesh: Mesh, rng: PRNGKey):
        attn_module = Qwen2_5_VisionAttention(
            mock_vllm_config.model_config.hf_config,
            mock_vllm_config.model_config.dtype, rngs, mesh)
        T, B, D = 10, 2, attn_module.hidden_size
        x = jax.random.normal(rng, (T, B, D))
        rotary_pos_emb = jax.random.normal(rng, (T, attn_module.head_dim // 2))
        with pytest.raises(
                AssertionError,
                match="Vision attention currently only supports batch size 1"):
            attn_module(x, rotary_pos_emb, use_fullattn=True)


class TestQwen2_5_VisionBlock:

    @patch('tpu_inference.models.jax.qwen2_5_vl.Qwen2_5_VisionMLP',
           autospec=True)
    @patch('tpu_inference.models.jax.qwen2_5_vl.Qwen2_5_VisionAttention',
           autospec=True)
    def test_forward(self, MockAttention: MagicMock, MockMLP: MagicMock,
                     mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                     mesh: Mesh, rng: PRNGKey):
        config = mock_vllm_config.model_config.hf_config
        dtype = mock_vllm_config.model_config.dtype
        D = config.vision_config.hidden_size
        T, B = 10, 1

        mock_attn_instance = MockAttention.return_value
        mock_attn_instance.return_value = jnp.zeros((T, B, D), dtype=dtype)
        mock_mlp_instance = MockMLP.return_value
        mock_mlp_instance.return_value = jnp.zeros((T, B, D), dtype=dtype)

        block = Qwen2_5_VisionBlock(config, dtype, rngs, mesh)
        x = jax.random.normal(rng, (T, B, D))
        rotary_pos_emb = jax.random.normal(
            rng, (T, config.vision_config.hidden_size //
                  config.vision_config.num_heads // 2))

        y = block(x, rotary_pos_emb, use_fullattn=True)
        assert y.shape == (T, B, D)
        mock_attn_instance.assert_called_once()
        mock_mlp_instance.assert_called_once()


class TestQwen2_5_VisionPatchEmbed:

    def test_forward(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                     rng: PRNGKey):
        vc = mock_vllm_config.model_config.hf_config.vision_config
        dtype = mock_vllm_config.model_config.dtype
        patch_embed = Qwen2_5_VisionPatchEmbed(
            rngs,
            patch_size=vc.patch_size,
            temporal_patch_size=vc.temporal_patch_size,
            in_channels=vc.in_channels,
            hidden_size=vc.hidden_size,
            dtype=dtype)
        num_patches = 4
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        x = jax.random.normal(rng, (num_patches, patch_dim))
        y = patch_embed(x)
        assert y.shape == (num_patches, vc.hidden_size)


class TestQwen2_5_VisionPatchMerger:

    def test_forward(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                     rng: PRNGKey):
        vc = mock_vllm_config.model_config.hf_config.vision_config
        dtype = mock_vllm_config.model_config.dtype
        merger = Qwen2_5_VisionPatchMerger(
            d_model=vc.out_hidden_size,
            context_dim=vc.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=1e-6),
            spatial_merge_size=vc.spatial_merge_size,
            dtype=dtype,
            rngs=rngs)
        x = jax.random.normal(rng,
                              (5, vc.spatial_merge_size**2, vc.hidden_size))
        y = merger(x)
        assert y.shape == (5, vc.out_hidden_size)


class TestQwen2_5_VisionRotaryEmbedding:

    def test_forward(self):
        dim = 16
        seqlen = 10
        rotary_emb = Qwen2_5_VisionRotaryEmbedding(dim=dim)
        emb = rotary_emb(seqlen)
        assert emb.shape == (seqlen, dim // 2)


class TestQwen2_5_VisionTransformer:

    @pytest.fixture
    def vision_transformer(self, mock_vllm_config: MockVllmConfig,
                           rngs: nnx.Rngs, mesh: Mesh):
        return Qwen2_5_VisionTransformer(mock_vllm_config, rngs, mesh)

    def test_rotary_pos_emb_thw(self,
                                vision_transformer: Qwen2_5_VisionTransformer):
        t, h, w = 2, 4, 4
        emb = vision_transformer.rotary_pos_emb_thw(t, h, w)
        vc = vision_transformer.config
        sm = vc.spatial_merge_size
        head_dim_half = (vc.hidden_size // vc.num_heads) // 2
        expected_shape = (t * (h // sm) * (w // sm), sm * sm, head_dim_half)
        assert emb.shape == expected_shape

    def test_get_window_index_thw(
            self, vision_transformer: Qwen2_5_VisionTransformer):
        grid_t, grid_h, grid_w = 1, 8, 8
        index_new, cu_seqlens_tmp = vision_transformer.get_window_index_thw(
            grid_t, grid_h, grid_w)
        vc = vision_transformer.config
        sm = vc.spatial_merge_size
        num_valid_indices = grid_t * (grid_h // sm) * (grid_w // sm)
        assert index_new.shape == (num_valid_indices, )
        assert jnp.all(index_new >= 0)

    def test_get_rope_by_thw(self,
                             vision_transformer: Qwen2_5_VisionTransformer):
        t, h, w = 1, 8, 8
        res = vision_transformer.get_rope_by_thw(t, h, w)
        assert isinstance(res, tuple)
        assert len(res) == 4
        rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw, cu_seqlens_thw = res

        vc = vision_transformer.config
        sm = vc.spatial_merge_size
        # The rotary embedding output for each position is head_dim // 2
        head_dim_rope = (vc.hidden_size // vc.num_heads) // 2
        expected_len = window_index_thw.shape[0] * sm * sm
        assert rotary_pos_emb_thw.shape == (expected_len, head_dim_rope)

    @pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
    def test_call(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
                  mesh: Mesh, rng: PRNGKey, enable_dynamic_image_sizes: bool):
        mock_vllm_config.additional_config = {
            "enable_dynamic_image_sizes": enable_dynamic_image_sizes
        }
        vision_transformer = Qwen2_5_VisionTransformer(mock_vllm_config, rngs,
                                                       mesh)
        # Mock the flash_attention call to avoid sharding errors in test environment
        for block in vision_transformer.blocks:
            # The mock should return a tensor of the same shape as the query 'q'
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        t_pix, h_pix, w_pix = 2, 84, 28

        # The number of patches is calculated from the pixel dimensions of the image/video
        num_patches = (t_pix // vc.temporal_patch_size) * \
                      (h_pix // vc.patch_size) * \
                      (w_pix // vc.patch_size)

        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        x = jax.random.normal(rng, (num_patches, patch_dim))

        # The grid_thw should be in terms of patch grid dimensions, not pixels
        t_grid = t_pix // vc.temporal_patch_size
        h_grid = h_pix // vc.patch_size
        w_grid = w_pix // vc.patch_size
        grid_thw = ((t_grid, h_grid, w_grid), )

        embeddings = vision_transformer(x, grid_thw)

        # The number of output tokens is determined by the grid dimensions and spatial merge size.
        expected_len = t_grid * (h_grid // vc.spatial_merge_size) * (
            w_grid // vc.spatial_merge_size)
        assert embeddings.shape == (expected_len, vc.out_hidden_size)


class TestQwen2_5_VLForConditionalGeneration:

    @pytest.fixture
    def model(self, mock_vllm_config: MockVllmConfig, rng: PRNGKey,
              mesh: Mesh):
        with patch('tpu_inference.models.jax.qwen2_5_vl.Qwen2_5_VisionTransformer', autospec=True) as MockVision, \
             patch('tpu_inference.models.jax.qwen2_5_vl.Qwen2ForCausalLM', autospec=True) as MockLM:
            mock_visual = MockVision.return_value
            mock_visual.dtype = mock_vllm_config.model_config.dtype
            mock_visual.config = mock_vllm_config.model_config.hf_config.vision_config
            mock_visual.spatial_merge_size = mock_vllm_config.model_config.hf_config.vision_config.spatial_merge_size

            model = Qwen2_5_VLForConditionalGeneration(mock_vllm_config, rng,
                                                       mesh)
            # Directly assign mocked instances
            model.visual = mock_visual
            model.language_model = MockLM.return_value
            yield model

    def test_validate_and_reshape_mm_tensor(
            self, model: Qwen2_5_VLForConditionalGeneration):
        data_list = [np.ones((2, 4)), np.ones((3, 4))]
        reshaped_list = model._validate_and_reshape_mm_tensor(
            data_list, "test_list")
        assert reshaped_list.shape == (5, 4)
        assert isinstance(reshaped_list, jax.Array)

        data_2d = np.ones((5, 4))
        reshaped_2d = model._validate_and_reshape_mm_tensor(data_2d, "test_2d")
        assert reshaped_2d.shape == (5, 4)

        data_3d = np.ones((2, 5, 4))
        reshaped_3d = model._validate_and_reshape_mm_tensor(data_3d, "test_3d")
        assert reshaped_3d.shape == (10, 4)

        with pytest.raises(ValueError, match="Incorrect type of test_invalid"):
            model._validate_and_reshape_mm_tensor("invalid", "test_invalid")

    def test_parse_and_validate_image_input(
            self, model: Qwen2_5_VLForConditionalGeneration):
        grid = ((2, 28, 28), )
        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        pixel_values = np.ones((4, patch_dim))

        parsed = model._parse_and_validate_image_input(
            grid, pixel_values=pixel_values)
        assert parsed is not None
        assert parsed['type'] == "pixel_values"
        assert parsed['pixel_values'].shape == (4, patch_dim)
        assert parsed['image_grid_thw'] == grid

        parsed_none = model._parse_and_validate_image_input(grid)
        assert parsed_none is None

    def test_parse_and_validate_multimodal_inputs(
            self, model: Qwen2_5_VLForConditionalGeneration):
        grid = ((2, 28, 28), )
        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        pixel_values = np.ones((4, patch_dim))

        mm_inputs = model._parse_and_validate_multimodal_inputs(
            grid, pixel_values=pixel_values)
        assert "image" in mm_inputs
        assert mm_inputs["image"]['type'] == "pixel_values"

        mm_inputs_empty = model._parse_and_validate_multimodal_inputs(grid)
        assert not mm_inputs_empty

    def test_process_image_input_pixels(
            self, model: Qwen2_5_VLForConditionalGeneration):
        grid_thw = ((2, 28, 28), (2, 28, 28))
        vc = model.config.vision_config
        num_patches = 8  # 4 per image
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        pixel_values = jnp.ones((num_patches, patch_dim))
        image_input = Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                                 pixel_values=pixel_values,
                                                 image_grid_thw=grid_thw)

        tokens_per_image = (2 * 28 * 28) // (vc.spatial_merge_size**2)
        mock_embeds = jnp.ones((tokens_per_image, vc.out_hidden_size))
        model.visual.return_value = mock_embeds

        embeddings = model._process_image_input(image_input)
        assert isinstance(embeddings, tuple)
        assert len(embeddings) == 2
        assert embeddings[0].shape == (tokens_per_image, vc.out_hidden_size)
        assert embeddings[1].shape == (tokens_per_image, vc.out_hidden_size)
        assert model.visual.call_count == 2

    def test_embed_multimodal(self, model: Qwen2_5_VLForConditionalGeneration):
        grid_thw = ((2, 28, 28), )
        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        pixel_values = np.ones((4, patch_dim))
        tokens_per_image = (2 * 28 * 28) // (vc.spatial_merge_size**2)
        mock_vision_output = jnp.ones((tokens_per_image, vc.out_hidden_size))

        with patch.object(model,
                          '_process_image_input',
                          return_value=(mock_vision_output, )) as mock_process:
            mm_embeds = model.embed_multimodal(grid_thw,
                                               pixel_values=pixel_values)
            mock_process.assert_called_once()
            assert isinstance(mm_embeds, tuple)
            assert len(mm_embeds) == 1
            assert mm_embeds[0].shape == (tokens_per_image, vc.out_hidden_size)

        mm_embeds_none = model.embed_multimodal(grid_thw)
        assert len(mm_embeds_none) == 0

    @patch('tpu_inference.models.jax.qwen2_5_vl.merge_multimodal_embeddings')
    def test_embed_input_ids(self, mock_merge_embeddings: MagicMock,
                             model: Qwen2_5_VLForConditionalGeneration,
                             rng: PRNGKey):
        input_ids = jax.random.randint(rng, (1, 10), 0,
                                       model.config.vocab_size)
        mock_text_embeds = jnp.ones((1, 10, model.config.hidden_size))
        model.language_model.model = MagicMock()
        model.language_model.model.embed = MagicMock(
            return_value=mock_text_embeds)

        embeds = model.embed_input_ids(input_ids, None)
        np.testing.assert_array_equal(embeds, mock_text_embeds)
        mock_merge_embeddings.assert_not_called()

        empty_mm = jnp.ones((0, model.config.hidden_size), )
        embeds_empty_mm = model.embed_input_ids(input_ids, empty_mm)
        np.testing.assert_array_equal(embeds_empty_mm, mock_text_embeds)
        mock_merge_embeddings.assert_not_called()

        mm_embeds = jnp.ones((5, model.config.hidden_size))
        mock_merged = jnp.ones((1, 15, model.config.hidden_size))
        mock_merge_embeddings.return_value = mock_merged

        embeds_mm = model.embed_input_ids(input_ids, mm_embeds)
        np.testing.assert_array_equal(embeds_mm, mock_merged)
        mock_merge_embeddings.assert_called_once_with(
            input_ids, mock_text_embeds, mm_embeds,
            [model.config.image_token_id, model.config.video_token_id])

    def test_call(self, model: Qwen2_5_VLForConditionalGeneration,
                  rng: PRNGKey):
        kv_caches = [MagicMock()]
        input_ids = jax.random.randint(rng, (1, 10), 0,
                                       model.config.vocab_size)
        attn_meta = MagicMock(spec=AttentionMetadata)
        mock_lm_output = ([MagicMock()],
                          jnp.ones((1, 10, model.config.hidden_size)), [])
        model.language_model.return_value = mock_lm_output

        new_kvs, x, aux_hidden_states = model(kv_caches, input_ids, attn_meta)
        model.language_model.assert_called_once_with(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attn_meta,
            inputs_embeds=None)
        assert len(new_kvs) == 1
        assert x.shape == (1, 10, model.config.hidden_size)
        assert len(aux_hidden_states) == 0

    def test_compute_logits(self, model: Qwen2_5_VLForConditionalGeneration,
                            rng: PRNGKey):
        hidden_states = jnp.ones((1, 10, model.config.hidden_size))
        mock_logits = jnp.ones((1, 10, model.config.vocab_size))
        model.language_model.compute_logits.return_value = mock_logits

        logits = model.compute_logits(hidden_states)
        np.testing.assert_array_equal(logits, mock_logits)
        model.language_model.compute_logits.assert_called_once_with(
            hidden_states)

    @patch('tpu_inference.models.jax.qwen2_5_vl.load_hf_weights')
    def test_load_weights(self, mock_load_weights: MagicMock,
                          model: Qwen2_5_VLForConditionalGeneration,
                          mock_vllm_config: MockVllmConfig, rng: PRNGKey,
                          mesh: Mesh):
        model.load_weights(rng)
        mock_load_weights.assert_called_once()
        kwargs = mock_load_weights.call_args.kwargs
        assert kwargs['vllm_config'] == mock_vllm_config
        assert kwargs['model'] is model
        assert "model.embed_tokens" in kwargs['metadata_map'].name_map
        assert "lm_head" in kwargs[
            'metadata_map'].name_map  # Should be present when not tied
        assert kwargs['mesh'] is mesh
        assert isinstance(model.rng, nnx.Rngs)
        assert model.language_model.rng is model.rng

    @patch('tpu_inference.models.jax.qwen2_5_vl.load_hf_weights')
    def test_load_weights_tied(self, mock_load_weights: MagicMock,
                               rng: PRNGKey, mesh: Mesh):
        mock_vllm_config_tied = MockVllmConfig(tie_word_embeddings=True)
        with patch('tpu_inference.models.jax.qwen2_5_vl.Qwen2_5_VisionTransformer', autospec=True), \
             patch('tpu_inference.models.jax.qwen2_5_vl.Qwen2ForCausalLM', autospec=True):
            model = Qwen2_5_VLForConditionalGeneration(mock_vllm_config_tied,
                                                       rng, mesh)

        model.load_weights(rng)
        mock_load_weights.assert_called_once()
        kwargs = mock_load_weights.call_args.kwargs
        assert "lm_head" not in kwargs['metadata_map'].name_map
