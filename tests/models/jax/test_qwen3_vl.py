"""Comprehensive tests for Qwen3VL JAX implementation.

Focuses on:
- Vision Transformer with DeepStack support
- Full ConditionalGeneration module
- Various batch cases for images and videos
"""

from typing import Tuple
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from vllm.config import (CacheConfig, DeviceConfig, MultiModalConfig,
                         ParallelConfig, SchedulerConfig)

from tpu_inference.models.jax.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionTransformer,
    SegmentIds,
    apply_rotary_pos_emb_vision,
    generate_full_segment_ids,
    get_mrope_input_positions,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata


# ==============================================================================
# Configuration Mocking
# ==============================================================================


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(self, hf_config, dtype):
        self.hf_config = hf_config
        self.dtype = dtype
        self.multimodal_config = MultiModalConfig(
            image_input_type="pixel",
            image_token_id=hf_config.image_token_id,
            image_input_shape=None)
        self.model = "mock_qwen3_vl"
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

    def get_vocab_size(self):
        return self.hf_config.vocab_size


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Qwen3VL model."""

    def __init__(
        self,
        tie_word_embeddings: bool = False,
        num_hidden_layers: int = 4,
        vision_depth: int = 4,
        enable_dynamic_image_sizes: bool = False,
    ):
        # Vision config for Qwen3VL
        # Qwen3VL uses SwiGLU MLP and has DeepStack support
        vision_config = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "patch_size": 14,
            "image_size": 28,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "out_hidden_size": 48,
            "depth": vision_depth,
            "num_heads": 4,
            "num_position_embeddings": 256,  # 16x16 grid
            "deepstack_visual_indexes": [1, 2] if vision_depth >= 3 else [0],
        }

        # Use Qwen2VLConfig as base since Qwen3VL may not be available yet
        hf_config = Qwen2VLConfig(
            vision_config=vision_config,
            hidden_size=48,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=96,
            rms_norm_eps=1e-6,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=32000,
            rope_theta=1000000.0,
        )

        self.model_config = MockModelConfig(hf_config, jnp.bfloat16)
        self.cache_config = MagicMock(spec=CacheConfig)
        self.cache_config.cache_dtype = jnp.bfloat16
        self.parallelism_config = MagicMock(spec=ParallelConfig)
        self.scheduler_config = MagicMock(spec=SchedulerConfig)
        self.device_config = MagicMock(spec=DeviceConfig)
        self.load_config = MagicMock()
        self.extra_configs = {}
        self.additional_config = {
            "enable_dynamic_image_sizes": enable_dynamic_image_sizes,
        }


# ==============================================================================
# Fixtures
# ==============================================================================


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


# ==============================================================================
# Utility Function Tests
# ==============================================================================


class TestUtils:
    """Tests for utility functions."""

    def test_apply_rotary_pos_emb_vision(self, rng: PRNGKey):
        """Test rotary position embedding application."""
        B, T, N, H = 1, 10, 4, 16
        x = jax.random.normal(rng, (B, T, N, H))
        rotary_pos_emb = jax.random.normal(rng, (T, H // 2))
        x_rotated = apply_rotary_pos_emb_vision(x, rotary_pos_emb)
        assert x_rotated.shape == (B, T, N, H)

    def test_apply_rotary_pos_emb_vision_batched(self, rng: PRNGKey):
        """Test rotary position embedding with larger batch."""
        B, T, N, H = 4, 16, 8, 32
        x = jax.random.normal(rng, (B, T, N, H))
        rotary_pos_emb = jax.random.normal(rng, (T, H // 2))
        x_rotated = apply_rotary_pos_emb_vision(x, rotary_pos_emb)
        assert x_rotated.shape == (B, T, N, H)

    def test_generate_full_segment_ids(self):
        """Test full attention segment ID generation."""
        seq_len = 10
        padded_seq_len = 16
        segment_ids = generate_full_segment_ids(seq_len, padded_seq_len)

        assert isinstance(segment_ids, SegmentIds)
        assert segment_ids.q.shape == (1, padded_seq_len)
        assert segment_ids.kv.shape == (1, padded_seq_len)

        # Valid tokens should have segment_id=1, padding should have 0
        expected = np.array([[1] * seq_len + [0] * (padded_seq_len - seq_len)])
        np.testing.assert_array_equal(segment_ids.q, expected)
        np.testing.assert_array_equal(segment_ids.kv, expected)

    def test_generate_full_segment_ids_no_padding(self):
        """Test segment IDs when sequence length equals padded length."""
        seq_len = 128
        padded_seq_len = 128
        segment_ids = generate_full_segment_ids(seq_len, padded_seq_len)

        assert segment_ids.q.shape == (1, padded_seq_len)
        # All tokens should have segment_id=1
        np.testing.assert_array_equal(segment_ids.q, np.ones((1, 128)))


class TestMRoPEPositions:
    """Tests for MRoPE position computation."""

    def test_mrope_text_only(self):
        """Test MRoPE positions for text-only input."""
        input_tokens = [1, 2, 3, 4, 5]  # Simple text tokens
        positions, delta = get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=None,
            video_grid_thw=None,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            spatial_merge_size=2,
        )

        assert positions.shape == (3, len(input_tokens))
        # For text-only, all 3 dimensions should have same positions
        np.testing.assert_array_equal(positions[0], positions[1])
        np.testing.assert_array_equal(positions[1], positions[2])

    def test_mrope_with_single_image(self):
        """Test MRoPE positions with a single image."""
        # Text tokens, vision_start, image_token, more text
        image_token_id = 151655
        vision_start_token_id = 151652

        # Create input with image: [text..., vision_start, image_tokens..., text...]
        input_tokens = [1, 2, vision_start_token_id, image_token_id, 3, 4, 5]
        image_grid_thw = [(1, 4, 4)]  # 1 temporal, 4x4 spatial -> 2x2 after merge

        positions, delta = get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            image_token_id=image_token_id,
            video_token_id=151656,
            vision_start_token_id=vision_start_token_id,
            spatial_merge_size=2,
        )

        assert positions.shape[0] == 3  # T, H, W dimensions
        # Total length should account for expanded image tokens
        # Image tokens: 1 * (4/2) * (4/2) = 4 tokens

    def test_mrope_with_video(self):
        """Test MRoPE positions with video input."""
        video_token_id = 151656
        vision_start_token_id = 151652

        input_tokens = [1, vision_start_token_id, video_token_id, 2, 3]
        video_grid_thw = [(2, 4, 4)]  # 2 temporal frames

        positions, delta = get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            image_token_id=151655,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            spatial_merge_size=2,
        )

        assert positions.shape[0] == 3


# ==============================================================================
# Vision Component Tests
# ==============================================================================


class TestQwen3VLVisionRotaryEmbedding:
    """Tests for vision rotary embedding."""

    def test_forward(self):
        """Test rotary embedding generation."""
        dim = 32
        seqlen = 16
        rotary_emb = Qwen3VLVisionRotaryEmbedding(dim=dim)
        emb = rotary_emb(seqlen)
        assert emb.shape == (seqlen, dim // 2)
        assert emb.dtype == jnp.bfloat16

    def test_different_theta(self):
        """Test rotary embedding with different theta values."""
        dim = 16
        seqlen = 8
        rotary_emb_default = Qwen3VLVisionRotaryEmbedding(dim=dim, theta=10000.0)
        rotary_emb_large = Qwen3VLVisionRotaryEmbedding(dim=dim, theta=1000000.0)

        emb_default = rotary_emb_default(seqlen)
        emb_large = rotary_emb_large(seqlen)

        # Different theta should produce different embeddings
        assert not np.allclose(emb_default, emb_large)


class TestQwen3VLVisionPatchMerger:
    """Tests for vision patch merger with postshuffle norm support."""

    def test_forward_final_merger(self, mock_vllm_config: MockVllmConfig,
                                   rngs: nnx.Rngs, rng: PRNGKey):
        """Test final merger (norm before reshape)."""
        vc = mock_vllm_config.model_config.hf_config.vision_config
        dtype = mock_vllm_config.model_config.dtype

        merger = Qwen3VLVisionPatchMerger(
            d_model=vc.out_hidden_size,
            context_dim=vc.hidden_size,
            spatial_merge_size=vc.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            use_postshuffle_norm=False,
        )

        # Input: (seq_len * spatial_merge_unit, hidden_size)
        spatial_merge_unit = vc.spatial_merge_size ** 2
        seq_len = 8
        x = jax.random.normal(rng, (seq_len * spatial_merge_unit, vc.hidden_size))
        y = merger(x)

        assert y.shape == (seq_len, vc.out_hidden_size)

    def test_forward_deepstack_merger(self, mock_vllm_config: MockVllmConfig,
                                       rngs: nnx.Rngs, rng: PRNGKey):
        """Test DeepStack merger (reshape before norm)."""
        vc = mock_vllm_config.model_config.hf_config.vision_config
        dtype = mock_vllm_config.model_config.dtype

        merger = Qwen3VLVisionPatchMerger(
            d_model=vc.out_hidden_size,
            context_dim=vc.hidden_size,
            spatial_merge_size=vc.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            use_postshuffle_norm=True,  # DeepStack uses postshuffle norm
        )

        spatial_merge_unit = vc.spatial_merge_size ** 2
        seq_len = 8
        x = jax.random.normal(rng, (seq_len * spatial_merge_unit, vc.hidden_size))
        y = merger(x)

        assert y.shape == (seq_len, vc.out_hidden_size)


# ==============================================================================
# Vision Transformer Tests
# ==============================================================================


class TestQwen3VLVisionTransformer:
    """Comprehensive tests for vision transformer with DeepStack."""

    @pytest.fixture
    def vision_transformer(self, mock_vllm_config: MockVllmConfig,
                           rngs: nnx.Rngs, mesh: Mesh):
        """Create vision transformer for testing."""
        return Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

    def test_rotary_pos_emb_thw(self, vision_transformer: Qwen3VLVisionTransformer):
        """Test rotary position embedding computation for THW grid."""
        t, h, w = 2, 8, 8
        emb = vision_transformer.rotary_pos_emb_thw(t, h, w)

        vc = vision_transformer.config
        sm = vc.spatial_merge_size
        head_dim_half = (vc.hidden_size // vc.num_heads) // 2
        # Expected shape: (t * merged_h * merged_w, spatial_merge_unit, head_dim_half)
        expected_shape = (t * (h // sm) * (w // sm), sm * sm, head_dim_half)
        assert emb.shape == expected_shape

    def test_fast_pos_embed_interpolate_single_image(
            self, vision_transformer: Qwen3VLVisionTransformer):
        """Test position embedding interpolation for single image."""
        grid_thw = ((1, 8, 8),)
        pos_embeds = vision_transformer.fast_pos_embed_interpolate(grid_thw)

        # Total patches = t * h * w = 1 * 8 * 8 = 64
        expected_patches = 1 * 8 * 8
        assert pos_embeds.shape == (expected_patches, vision_transformer.hidden_size)

    def test_fast_pos_embed_interpolate_multiple_images(
            self, vision_transformer: Qwen3VLVisionTransformer):
        """Test position embedding interpolation for multiple images."""
        grid_thw = ((1, 4, 4), (1, 8, 8), (1, 6, 6))
        pos_embeds = vision_transformer.fast_pos_embed_interpolate(grid_thw)

        # Total patches = 16 + 64 + 36 = 116
        expected_patches = 4 * 4 + 8 * 8 + 6 * 6
        assert pos_embeds.shape == (expected_patches, vision_transformer.hidden_size)

    def test_fast_pos_embed_interpolate_video(
            self, vision_transformer: Qwen3VLVisionTransformer):
        """Test position embedding interpolation for video."""
        grid_thw = ((4, 8, 8),)  # 4 temporal frames
        pos_embeds = vision_transformer.fast_pos_embed_interpolate(grid_thw)

        # Total patches = 4 * 8 * 8 = 256
        expected_patches = 4 * 8 * 8
        assert pos_embeds.shape == (expected_patches, vision_transformer.hidden_size)

    @pytest.mark.parametrize("grid_thw,expected_patches", [
        # Single image
        (((1, 4, 4),), 16),
        # Single larger image
        (((1, 8, 8),), 64),
        # Multiple images of same size
        (((1, 4, 4), (1, 4, 4)),  32),
        # Multiple images of different sizes
        (((1, 4, 4), (1, 8, 8)), 80),
        # Video (multiple temporal frames)
        (((2, 4, 4),), 32),
        # Multiple videos
        (((2, 4, 4), (4, 4, 4)), 96),
        # Mixed images and videos
        (((1, 4, 4), (2, 8, 8)), 16 + 128),
    ])
    def test_call_various_inputs(
            self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
            mesh: Mesh, rng: PRNGKey,
            grid_thw: Tuple[Tuple[int, int, int], ...],
            expected_patches: int):
        """Test vision transformer with various image/video inputs."""
        vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

        # Mock flash attention to avoid sharding issues
        for block in vision_transformer.blocks:
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        # Calculate total number of patches
        num_patches = sum(t * h * w for t, h, w in grid_thw)
        x = jax.random.normal(rng, (num_patches, patch_dim))

        hidden_states, deepstack_features = vision_transformer(x, grid_thw)

        # Expected output tokens after spatial merging
        expected_tokens = sum(
            t * (h // vc.spatial_merge_size) * (w // vc.spatial_merge_size)
            for t, h, w in grid_thw
        )

        assert hidden_states.shape == (expected_tokens, vc.out_hidden_size)
        assert isinstance(deepstack_features, list)

    def test_deepstack_features(
            self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
            mesh: Mesh, rng: PRNGKey):
        """Test that DeepStack features are collected at specified layers."""
        vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

        # Mock flash attention
        for block in vision_transformer.blocks:
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        grid_thw = ((1, 4, 4),)
        num_patches = 16
        x = jax.random.normal(rng, (num_patches, patch_dim))

        hidden_states, deepstack_features = vision_transformer(x, grid_thw)

        # Should have features from DeepStack layers
        deepstack_indexes = vision_transformer.deepstack_visual_indexes
        assert len(deepstack_features) == len(deepstack_indexes)

        # Each DeepStack feature should have same token count as final output
        expected_tokens = 1 * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size)
        for feat in deepstack_features:
            assert feat.shape == (expected_tokens, vc.out_hidden_size)


# ==============================================================================
# Full Model Tests
# ==============================================================================


class TestQwen3VLForConditionalGeneration:
    """Comprehensive tests for the full Qwen3VL model."""

    @pytest.fixture
    def model(self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        """Create model with mocked components for testing."""
        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLVisionTransformer', autospec=True) as MockVision, \
             patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLModel', autospec=True) as MockLM:

            vc = mock_vllm_config.model_config.hf_config.vision_config
            mock_visual = MockVision.return_value
            mock_visual.dtype = mock_vllm_config.model_config.dtype
            mock_visual.config = vc
            mock_visual.spatial_merge_size = vc.spatial_merge_size

            model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
            model.visual = mock_visual
            model.language_model = MockLM.return_value
            model.language_model.embed = MagicMock()
            yield model

    def test_get_input_embeddings_text_only(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test input embeddings for text-only input."""
        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0,
                                       model.config.vocab_size)

        mock_text_embeds = jnp.ones((batch_size, seq_len, model.config.hidden_size))
        model.language_model.embed.return_value = mock_text_embeds

        embeds = model.get_input_embeddings(input_ids, None)
        np.testing.assert_array_equal(embeds, mock_text_embeds)

    def test_get_input_embeddings_empty_multimodal(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test input embeddings with empty multimodal input."""
        batch_size, seq_len = 1, 10
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0,
                                       model.config.vocab_size)

        mock_text_embeds = jnp.ones((batch_size, seq_len, model.config.hidden_size))
        model.language_model.embed.return_value = mock_text_embeds

        empty_mm = jnp.ones((0, model.config.hidden_size))
        embeds = model.get_input_embeddings(input_ids, empty_mm)
        np.testing.assert_array_equal(embeds, mock_text_embeds)

    @patch('tpu_inference.models.jax.qwen3_vl.merge_multimodal_embeddings')
    def test_get_input_embeddings_with_multimodal(
            self, mock_merge: MagicMock,
            model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test input embeddings with multimodal content."""
        batch_size, seq_len = 1, 20
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0,
                                       model.config.vocab_size)

        mock_text_embeds = jnp.ones((batch_size, seq_len, model.config.hidden_size))
        model.language_model.embed.return_value = mock_text_embeds

        mm_embeds = jnp.ones((8, model.config.hidden_size))
        mock_merged = jnp.ones((batch_size, seq_len + 8, model.config.hidden_size))
        mock_merge.return_value = mock_merged

        embeds = model.get_input_embeddings(input_ids, mm_embeds)
        np.testing.assert_array_equal(embeds, mock_merged)
        mock_merge.assert_called_once_with(
            input_ids, mock_text_embeds, mm_embeds,
            [model.image_token_id, model.video_token_id])

    def test_get_multimodal_embeddings_single_image(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test multimodal embeddings for single image."""
        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2

        grid_thw = ((1, 4, 4),)
        num_patches = 16
        pixel_values = jax.random.normal(rng, (num_patches, patch_dim))

        tokens_per_image = 1 * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size)
        mock_embeds = jnp.ones((tokens_per_image, vc.out_hidden_size))
        mock_deepstack = [jnp.ones((tokens_per_image, vc.out_hidden_size))]

        model.visual.return_value = (mock_embeds, mock_deepstack)

        image_embeds, deepstack_embeds = model.get_multimodal_embeddings(
            pixel_values, grid_thw)

        assert image_embeds.shape == (tokens_per_image, vc.out_hidden_size)
        assert len(deepstack_embeds) == 1

    def test_get_multimodal_embeddings_multiple_images(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test multimodal embeddings for multiple images."""
        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2

        grid_thw = ((1, 4, 4), (1, 8, 8))
        num_patches = 16 + 64
        pixel_values = jax.random.normal(rng, (num_patches, patch_dim))

        total_tokens = (
            1 * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size) +
            1 * (8 // vc.spatial_merge_size) * (8 // vc.spatial_merge_size)
        )
        mock_embeds = jnp.ones((total_tokens, vc.out_hidden_size))
        mock_deepstack = [jnp.ones((total_tokens, vc.out_hidden_size))]

        model.visual.return_value = (mock_embeds, mock_deepstack)

        image_embeds, deepstack_embeds = model.get_multimodal_embeddings(
            pixel_values, grid_thw)

        assert image_embeds.shape == (total_tokens, vc.out_hidden_size)

    def test_get_multimodal_embeddings_video(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test multimodal embeddings for video input."""
        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2

        # Video with 4 temporal frames
        grid_thw = ((4, 8, 8),)
        num_patches = 4 * 8 * 8
        pixel_values = jax.random.normal(rng, (num_patches, patch_dim))

        total_tokens = 4 * (8 // vc.spatial_merge_size) * (8 // vc.spatial_merge_size)
        mock_embeds = jnp.ones((total_tokens, vc.out_hidden_size))
        mock_deepstack = [jnp.ones((total_tokens, vc.out_hidden_size))]

        model.visual.return_value = (mock_embeds, mock_deepstack)

        image_embeds, deepstack_embeds = model.get_multimodal_embeddings(
            pixel_values, grid_thw)

        assert image_embeds.shape == (total_tokens, vc.out_hidden_size)

    def test_call_text_only(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test forward pass with text-only input."""
        kv_caches = [MagicMock()]
        batch_size, seq_len = 2, 16
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0,
                                       model.config.vocab_size)
        attn_meta = MagicMock(spec=AttentionMetadata)

        mock_hidden = jnp.ones((batch_size, seq_len, model.config.hidden_size))
        model.language_model.return_value = ([MagicMock()], mock_hidden)

        new_kvs, x, aux = model(kv_caches, input_ids, attn_meta)

        model.language_model.assert_called_once()
        assert x.shape == (batch_size, seq_len, model.config.hidden_size)
        assert len(aux) == 0

    def test_call_with_images(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test forward pass with image input."""
        kv_caches = [MagicMock()]
        batch_size, seq_len = 1, 24
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0,
                                       model.config.vocab_size)
        attn_meta = MagicMock(spec=AttentionMetadata)

        vc = model.config.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
        grid_thw = ((1, 4, 4),)
        num_patches = 16
        pixel_values = jax.random.normal(rng, (num_patches, patch_dim))

        tokens_per_image = 1 * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size)
        mock_embeds = jnp.ones((tokens_per_image, vc.out_hidden_size))
        mock_deepstack = [jnp.ones((tokens_per_image, vc.out_hidden_size))]
        model.visual.return_value = (mock_embeds, mock_deepstack)

        mock_hidden = jnp.ones((batch_size, seq_len, model.config.hidden_size))
        model.language_model.return_value = ([MagicMock()], mock_hidden)
        model.language_model.embed = MagicMock(
            return_value=jnp.ones((batch_size, seq_len, model.config.hidden_size)))

        with patch('tpu_inference.models.jax.qwen3_vl.merge_multimodal_embeddings') as mock_merge:
            mock_merge.return_value = jnp.ones((batch_size, seq_len, model.config.hidden_size))
            new_kvs, x, aux = model(
                kv_caches, input_ids, attn_meta,
                pixel_values=pixel_values, image_grid_thw=grid_thw)

        model.visual.assert_called_once()
        assert x.shape == (batch_size, seq_len, model.config.hidden_size)

    def test_compute_logits(
            self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey):
        """Test logits computation."""
        batch_size, seq_len = 2, 10
        hidden_states = jnp.ones((batch_size, seq_len, model.config.hidden_size))
        mock_logits = jnp.ones((batch_size, seq_len, model.config.vocab_size))

        model.language_model.compute_logits.return_value = mock_logits

        logits = model.compute_logits(hidden_states)
        np.testing.assert_array_equal(logits, mock_logits)
        model.language_model.compute_logits.assert_called_once_with(hidden_states)

    def test_get_mrope_input_positions_wrapper(
            self, model: Qwen3VLForConditionalGeneration):
        """Test MRoPE position computation wrapper method."""
        input_tokens = [1, 2, 3, 4, 5]
        positions, delta = model.get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=None,
            video_grid_thw=None,
        )

        assert positions.shape == (3, len(input_tokens))

    @patch('tpu_inference.models.jax.qwen3_vl.load_hf_weights')
    def test_load_weights(
            self, mock_load_weights: MagicMock,
            mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        """Test weight loading."""
        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLVisionTransformer', autospec=True), \
             patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLModel', autospec=True):
            model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)

        model.load_weights(rng)
        mock_load_weights.assert_called_once()
        kwargs = mock_load_weights.call_args.kwargs

        assert kwargs['vllm_config'] == mock_vllm_config
        assert kwargs['model'] is model
        assert kwargs['mesh'] is mesh

        # Check weight mappings
        name_map = kwargs['metadata_map'].name_map
        assert "model.embed_tokens" in name_map
        assert "visual.patch_embed.proj" in name_map
        assert "visual.merger.ln_q" in name_map

    @patch('tpu_inference.models.jax.qwen3_vl.load_hf_weights')
    def test_load_weights_tied_embeddings(
            self, mock_load_weights: MagicMock, rng: PRNGKey, mesh: Mesh):
        """Test weight loading with tied embeddings."""
        mock_vllm_config = MockVllmConfig(tie_word_embeddings=True)

        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLVisionTransformer', autospec=True), \
             patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLModel', autospec=True):
            model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)

        model.load_weights(rng)
        kwargs = mock_load_weights.call_args.kwargs
        name_map = kwargs['metadata_map'].name_map

        # lm_head should not be in mappings when tied
        assert "lm_head" not in name_map


# ==============================================================================
# Batch Processing Tests
# ==============================================================================


class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    @pytest.fixture
    def full_model(self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        """Create actual model (not mocked) for batch testing."""
        with patch('tpu_inference.models.jax.qwen3_vl.sharded_flash_attention') as mock_flash:
            # Mock flash attention to return appropriate shape
            def flash_side_effect(mesh, causal, sm_scale, vmem_limit_bytes):
                return MagicMock(side_effect=lambda q, k, v, seg: jnp.ones_like(q))
            mock_flash.side_effect = flash_side_effect

            model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)

            # Mock flash attention in vision blocks
            for block in model.visual.blocks:
                block.attn.flash_attention = MagicMock(
                    side_effect=lambda q, k, v, seg: jnp.ones_like(q))

            yield model

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_vision_transformer_batch_sizes(
            self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
            mesh: Mesh, rng: PRNGKey, batch_size: int):
        """Test vision transformer with different batch sizes.

        Note: Current implementation only supports batch_size=1 for vision.
        This test validates that constraint.
        """
        vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

        # Mock flash attention
        for block in vision_transformer.blocks:
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        # Single image per batch item
        grid_thw = ((1, 4, 4),)
        num_patches = 16
        x = jax.random.normal(rng, (num_patches, patch_dim))

        # Vision transformer processes images one at a time
        hidden_states, deepstack_features = vision_transformer(x, grid_thw)

        expected_tokens = 1 * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size)
        assert hidden_states.shape == (expected_tokens, vc.out_hidden_size)

    @pytest.mark.parametrize("num_images", [1, 2, 3, 5])
    def test_multiple_images_in_batch(
            self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
            mesh: Mesh, rng: PRNGKey, num_images: int):
        """Test processing multiple images in a single forward pass."""
        vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

        for block in vision_transformer.blocks:
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        # Multiple images of same size
        grid_thw = tuple((1, 4, 4) for _ in range(num_images))
        num_patches = 16 * num_images
        x = jax.random.normal(rng, (num_patches, patch_dim))

        hidden_states, deepstack_features = vision_transformer(x, grid_thw)

        tokens_per_image = 1 * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size)
        expected_tokens = tokens_per_image * num_images
        assert hidden_states.shape == (expected_tokens, vc.out_hidden_size)

    @pytest.mark.parametrize("temporal_frames", [1, 2, 4, 8])
    def test_video_temporal_frames(
            self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
            mesh: Mesh, rng: PRNGKey, temporal_frames: int):
        """Test video processing with different temporal frame counts."""
        vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

        for block in vision_transformer.blocks:
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        grid_thw = ((temporal_frames, 4, 4),)
        num_patches = temporal_frames * 4 * 4
        x = jax.random.normal(rng, (num_patches, patch_dim))

        hidden_states, deepstack_features = vision_transformer(x, grid_thw)

        expected_tokens = temporal_frames * (4 // vc.spatial_merge_size) * (4 // vc.spatial_merge_size)
        assert hidden_states.shape == (expected_tokens, vc.out_hidden_size)


# ==============================================================================
# Text Model Tests
# ==============================================================================


class TestQwen3VLModel:
    """Tests for the text model with DeepStack injection."""

    @pytest.fixture
    def text_model(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs, mesh: Mesh):
        """Create text model for testing."""
        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3DecoderLayer') as MockLayer:
            # Mock decoder layers
            mock_layer = MagicMock()
            mock_layer.return_value = (MagicMock(), jnp.zeros((1, 10, 48)))
            MockLayer.return_value = mock_layer

            model = Qwen3VLModel(mock_vllm_config, rngs, mesh)
            yield model

    def test_forward_text_only(
            self, text_model: Qwen3VLModel, rng: PRNGKey):
        """Test text model forward pass without visual features."""
        batch_size, seq_len = 1, 16
        hidden_size = 48  # From mock config

        kv_caches = [MagicMock() for _ in range(4)]  # 4 layers
        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 32000)
        attn_meta = MagicMock(spec=AttentionMetadata)

        # Mock layer forward
        for layer in text_model.layers:
            layer.return_value = (MagicMock(), jnp.ones((batch_size, seq_len, hidden_size)))

        kv_caches, hidden = text_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attn_meta,
        )

        assert hidden.shape == (batch_size, seq_len, hidden_size)

    def test_compute_logits_tied(self, mock_vllm_config: MockVllmConfig,
                                   rngs: nnx.Rngs, mesh: Mesh, rng: PRNGKey):
        """Test logits computation with tied embeddings."""
        mock_config = MockVllmConfig(tie_word_embeddings=True)

        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3DecoderLayer'):
            model = Qwen3VLModel(mock_config, rngs, mesh)

        batch_size, seq_len = 1, 8
        hidden_states = jax.random.normal(rng, (batch_size, seq_len, 48))

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (batch_size, seq_len, 32000)  # vocab_size

    def test_compute_logits_untied(self, mock_vllm_config: MockVllmConfig,
                                     rngs: nnx.Rngs, mesh: Mesh, rng: PRNGKey):
        """Test logits computation with separate lm_head."""
        mock_config = MockVllmConfig(tie_word_embeddings=False)

        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3DecoderLayer'):
            model = Qwen3VLModel(mock_config, rngs, mesh)

        batch_size, seq_len = 1, 8
        hidden_states = jax.random.normal(rng, (batch_size, seq_len, 48))

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (batch_size, seq_len, 32000)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for end-to-end functionality."""

    def test_vision_to_language_pipeline(
            self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        """Test the full vision-to-language pipeline."""
        with patch('tpu_inference.models.jax.qwen3_vl.sharded_flash_attention') as mock_flash:
            def flash_side_effect(mesh, causal, sm_scale, vmem_limit_bytes):
                return MagicMock(side_effect=lambda q, k, v, seg: jnp.ones_like(q))
            mock_flash.side_effect = flash_side_effect

            # Create vision transformer
            rngs = nnx.Rngs(params=rng)
            vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

            for block in vision_transformer.blocks:
                block.attn.flash_attention = MagicMock(
                    side_effect=lambda q, k, v, seg: jnp.ones_like(q))

            vc = vision_transformer.config
            patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

            # Process image
            grid_thw = ((1, 8, 8),)
            num_patches = 64
            pixel_values = jax.random.normal(rng, (num_patches, patch_dim))

            hidden_states, deepstack_features = vision_transformer(pixel_values, grid_thw)

            # Verify output can be used by language model
            expected_tokens = 1 * (8 // vc.spatial_merge_size) * (8 // vc.spatial_merge_size)
            assert hidden_states.shape[0] == expected_tokens
            assert hidden_states.shape[1] == vc.out_hidden_size

            # DeepStack features should have same shape
            for feat in deepstack_features:
                assert feat.shape == hidden_states.shape

    def test_mixed_image_video_batch(
            self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        """Test processing mixed images and videos in same batch."""
        rngs = nnx.Rngs(params=rng)
        vision_transformer = Qwen3VLVisionTransformer(mock_vllm_config, rngs, mesh)

        for block in vision_transformer.blocks:
            block.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        vc = vision_transformer.config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        # Mixed: 2 images + 1 video
        grid_thw = (
            (1, 4, 4),   # Image 1
            (1, 8, 8),   # Image 2
            (4, 4, 4),   # Video (4 frames)
        )
        num_patches = 16 + 64 + 64
        pixel_values = jax.random.normal(rng, (num_patches, patch_dim))

        hidden_states, deepstack_features = vision_transformer(pixel_values, grid_thw)

        # Calculate expected tokens
        sm = vc.spatial_merge_size
        expected_tokens = (
            1 * (4 // sm) * (4 // sm) +  # Image 1
            1 * (8 // sm) * (8 // sm) +  # Image 2
            4 * (4 // sm) * (4 // sm)    # Video
        )
        assert hidden_states.shape == (expected_tokens, vc.out_hidden_size)
