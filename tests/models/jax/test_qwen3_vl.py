from __future__ import annotations

from typing import Tuple
from unittest.mock import MagicMock, patch

import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh

pytest.importorskip("vllm")
pytest.importorskip("transformers")
from transformers import Qwen3Config
from vllm.config import CacheConfig, DeviceConfig, MultiModalConfig, ParallelConfig, SchedulerConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.models.jax.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionTransformer,
    SegmentIds,
    _infer_pos_embed_grid_hw,
    apply_interleaved_mrope,
    apply_rotary_pos_emb_vision,
    compute_vision_counts_per_sequence,
    generate_segment_ids_from_grid_thw,
    get_mrope_input_positions,
    pad_segment_ids_for_attention,
    rotate_half,
)
from tpu_inference.runner.kv_cache import create_kv_caches


# --- Configuration Mocking ---
class MockVisionConfig:
    """Mock vision config for Qwen3VL testing."""

    def __init__(
        self,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        patch_size: int = 2,
        image_size: int = 8,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 64,
        depth: int = 1,
        num_heads: int = 4,
        num_position_embeddings: int = 16,
        deepstack_visual_indexes: Tuple[int, ...] = (0,),
        tokens_per_second: float = 1.0,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.tokens_per_second = tokens_per_second

    def to_dict(self) -> dict:
        """Return dict for HuggingFace config JSON serialization."""
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "temporal_patch_size": self.temporal_patch_size,
            "in_channels": self.in_channels,
            "spatial_merge_size": self.spatial_merge_size,
            "out_hidden_size": self.out_hidden_size,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "num_position_embeddings": self.num_position_embeddings,
            "deepstack_visual_indexes": list(self.deepstack_visual_indexes),
            "tokens_per_second": self.tokens_per_second,
        }


class MockModelConfig:
    """A mock ModelConfig sufficient for testing the Qwen3 VL model."""

    def __init__(self, hf_config: Qwen3Config, dtype: jnp.dtype):
        self.hf_config = hf_config
        self.dtype = dtype
        self.multimodal_config = MultiModalConfig(
            image_input_type="pixel",
            image_token_id=hf_config.image_token_id,
            image_input_shape=None,
        )
        self.model = "mock_qwen3_vl"
        self.tokenizer = "mock_tokenizer"
        self.tokenizer_mode = "auto"
        self.trust_remote_code = True
        self.seed = 0

    def is_multimodal_model(self) -> bool:
        return True

    def get_hidden_size(self) -> int:
        return int(self.hf_config.hidden_size)

    def get_head_size(self) -> int:
        return int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)

    def get_vocab_size(self) -> int:
        return int(self.hf_config.vocab_size)


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Qwen3 VL model."""

    def __init__(self, tie_word_embeddings: bool = False):
        vision_config = MockVisionConfig()
        hf_config = Qwen3Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=tie_word_embeddings,
        )
        hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        # Multimodal special tokens must be within vocab_size for embedding lookups.
        hf_config.image_token_id = 100
        hf_config.video_token_id = 101
        hf_config.vision_start_token_id = 102

        # Attach a minimal vision config object with the attributes used by the model.
        hf_config.vision_config = vision_config

        # Enable MRoPE branch in apply_rope when positions is (3, seq_len).
        hf_config.rope_scaling = {"mrope_section": [4, 2, 2]}

        self.model_config = MockModelConfig(hf_config, jnp.bfloat16)
        self.cache_config = MagicMock(spec=CacheConfig)
        self.cache_config.cache_dtype = "auto"
        self.parallelism_config = MagicMock(spec=ParallelConfig)
        self.scheduler_config = MagicMock(spec=SchedulerConfig)
        self.device_config = MagicMock(spec=DeviceConfig)
        self.load_config = MagicMock()
        self.extra_configs = {}
        self.additional_config = {"enable_dynamic_image_sizes": False}


def _num_placeholders_for_grid(
    grid_thw: Tuple[int, int, int], spatial_merge_size: int
) -> int:
    t, h, w = grid_thw
    return int(t * (h // spatial_merge_size) * (w // spatial_merge_size))


def _make_attention_metadata(seq_len: int) -> AttentionMetadata:
    num_reqs = 1
    max_num_blocks_per_req = 4
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :], (3, seq_len)
    )
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.array([seq_len], dtype=jnp.int32)
    query_start_loc = jnp.array([0, seq_len], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, num_reqs], dtype=jnp.int32)
    return AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )


def _make_kv_caches(model: Qwen3VLForConditionalGeneration,
                    mesh: Mesh) -> list[jax.Array]:
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.language_model.layers[0].self_attn.head_dim
    layer_names = ["layer"] * model.config.num_hidden_layers
    return create_kv_caches(
        num_blocks=4,
        block_size=32,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        mesh=mesh,
        layer_names=layer_names,
        cache_dtype=jnp.bfloat16,
    )


# --- Fixtures ---
@pytest.fixture(scope="module")
def mesh() -> Mesh:
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
    """Provides a mock VllmConfig for testing."""
    return MockVllmConfig()


@pytest.fixture
def rngs(rng: PRNGKey) -> nnx.Rngs:
    """Provides flax nnx.Rngs for model initialization."""
    return nnx.Rngs(params=rng)


@pytest.fixture
def hf_config(mock_vllm_config: MockVllmConfig) -> Qwen3Config:
    """Provides the HuggingFace config from the mock vllm config."""
    return mock_vllm_config.model_config.hf_config


class TestUtils:
    def test_apply_rotary_pos_emb_vision_shapes(self, rng: PRNGKey):
        b, t, n, h = 1, 6, 4, 16
        x = jax.random.normal(rng, (b, t, n, h))
        rotary_pos_emb = jax.random.normal(rng, (t, h // 2))
        y = apply_rotary_pos_emb_vision(x, rotary_pos_emb)
        assert y.shape == x.shape

    def test_generate_segment_ids_from_grid_thw(self):
        grid_thw = ((1, 4, 4), (2, 4, 4))
        segment_ids = generate_segment_ids_from_grid_thw(grid_thw)
        # (1,4,4) -> 16 tokens (1 frame), (2,4,4) -> 32 tokens (2 frames).
        # Each frame gets a unique segment ID to prevent cross-frame attention.
        assert segment_ids.shape == (48,)
        np.testing.assert_array_equal(segment_ids[:16], np.ones(16, dtype=np.int32))      # image frame
        np.testing.assert_array_equal(segment_ids[16:32], np.full(16, 2, dtype=np.int32)) # video frame 1
        np.testing.assert_array_equal(segment_ids[32:], np.full(16, 3, dtype=np.int32))   # video frame 2

    def test_pad_segment_ids_for_attention(self):
        segment_ids = jnp.array([1, 1, 2, 2], dtype=jnp.int32)
        padded = pad_segment_ids_for_attention(segment_ids, padded_seq_len=8)
        assert isinstance(padded, SegmentIds)
        assert padded.q.shape == (1, 8)
        assert padded.kv.shape == (1, 8)
        np.testing.assert_array_equal(
            np.array(padded.q), np.array([[1, 1, 2, 2, 0, 0, 0, 0]], dtype=np.int32)
        )

    def test_pad_segment_ids_exact_length_no_padding(self):
        segment_ids = jnp.array([1, 1, 2, 2], dtype=jnp.int32)
        padded = pad_segment_ids_for_attention(segment_ids, padded_seq_len=4)
        np.testing.assert_array_equal(np.array(padded.q), np.array([[1, 1, 2, 2]], dtype=np.int32))

    def test_pad_segment_ids_empty_input(self):
        segment_ids = jnp.array([], dtype=jnp.int32)
        padded = pad_segment_ids_for_attention(segment_ids, padded_seq_len=4)
        np.testing.assert_array_equal(np.array(padded.q), np.array([[0, 0, 0, 0]], dtype=np.int32))

    def test_generate_segment_ids_single_grid(self):
        segment_ids = generate_segment_ids_from_grid_thw(((1, 2, 2),))
        assert segment_ids.shape == (4,)
        np.testing.assert_array_equal(np.array(segment_ids), np.array([1, 1, 1, 1], dtype=np.int32))


class TestInferPosEmbedGridHw:
    """Tests for _infer_pos_embed_grid_hw grid inference helper."""

    def test_perfect_square(self):
        # 16 -> (4, 4)
        assert _infer_pos_embed_grid_hw(16) == (4, 4)

    def test_large_perfect_square(self):
        # 2304 -> (48, 48) - default Qwen3VL size
        assert _infer_pos_embed_grid_hw(2304) == (48, 48)

    def test_rectangular_factorization_prefers_closer_to_square(self):
        # 12 -> (3, 4) not (2, 6) - algorithm starts from sqrt and goes down
        h, w = _infer_pos_embed_grid_hw(12)
        assert h * w == 12
        assert h == 3 and w == 4

    def test_another_rectangular_case(self):
        # 18 -> (3, 6)
        h, w = _infer_pos_embed_grid_hw(18)
        assert h * w == 18
        assert h == 3 and w == 6

    def test_prime_number_returns_nx1(self):
        # 7 -> (7, 1) or (1, 7) - no other factorization
        h, w = _infer_pos_embed_grid_hw(7)
        assert h * w == 7
        # Prime numbers result in (n, 1)
        assert (h == 7 and w == 1) or (h == 1 and w == 7)

    def test_prime_number_13(self):
        h, w = _infer_pos_embed_grid_hw(13)
        assert h * w == 13

    def test_one_returns_1x1(self):
        assert _infer_pos_embed_grid_hw(1) == (1, 1)

    def test_two_returns_valid_factorization(self):
        h, w = _infer_pos_embed_grid_hw(2)
        assert h * w == 2


class TestApplyInterleavedMrope:
    """Tests for apply_interleaved_mrope MRoPE interleaving logic."""

    def test_equal_sections_interleave_correctly(self):
        # mrope_section = [4, 4, 4] with head_dim//2 = 12
        freqs = jnp.ones((3, 1, 8, 12), dtype=jnp.float32)
        result = apply_interleaved_mrope(freqs, [4, 4, 4])
        assert result.shape == (1, 8, 12)

    def test_unequal_sections_with_t_remainder(self):
        # mrope_section = [6, 3, 3] - T has 3 extra dims after interleaving
        freqs = jnp.arange(36, dtype=jnp.float32).reshape(3, 1, 1, 12)
        result = apply_interleaved_mrope(freqs, [6, 3, 3])
        assert result.shape == (1, 1, 12)

    def test_typical_qwen3vl_section(self):
        # Default Qwen3VL dense uses [24, 20, 20] for head_dim=128 (half=64)
        freqs = jnp.ones((3, 2, 16, 64), dtype=jnp.float32)
        result = apply_interleaved_mrope(freqs, [24, 20, 20])
        assert result.shape == (2, 16, 64)

    def test_small_section_values(self):
        # mrope_section = [2, 2, 2] with head_dim//2 = 6
        freqs = jnp.ones((3, 1, 4, 6), dtype=jnp.float32)
        result = apply_interleaved_mrope(freqs, [2, 2, 2])
        assert result.shape == (1, 4, 6)

    def test_interleaving_preserves_values(self):
        # Verify the interleaving pattern is correct
        # With [1, 1, 1], we should get [T0, H0, W0] ordering
        # freqs shape must be (3, bs, seq_len, head_dim//2) where head_dim//2 = sum(mrope_section) = 3
        t_val, h_val, w_val = 1.0, 2.0, 3.0
        # Create freqs with shape (3, 1, 1, 3) - each axis has values at its section indices
        freqs = jnp.zeros((3, 1, 1, 3), dtype=jnp.float32)
        freqs = freqs.at[0, 0, 0, 0].set(t_val)  # T section is [0:1]
        freqs = freqs.at[1, 0, 0, 1].set(h_val)  # H section is [1:2]
        freqs = freqs.at[2, 0, 0, 2].set(w_val)  # W section is [2:3]
        result = apply_interleaved_mrope(freqs, [1, 1, 1])
        # Result should be (1, 1, 3) with values interleaved as [T0, H0, W0]
        assert result.shape == (1, 1, 3)
        np.testing.assert_allclose(np.array(result[0, 0]), [t_val, h_val, w_val])

    def test_zero_section_allowed(self):
        # Edge case: one section can be zero
        freqs = jnp.ones((3, 1, 4, 6), dtype=jnp.float32)
        result = apply_interleaved_mrope(freqs, [3, 3, 0])
        assert result.shape == (1, 4, 6)


class TestComputeVisionCountsPerSequence:
    """Tests for compute_vision_counts_per_sequence batch utility."""

    def test_single_sequence_with_images(self):
        input_ids = jnp.array([[1, 100, 2, 100, 3]])  # 2 image tokens
        attention_mask = jnp.ones_like(input_ids)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 2
        assert int(num_videos[0]) == 0

    def test_single_sequence_with_videos(self):
        input_ids = jnp.array([[1, 101, 2, 101, 101, 3]])  # 3 video tokens
        attention_mask = jnp.ones_like(input_ids)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 0
        assert int(num_videos[0]) == 3

    def test_mixed_images_and_videos(self):
        input_ids = jnp.array([[100, 1, 101, 2, 100]])  # 2 images, 1 video
        attention_mask = jnp.ones_like(input_ids)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 2
        assert int(num_videos[0]) == 1

    def test_masked_positions_ignored(self):
        input_ids = jnp.array([[100, 100, 100]])  # 3 image tokens
        attention_mask = jnp.array([[1, 0, 1]])  # middle one masked
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 2  # only counts unmasked

    def test_all_masked_returns_zero(self):
        input_ids = jnp.array([[100, 100, 100]])
        attention_mask = jnp.zeros_like(input_ids)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 0
        assert int(num_videos[0]) == 0

    def test_batch_processing(self):
        input_ids = jnp.array([[100, 1, 2], [1, 101, 101]])
        attention_mask = jnp.ones_like(input_ids)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        np.testing.assert_array_equal(np.array(num_images), [1, 0])
        np.testing.assert_array_equal(np.array(num_videos), [0, 2])

    def test_no_vision_tokens(self):
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        attention_mask = jnp.ones_like(input_ids)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 0
        assert int(num_videos[0]) == 0

    def test_empty_sequence(self):
        input_ids = jnp.array([[]], dtype=jnp.int32).reshape(1, 0)
        attention_mask = jnp.array([[]], dtype=jnp.int32).reshape(1, 0)
        num_images, num_videos = compute_vision_counts_per_sequence(
            input_ids, attention_mask, image_token_id=100, video_token_id=101
        )
        assert int(num_images[0]) == 0
        assert int(num_videos[0]) == 0


class TestRotateHalf:
    """Tests for rotate_half RoPE helper function."""

    def test_basic_rotation(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        # First half becomes negated second half, second half becomes first half
        # [-3, -4, 1, 2]
        np.testing.assert_allclose(np.array(result), [-3.0, -4.0, 1.0, 2.0])

    def test_2d_input(self):
        x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = rotate_half(x)
        expected = jnp.array([[-3.0, -4.0, 1.0, 2.0], [-7.0, -8.0, 5.0, 6.0]])
        np.testing.assert_allclose(np.array(result), np.array(expected))

    def test_3d_input(self):
        x = jnp.ones((2, 3, 4))
        result = rotate_half(x)
        assert result.shape == x.shape
        # First half of last dim should be -1, second half should be 1
        np.testing.assert_allclose(np.array(result[..., :2]), -1.0 * np.ones((2, 3, 2)))
        np.testing.assert_allclose(np.array(result[..., 2:]), 1.0 * np.ones((2, 3, 2)))

    def test_preserves_dtype(self):
        x = jnp.ones((4,), dtype=jnp.bfloat16)
        result = rotate_half(x)
        assert result.dtype == jnp.bfloat16


class TestMRoPEPositions:
    def test_mrope_text_only_positions(self, hf_config: Qwen3Config):
        tokens = [1, 2, 3, 4]
        positions, delta = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=None,
            video_grid_thw=None,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))
        assert int(delta) == 0
        np.testing.assert_array_equal(positions[0], positions[1])
        np.testing.assert_array_equal(positions[1], positions[2])

    def test_mrope_vision_start_at_end_does_not_crash(self, hf_config: Qwen3Config):
        tokens = [1, 2, hf_config.vision_start_token_id]
        positions, delta = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=None,
            video_grid_thw=None,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))
        assert int(delta) == 0

    def test_mrope_single_image_3d_structure(self, hf_config: Qwen3Config):
        grid = (1, 4, 4)
        n_img = _num_placeholders_for_grid(grid, hf_config.vision_config.spatial_merge_size)

        tokens = [11, hf_config.vision_start_token_id] + [hf_config.image_token_id] * n_img + [12]
        positions, _ = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=[grid],
            video_grid_thw=None,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))

        # Text (including vision_start) should have identical positions in all 3 dims.
        np.testing.assert_array_equal(positions[0, :2], positions[1, :2])
        np.testing.assert_array_equal(positions[1, :2], positions[2, :2])

        # Image placeholders should have constant T but varying H/W.
        t_slice = np.array(positions[0, 2 : 2 + n_img])
        h_slice = np.array(positions[1, 2 : 2 + n_img])
        w_slice = np.array(positions[2, 2 : 2 + n_img])
        assert len(set(t_slice.tolist())) == 1
        assert len(set(h_slice.tolist())) > 1
        assert len(set(w_slice.tolist())) > 1

    def test_mrope_consecutive_image_then_video(self, hf_config: Qwen3Config):
        img_grid = (1, 4, 4)
        vid_grid = (2, 4, 4)
        n_img = _num_placeholders_for_grid(img_grid, hf_config.vision_config.spatial_merge_size)
        n_vid = _num_placeholders_for_grid(vid_grid, hf_config.vision_config.spatial_merge_size)

        tokens = (
            [hf_config.vision_start_token_id]
            + [hf_config.image_token_id] * n_img
            + [hf_config.vision_start_token_id]
            + [hf_config.video_token_id] * n_vid
        )
        positions, _ = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=[img_grid],
            video_grid_thw=[vid_grid],
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))

        # Video temporal positions should not be constant when t>1.
        video_start = 1 + n_img + 1
        t_vid = np.array(positions[0, video_start : video_start + n_vid])
        assert len(set(t_vid.tolist())) > 1

    def test_mrope_multiple_images_consecutive(self, hf_config: Qwen3Config):
        """Two images back-to-back without text between."""
        grid = (1, 4, 4)
        n_img = _num_placeholders_for_grid(grid, hf_config.vision_config.spatial_merge_size)
        tokens = (
            [hf_config.vision_start_token_id]
            + [hf_config.image_token_id] * n_img
            + [hf_config.vision_start_token_id]
            + [hf_config.image_token_id] * n_img
        )
        positions, _ = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=[grid, grid],
            video_grid_thw=None,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))

    def test_mrope_video_before_image_order(self, hf_config: Qwen3Config):
        """Video tokens appear before image tokens in the sequence."""
        img_grid = (1, 4, 4)
        vid_grid = (2, 4, 4)
        n_img = _num_placeholders_for_grid(img_grid, hf_config.vision_config.spatial_merge_size)
        n_vid = _num_placeholders_for_grid(vid_grid, hf_config.vision_config.spatial_merge_size)
        tokens = (
            [hf_config.vision_start_token_id]
            + [hf_config.video_token_id] * n_vid
            + [hf_config.vision_start_token_id]
            + [hf_config.image_token_id] * n_img
        )
        positions, _ = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=[img_grid],
            video_grid_thw=[vid_grid],
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))

    def test_mrope_extra_grids_allowed(self, hf_config: Qwen3Config):
        """Extra grids beyond what's needed should not raise."""
        grid = (1, 4, 4)
        n_img = _num_placeholders_for_grid(grid, hf_config.vision_config.spatial_merge_size)
        tokens = [hf_config.vision_start_token_id] + [hf_config.image_token_id] * n_img
        # Provide 3 grids but only 1 image
        positions, _ = get_mrope_input_positions(
            input_tokens=tokens,
            image_grid_thw=[grid, grid, grid],
            video_grid_thw=None,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=hf_config.vision_start_token_id,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        assert positions.shape == (3, len(tokens))


class TestRopeInterface:
    def test_apply_rope_accepts_mrope_positions_when_configured(self, hf_config: Qwen3Config):
        seq_len = 8
        num_heads = 2
        head_dim = 16
        x = jnp.ones((seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        positions_3d = jnp.stack([positions, positions, positions])

        y = apply_rope(
            x,
            positions_3d,
            head_dim=head_dim,
            rope_theta=float(hf_config.rope_theta),
            rope_scaling=getattr(hf_config, "rope_scaling", None),
        )
        assert y.shape == x.shape
        assert y.dtype == x.dtype


class TestVisionPosEmbedInterpolation:
    def test_fast_pos_embed_interpolate_supports_rectangular_base_grid(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        cfg = copy.deepcopy(mock_vllm_config.model_config.hf_config)
        cfg.vision_config = MockVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            patch_size=2,
            image_size=(6, 8),  # 3x4 in patch space
            temporal_patch_size=1,
            in_channels=3,
            spatial_merge_size=2,
            out_hidden_size=64,
            depth=0,  # avoid flash attention dependency in this unit test
            num_heads=4,
            num_position_embeddings=12,  # 3x4 grid (non-square)
            deepstack_visual_indexes=(),
        )
        model_config = MockModelConfig(hf_config=cfg, dtype=mock_vllm_config.model_config.dtype)
        rect_vllm_config = MockVllmConfig()
        rect_vllm_config.model_config = model_config

        vision = Qwen3VLVisionTransformer(rect_vllm_config, nnx.Rngs(params=rng), mesh)
        assert vision.pos_embed_grid_h != vision.pos_embed_grid_w

        pos_embeds = vision.fast_pos_embed_interpolate(((1, 2, 4),))
        assert pos_embeds.shape == (8, cfg.vision_config.hidden_size)


class TestQwen3VLForConditionalGeneration:
    """Tests for the Qwen3VL model with proper mocking."""

    @pytest.fixture
    def model(self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        """Creates a model with mocked vision and language components."""
        with patch('tpu_inference.models.jax.qwen3_vl.Qwen3VLVisionTransformer', autospec=True) as MockVision, \
             patch('tpu_inference.models.jax.qwen3_vl.Qwen3ForCausalLM', autospec=True) as MockLM:
            mock_visual = MockVision.return_value
            mock_visual.dtype = mock_vllm_config.model_config.dtype
            mock_visual.config = mock_vllm_config.model_config.hf_config.vision_config
            mock_visual.spatial_merge_size = mock_vllm_config.model_config.hf_config.vision_config.spatial_merge_size

            model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
            model.visual = mock_visual
            model.language_model = MockLM.return_value
            yield model

    def test_embed_multimodal_none_pixel_values_returns_empty(
        self, model: Qwen3VLForConditionalGeneration
    ):
        """embed_multimodal with None pixel_values should return empty dict."""
        result = model.embed_multimodal(((1, 4, 4),), pixel_values=None)
        assert result == {}

    def test_embed_multimodal_empty_grid_returns_empty(
        self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey
    ):
        """embed_multimodal with empty grid should return empty dict."""
        vc = model.config.vision_config
        patch_dim = int(vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size)
        pixel_values = jax.random.normal(rng, (16, patch_dim)).astype(model.vllm_config.model_config.dtype)
        result = model.embed_multimodal((), pixel_values=pixel_values)
        assert result == {}

    def test_embed_multimodal_single_image(
        self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey
    ):
        """Single image grid should call visual encoder and return embeddings."""
        grid = (1, 4, 4)
        vc = model.config.vision_config
        patch_dim = int(vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size)
        num_patches = grid[0] * grid[1] * grid[2]
        n_output_tokens = _num_placeholders_for_grid(grid, vc.spatial_merge_size)
        pixel_values = jax.random.normal(rng, (num_patches, patch_dim)).astype(model.vllm_config.model_config.dtype)

        mock_vision_output = jnp.ones((n_output_tokens, vc.out_hidden_size))
        mock_deepstack = [jnp.ones((n_output_tokens, vc.out_hidden_size))]
        model.visual.return_value = (mock_vision_output, mock_deepstack)

        result = model.embed_multimodal((grid,), pixel_values=pixel_values)
        assert isinstance(result, dict)
        embeds = result.get("embeds", ())
        deepstack = result.get("deepstack")
        assert isinstance(embeds, tuple)
        assert len(embeds) == 1
        assert deepstack is not None
        assert len(deepstack) == 1
        model.visual.assert_called_once()

    @patch('tpu_inference.models.jax.qwen3_vl.merge_multimodal_embeddings')
    def test_get_input_embeddings_without_multimodal(
        self, mock_merge: MagicMock, model: Qwen3VLForConditionalGeneration, rng: PRNGKey
    ):
        """get_input_embeddings with None/empty multimodal should return text embeddings."""
        input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        mock_text_embeds = jnp.ones((5, model.config.hidden_size))
        model.language_model.embed = MagicMock(return_value=mock_text_embeds)

        # Test with None
        result = model.get_input_embeddings(input_ids, None)
        np.testing.assert_array_equal(np.array(result), np.array(mock_text_embeds))
        mock_merge.assert_not_called()

        # Test with empty array
        empty_mm = jnp.empty((0, model.config.hidden_size), dtype=model.vllm_config.model_config.dtype)
        result = model.get_input_embeddings(input_ids, empty_mm)
        np.testing.assert_array_equal(np.array(result), np.array(mock_text_embeds))
        mock_merge.assert_not_called()

    @patch('tpu_inference.models.jax.qwen3_vl.merge_multimodal_embeddings')
    def test_get_input_embeddings_with_multimodal(
        self, mock_merge: MagicMock, model: Qwen3VLForConditionalGeneration, rng: PRNGKey
    ):
        """get_input_embeddings with multimodal should call merge function."""
        input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        mock_text_embeds = jnp.ones((5, model.config.hidden_size))
        model.language_model.embed = MagicMock(return_value=mock_text_embeds)

        mm_embeds = jnp.ones((3, model.config.hidden_size))
        mock_merged = jnp.ones((5, model.config.hidden_size))
        mock_merge.return_value = mock_merged

        result = model.get_input_embeddings(input_ids, mm_embeds)
        np.testing.assert_array_equal(np.array(result), np.array(mock_merged))
        mock_merge.assert_called_once_with(
            input_ids, mock_text_embeds, mm_embeds,
            [model.config.image_token_id, model.config.video_token_id])

    def test_embed_input_ids_delegates_to_get_input_embeddings(
        self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey
    ):
        """embed_input_ids should delegate to get_input_embeddings."""
        input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        mock_embeds = jnp.ones((5, model.config.hidden_size))

        with patch.object(model, 'get_input_embeddings', return_value=mock_embeds) as mock_get:
            result = model.embed_input_ids(input_ids)
            np.testing.assert_array_equal(np.array(result), np.array(mock_embeds))
            mock_get.assert_called_once_with(input_ids, None)

    def test_call_delegates_to_language_model(
        self, model: Qwen3VLForConditionalGeneration, rng: PRNGKey
    ):
        """Model call should delegate to language model."""
        kv_caches = [MagicMock()]
        input_ids = jax.random.randint(rng, (10,), 0, model.config.vocab_size)
        attn_meta = MagicMock(spec=AttentionMetadata)
        mock_lm_output = ([MagicMock()], jnp.ones((10, model.config.hidden_size)), [])
        model.language_model.return_value = mock_lm_output

        new_kvs, x, aux_hidden_states = model(kv_caches, input_ids, attn_meta)
        model.language_model.assert_called_once()
        assert len(new_kvs) == 1
        assert x.shape == (10, model.config.hidden_size)

    def test_compute_logits_delegates_to_language_model(
        self, model: Qwen3VLForConditionalGeneration
    ):
        """compute_logits should delegate to language model."""
        hidden_states = jnp.ones((10, model.config.hidden_size))
        mock_logits = jnp.ones((10, model.config.vocab_size))
        model.language_model.compute_logits = MagicMock(return_value=mock_logits)

        logits = model.compute_logits(hidden_states)
        np.testing.assert_array_equal(np.array(logits), np.array(mock_logits))
        model.language_model.compute_logits.assert_called_once_with(hidden_states)

    @patch('tpu_inference.models.jax.qwen3_vl.load_hf_weights')
    def test_load_weights(
        self, mock_load_weights: MagicMock, model: Qwen3VLForConditionalGeneration,
        mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """load_weights should call load_hf_weights with proper args."""
        model.load_weights(rng)
        mock_load_weights.assert_called_once()
        kwargs = mock_load_weights.call_args.kwargs
        assert kwargs['vllm_config'] == mock_vllm_config
        assert kwargs['model'] is model
        assert kwargs['mesh'] is mesh


class TestServingIntegration:
    """Integration tests that create actual model instances."""

    def test_text_only_forward_is_causal(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
        input_ids_1 = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.int32)
        input_ids_2 = jnp.array([1, 2, 3, 4, 5, 7], dtype=jnp.int32)

        attn_meta = _make_attention_metadata(input_ids_1.shape[0])

        kv_caches_1 = _make_kv_caches(model, mesh)
        kv_caches_2 = _make_kv_caches(model, mesh)

        _, out_1, _ = model(kv_caches_1, input_ids_1, attn_meta)
        _, out_2, _ = model(kv_caches_2, input_ids_2, attn_meta)
        np.testing.assert_allclose(
            np.array(out_1)[:-1, :], np.array(out_2)[:-1, :], rtol=0, atol=0
        )

    def test_get_mrope_input_positions_wrapper_signature_and_slicing(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)

        grid = (1, 4, 4)
        n_img = _num_placeholders_for_grid(grid, model.spatial_merge_size)
        tokens = [1, model.vision_start_token_id] + [model.image_token_id] * n_img + [2, 3]

        positions, delta = model.get_mrope_input_positions(
            input_tokens=tokens,
            hf_config=model.config,
            image_grid_thw=[grid],
            video_grid_thw=None,
            context_len=2,
            seq_len=len(tokens) - 1,
            audio_feature_lengths=[0],
            use_audio_in_video=False,
        )
        assert positions.shape == (3, (len(tokens) - 1) - 2)
        assert isinstance(delta, int)

    def test_vision_encoder_and_embedding_merge_end_to_end(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)

        img_grid = (1, 4, 4)
        vid_grid = (2, 4, 4)
        n_img = _num_placeholders_for_grid(img_grid, model.spatial_merge_size)
        n_vid = _num_placeholders_for_grid(vid_grid, model.spatial_merge_size)

        # Vision encoder inputs: patchified pixels with shape (sum(t*h*w), patch_dim).
        vc = model.config.vision_config
        patch_dim = int(vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size)
        total_patches = int(img_grid[0] * img_grid[1] * img_grid[2] + vid_grid[0] * vid_grid[1] * vid_grid[2])
        pixel_values = jax.random.normal(rng, (total_patches, patch_dim)).astype(model.vllm_config.model_config.dtype)

        mm_result = model.embed_multimodal(
            (img_grid, vid_grid), pixel_values=pixel_values)
        assert isinstance(mm_result, dict)
        embeds = mm_result.get("embeds", ())
        deepstack = mm_result.get("deepstack")
        assert isinstance(embeds, tuple)
        assert len(embeds) == 2
        assert deepstack is not None
        assert len(deepstack) == 2
        assert len(deepstack[0]) == len(deepstack[1]) == 1
        assert deepstack[0][0].shape[0] == n_img
        assert deepstack[1][0].shape[0] == n_vid

        mm_flat = jnp.concatenate(embeds, axis=0)
        assert mm_flat.shape[0] == (n_img + n_vid)

        # Build a realistic placeholder sequence: vision_start + placeholders per item.
        input_ids_list = (
            [11, model.vision_start_token_id]
            + [model.image_token_id] * n_img
            + [12, model.vision_start_token_id]
            + [model.video_token_id] * n_vid
            + [13]
        )
        input_ids = jnp.array(input_ids_list, dtype=jnp.int32)

        # Merge multimodal embeddings into text embeddings.
        base_text = model.language_model.embed(input_ids)
        merged = model.get_input_embeddings(input_ids, mm_flat)
        assert merged.shape == base_text.shape

        placeholder_mask = (np.array(input_ids) == model.image_token_id) | (
            np.array(input_ids) == model.video_token_id
        )
        merged_np = np.array(merged)
        base_np = np.array(base_text)
        mm_np = np.array(mm_flat)

        # Placeholders are overwritten in-order.
        np.testing.assert_allclose(merged_np[placeholder_mask], mm_np, rtol=0, atol=0)
        # Non-placeholders remain identical to base text embeddings.
        np.testing.assert_allclose(merged_np[~placeholder_mask], base_np[~placeholder_mask], rtol=0, atol=0)

        # The serving wrapper should produce positions compatible with apply_rope's MRoPE path.
        positions_3d, _ = model.get_mrope_input_positions(
            input_tokens=input_ids_list,
            hf_config=model.config,
            image_grid_thw=[img_grid],
            video_grid_thw=[vid_grid],
            context_len=0,
            seq_len=len(input_ids_list),
        )
        assert positions_3d.shape == (3, len(input_ids_list))

        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
        q = jax.random.normal(rng, (len(input_ids_list), model.config.num_attention_heads, head_dim)).astype(jnp.bfloat16)
        q_rot = apply_rope(
            q,
            positions_3d,
            head_dim=head_dim,
            rope_theta=float(model.config.rope_theta),
            rope_scaling=getattr(model.config, "rope_scaling", None),
        )
        assert q_rot.shape == q.shape

    def test_mrope_wrapper_context_len_slicing(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """Test context_len and seq_len slicing behavior in wrapper."""
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # Full positions
        positions_full, _ = model.get_mrope_input_positions(
            input_tokens=tokens,
            hf_config=model.config,
            image_grid_thw=None,
            video_grid_thw=None,
            context_len=0,
            seq_len=len(tokens),
        )
        assert positions_full.shape == (3, 8)

        # Sliced positions: context_len=2, seq_len=6 -> positions[2:6]
        positions_sliced, _ = model.get_mrope_input_positions(
            input_tokens=tokens,
            hf_config=model.config,
            image_grid_thw=None,
            video_grid_thw=None,
            context_len=2,
            seq_len=6,
        )
        assert positions_sliced.shape == (3, 4)
        np.testing.assert_array_equal(
            np.array(positions_sliced), np.array(positions_full[:, 2:6])
        )

    def test_mrope_wrapper_seq_len_none_uses_full_length(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """When seq_len is None, should use full token length."""
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
        tokens = [1, 2, 3, 4]
        positions, _ = model.get_mrope_input_positions(
            input_tokens=tokens,
            hf_config=model.config,
            image_grid_thw=None,
            video_grid_thw=None,
            context_len=0,
            seq_len=None,
        )
        # seq_len=None means slice [:, 0:None] which is the full array
        assert positions.shape == (3, 4)

    def test_kv_cache_updates_after_forward(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """KV cache should be updated after a forward pass."""
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
        input_ids = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        attn_meta = _make_attention_metadata(input_ids.shape[0])

        kv_caches = _make_kv_caches(model, mesh)
        kv_caches = [cache * 0 for cache in kv_caches]
        before_norm = np.array(jnp.linalg.norm(kv_caches[0]))

        kv_caches, _, _ = model(kv_caches, input_ids, attn_meta)
        after_norm = np.array(jnp.linalg.norm(kv_caches[0]))

        assert before_norm == 0
        assert after_norm > 0

    def test_deepstack_injection_changes_placeholder_positions(
        self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """DeepStack embeddings should affect outputs at placeholder positions."""
        model = Qwen3VLForConditionalGeneration(mock_vllm_config, rng, mesh)
        grid = (1, 4, 4)
        n_img = _num_placeholders_for_grid(grid, model.spatial_merge_size)
        input_ids = jnp.array(
            [1, model.vision_start_token_id] + [model.image_token_id] * n_img + [2],
            dtype=jnp.int32,
        )
        attn_meta = _make_attention_metadata(input_ids.shape[0])

        deepstack = [
            jnp.full((n_img, model.config.hidden_size), 0.5,
                     dtype=model.vllm_config.model_config.dtype)
        ]
        inputs_embeds = model.get_input_embeddings(input_ids, None)

        kv_caches_base = _make_kv_caches(model, mesh)
        kv_caches_deep = _make_kv_caches(model, mesh)
        _, out_base, _ = model(kv_caches_base, input_ids, attn_meta, inputs_embeds)
        _, out_deep, _ = model(kv_caches_deep, input_ids, attn_meta,
                               inputs_embeds, deepstack)

        diff = np.abs(np.array(out_deep - out_base))
        placeholder_mask = np.array(input_ids) == model.image_token_id
        assert diff[placeholder_mask].max() > 0
