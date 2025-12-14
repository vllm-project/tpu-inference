"""Meaningful (non-mocked) tests for the Qwen3-VL JAX implementation.

These tests avoid `unittest.mock` and focus on the integration boundaries that
are easy to get wrong in serving:
- MRoPE position ID generation + serving-style wrapper signature.
- Vision encoder end-to-end on TPU (flash attention + segment IDs).
- Multimodal embedding merge correctness for placeholder tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh

# Optional deps required by the model code. Skip cleanly if not available.
pytest.importorskip("vllm")
pytest.importorskip("transformers")
from transformers import Qwen3Config

from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.models.jax.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    SegmentIds,
    apply_rotary_pos_emb_vision,
    generate_segment_ids_from_grid_thw,
    get_mrope_input_positions,
    pad_segment_ids_for_attention,
)


@dataclass(frozen=True)
class VisionTestConfig:
    hidden_size: int = 32
    intermediate_size: int = 64
    patch_size: int = 2
    image_size: int = 8
    temporal_patch_size: int = 1
    in_channels: int = 3
    spatial_merge_size: int = 2
    out_hidden_size: int = 64
    depth: int = 1
    num_heads: int = 4
    num_position_embeddings: int = 16  # 4x4 grid
    deepstack_visual_indexes: Tuple[int, ...] = (0,)
    tokens_per_second: float = 1.0


@dataclass(frozen=True)
class TestModelConfig:
    hf_config: Qwen3Config
    dtype: jnp.dtype

    def is_multimodal_model(self) -> bool:
        return True

    def get_hidden_size(self) -> int:
        return int(self.hf_config.hidden_size)

    def get_head_size(self) -> int:
        return int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)

    def get_vocab_size(self) -> int:
        return int(self.hf_config.vocab_size)


@dataclass(frozen=True)
class TestCacheConfig:
    cache_dtype: str = "auto"


@dataclass(frozen=True)
class TestVllmConfig:
    model_config: TestModelConfig
    cache_config: TestCacheConfig
    additional_config: dict


def _is_tpu() -> bool:
    """Check if running on TPU, compatible with Docker/Ray multihost serving."""
    try:
        devices = jax.devices()
        if not devices:
            return False
        # In Ray/Docker setups, jax.devices() returns the backend devices.
        # Check the default backend instead of individual device platform.
        return jax.default_backend() == "tpu"
    except Exception:
        return False


def _num_placeholders_for_grid(
    grid_thw: Tuple[int, int, int], spatial_merge_size: int
) -> int:
    t, h, w = grid_thw
    return int(t * (h // spatial_merge_size) * (w // spatial_merge_size))


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Creates a 1-device mesh to avoid head/data sharding constraints."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, -1))
    with Mesh(device_mesh, axis_names=("data", "attn_dp", "model")) as m:
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def hf_config() -> Qwen3Config:
    # Small config for test speed, but with Qwen3-style fields.
    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )

    # Multimodal special tokens must be within vocab_size for embedding lookups.
    cfg.image_token_id = 100
    cfg.video_token_id = 101
    cfg.vision_start_token_id = 102

    # Attach a minimal vision config object with the attributes used by the model.
    cfg.vision_config = VisionTestConfig()

    # Enable MRoPE branch in apply_rope when positions is (3, seq_len).
    # Keep mrope_section small enough for head_dim_original (hidden/heads=16).
    cfg.rope_scaling = {"mrope_section": [2, 2, 0]}

    return cfg


@pytest.fixture(scope="module")
def vllm_config(hf_config: Qwen3Config) -> TestVllmConfig:
    model_config = TestModelConfig(hf_config=hf_config, dtype=jnp.bfloat16)
    cache_config = TestCacheConfig(cache_dtype="auto")
    return TestVllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        additional_config={"enable_dynamic_image_sizes": False},
    )


@pytest.fixture
def rngs(rng: PRNGKey) -> nnx.Rngs:
    return nnx.Rngs(params=rng)


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
        # (1,4,4) -> 16 tokens (t*h*w), (2,4,4) -> 32 tokens.
        assert segment_ids.shape == (48,)
        np.testing.assert_array_equal(segment_ids[:16], np.ones(16, dtype=np.int32))
        np.testing.assert_array_equal(segment_ids[16:], np.full(32, 2, dtype=np.int32))

    def test_pad_segment_ids_for_attention(self):
        segment_ids = jnp.array([1, 1, 2, 2], dtype=jnp.int32)
        padded = pad_segment_ids_for_attention(segment_ids, padded_seq_len=8)
        assert isinstance(padded, SegmentIds)
        assert padded.q.shape == (1, 8)
        assert padded.kv.shape == (1, 8)
        np.testing.assert_array_equal(
            np.array(padded.q), np.array([[1, 1, 2, 2, 0, 0, 0, 0]], dtype=np.int32)
        )


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
            second_per_grid_ts=[2.0],  # make temporal positions clearly differ
            tokens_per_second=1.0,
        )
        assert positions.shape == (3, len(tokens))

        # Video temporal positions should not be constant when t>1.
        video_start = 1 + n_img + 1
        t_vid = np.array(positions[0, video_start : video_start + n_vid])
        assert len(set(t_vid.tolist())) > 1


class TestRopeInterface:
    def test_apply_rope_requires_mrope_section_for_3d_positions(self):
        seq_len = 8
        num_heads = 2
        head_dim = 16
        x = jnp.zeros((seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        positions = jnp.zeros((3, seq_len), dtype=jnp.int32)
        with pytest.raises(ValueError, match="mrope_section"):
            _ = apply_rope(x, positions, head_dim=head_dim, rope_scaling={})

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


class TestServingIntegration:
    def test_get_mrope_input_positions_wrapper_signature_and_slicing(
        self, vllm_config: TestVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLForConditionalGeneration(vllm_config, rng, mesh)

        grid = (1, 4, 4)
        n_img = _num_placeholders_for_grid(grid, model.spatial_merge_size)
        tokens = [1, model.vision_start_token_id] + [model.image_token_id] * n_img + [2, 3]

        positions, delta = model.get_mrope_input_positions(
            input_tokens=tokens,
            hf_config=model.config,
            image_grid_thw=[grid],
            video_grid_thw=None,
            second_per_grid_ts=None,
            context_len=2,
            seq_len=len(tokens) - 1,
            audio_feature_lengths=[0],
            use_audio_in_video=False,
        )
        assert positions.shape == (3, (len(tokens) - 1) - 2)
        assert isinstance(delta, int)

    @pytest.mark.skipif(not _is_tpu(), reason="TPU required for flash attention vision encoder")
    def test_vision_encoder_and_embedding_merge_end_to_end(
        self, vllm_config: TestVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLForConditionalGeneration(vllm_config, rng, mesh)

        img_grid = (1, 4, 4)
        vid_grid = (2, 4, 4)
        n_img = _num_placeholders_for_grid(img_grid, model.spatial_merge_size)
        n_vid = _num_placeholders_for_grid(vid_grid, model.spatial_merge_size)

        # Vision encoder inputs: patchified pixels with shape (sum(t*h*w), patch_dim).
        vc = model.config.vision_config
        patch_dim = int(vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size)
        total_patches = int(img_grid[0] * img_grid[1] * img_grid[2] + vid_grid[0] * vid_grid[1] * vid_grid[2])
        pixel_values = jax.random.normal(rng, (total_patches, patch_dim)).astype(model.vllm_config.model_config.dtype)

        embeds = model.get_multimodal_embeddings((img_grid, vid_grid), pixel_values=pixel_values)
        assert isinstance(embeds, tuple)
        assert len(embeds) == 2
        assert model._deepstack_cache is not None

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
            second_per_grid_ts=[1.0],
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
