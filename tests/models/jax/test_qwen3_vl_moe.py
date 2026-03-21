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
from tpu_inference.models.jax.qwen3_vl_moe import (
    Qwen3VLMoeDecoderLayer,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeForConditionalGeneration,
)
from tpu_inference.runner.kv_cache import create_kv_caches


# --- Configuration Mocking ---
class MockVisionConfig:
    """Mock vision config for Qwen3VL MoE testing."""

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
    """A mock ModelConfig for testing the Qwen3 VL MoE model."""

    def __init__(self, hf_config: Qwen3Config, dtype: jnp.dtype):
        self.hf_config = hf_config
        self.dtype = dtype
        self.model = "mock_qwen3_vl_moe"
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


def _make_hf_config(
    tie_word_embeddings: bool = False,
    num_experts: int = 4,
    vision_config: MockVisionConfig | None = None,
) -> Qwen3Config:
    """Build a Qwen3Config with MoE and VL attributes attached."""
    if vision_config is None:
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
    hf_config.image_token_id = 100
    hf_config.video_token_id = 101
    hf_config.vision_start_token_id = 102
    hf_config.vision_config = vision_config
    hf_config.rope_scaling = {"mrope_section": [4, 2, 2]}
    # MoE-specific
    hf_config.num_experts = num_experts
    hf_config.num_experts_per_tok = 2
    hf_config.moe_intermediate_size = 32
    hf_config.decoder_sparse_step = 1
    hf_config.mlp_only_layers = []
    return hf_config


class MockMoeVllmConfig:
    """A mock VllmConfig for testing the Qwen3 VL MoE model."""

    def __init__(
        self,
        tie_word_embeddings: bool = False,
        num_experts: int = 4,
    ):
        hf_config = _make_hf_config(
            tie_word_embeddings=tie_word_embeddings,
            num_experts=num_experts,
        )
        self.model_config = MockModelConfig(hf_config, jnp.bfloat16)
        self.cache_config = MagicMock(spec=CacheConfig)
        self.cache_config.cache_dtype = "auto"
        self.quant_config = None
        self.load_config = MagicMock()
        self.additional_config = {}


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


def _make_kv_caches(model: Qwen3VLMoeForConditionalGeneration,
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
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices())
    return Mesh(devices.reshape((len(devices), 1, 1)),
                axis_names=('data', 'attn_dp', 'model'))


@pytest.fixture
def rng() -> PRNGKey:
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_vllm_config() -> MockMoeVllmConfig:
    """Config with MoE experts for unit tests (mocked components)."""
    return MockMoeVllmConfig()


@pytest.fixture
def dense_vllm_config() -> MockMoeVllmConfig:
    """Config with num_experts=0 for integration tests (real dense layers)."""
    return MockMoeVllmConfig(num_experts=0)


@pytest.fixture
def rngs(rng: PRNGKey) -> nnx.Rngs:
    return nnx.Rngs(params=rng)


@pytest.fixture
def hf_config(mock_vllm_config: MockMoeVllmConfig) -> Qwen3Config:
    return mock_vllm_config.model_config.hf_config


# --- MoE Layer Type Selection ---
class TestMoeLayerTypeSelection:
    """Tests for MoE vs dense MLP selection in Qwen3VLMoeDecoderLayer."""

    @patch('tpu_inference.models.jax.qwen3_vl_moe.Qwen3MoeSparseMoeBlock')
    def test_all_layers_use_moe_when_configured(
        self, mock_moe_cls, mock_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """All layers should be MoE when decoder_sparse_step=1 and num_experts>0."""
        hf_config = mock_vllm_config.model_config.hf_config
        rngs = nnx.Rngs(params=rng)
        for layer_idx in range(hf_config.num_hidden_layers):
            Qwen3VLMoeDecoderLayer(
                config=hf_config,
                dtype=jnp.bfloat16,
                rng=rngs,
                mesh=mesh,
                kv_cache_dtype="auto",
                quant_config=None,
                layer_idx=layer_idx,
                vllm_config=mock_vllm_config,
                prefix=f"layers.{layer_idx}",
            )
        assert mock_moe_cls.call_count == hf_config.num_hidden_layers

    def test_dense_fallback_for_mlp_only_layers(
        self, rng: PRNGKey, mesh: Mesh
    ):
        """Layers in mlp_only_layers should use dense MLP instead of MoE."""
        config = MockMoeVllmConfig(num_experts=4)
        hf_config = config.model_config.hf_config
        hf_config.mlp_only_layers = [0]
        rngs = nnx.Rngs(params=rng)
        layer = Qwen3VLMoeDecoderLayer(
            config=hf_config,
            dtype=jnp.bfloat16,
            rng=rngs,
            mesh=mesh,
            kv_cache_dtype="auto",
            quant_config=None,
            layer_idx=0,
            vllm_config=config,
            prefix="layers.0",
        )
        from tpu_inference.models.jax.qwen2 import Qwen2MLP
        assert isinstance(layer.mlp, Qwen2MLP)

    def test_dense_fallback_when_no_experts(self, rng: PRNGKey, mesh: Mesh):
        """Layers should use dense MLP when num_experts=0."""
        config = MockMoeVllmConfig(num_experts=0)
        hf_config = config.model_config.hf_config
        rngs = nnx.Rngs(params=rng)
        layer = Qwen3VLMoeDecoderLayer(
            config=hf_config,
            dtype=jnp.bfloat16,
            rng=rngs,
            mesh=mesh,
            kv_cache_dtype="auto",
            quant_config=None,
            layer_idx=0,
            vllm_config=config,
            prefix="layers.0",
        )
        from tpu_inference.models.jax.qwen2 import Qwen2MLP
        assert isinstance(layer.mlp, Qwen2MLP)

    def test_sparse_step_skips_moe(self, rng: PRNGKey, mesh: Mesh):
        """decoder_sparse_step=2 means only odd-indexed layers get MoE."""
        config = MockMoeVllmConfig(num_experts=4)
        hf_config = config.model_config.hf_config
        hf_config.decoder_sparse_step = 2
        hf_config.num_hidden_layers = 4
        rngs = nnx.Rngs(params=rng)

        from tpu_inference.models.jax.qwen2 import Qwen2MLP

        with patch('tpu_inference.models.jax.qwen3_vl_moe.Qwen3MoeSparseMoeBlock') as mock_moe:
            for idx in range(4):
                layer = Qwen3VLMoeDecoderLayer(
                    config=hf_config,
                    dtype=jnp.bfloat16,
                    rng=rngs,
                    mesh=mesh,
                    kv_cache_dtype="auto",
                    quant_config=None,
                    layer_idx=idx,
                    vllm_config=config,
                    prefix=f"layers.{idx}",
                )
                # (idx+1) % 2 == 0 -> MoE for idx=1,3; dense for idx=0,2
                if (idx + 1) % 2 == 0:
                    assert not isinstance(layer.mlp, Qwen2MLP)
                else:
                    assert isinstance(layer.mlp, Qwen2MLP)


# --- Mocked Unit Tests ---
class TestQwen3VLMoeForConditionalGeneration:
    """Tests for the Qwen3VL MoE model with mocked vision and language components."""

    @pytest.fixture
    def model(self, mock_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh):
        with patch('tpu_inference.models.jax.qwen3_vl_moe.Qwen3VLVisionTransformer', autospec=True) as MockVision, \
             patch('tpu_inference.models.jax.qwen3_vl_moe.Qwen3VLMoeTextModel', autospec=True) as MockLM:
            mock_visual = MockVision.return_value
            mock_visual.dtype = mock_vllm_config.model_config.dtype
            mock_visual.config = mock_vllm_config.model_config.hf_config.vision_config
            mock_visual.spatial_merge_size = mock_vllm_config.model_config.hf_config.vision_config.spatial_merge_size

            model = Qwen3VLMoeForConditionalGeneration(mock_vllm_config, rng, mesh)
            model.visual = mock_visual
            model.language_model = MockLM.return_value
            yield model

    def test_embed_multimodal_none_pixel_values_returns_empty(
        self, model: Qwen3VLMoeForConditionalGeneration
    ):
        result = model.embed_multimodal(((1, 4, 4),), pixel_values=None)
        assert result == {}

    def test_embed_multimodal_empty_grid_returns_empty(
        self, model: Qwen3VLMoeForConditionalGeneration, rng: PRNGKey
    ):
        vc = model.config.vision_config
        patch_dim = int(vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size)
        pixel_values = jax.random.normal(rng, (16, patch_dim)).astype(model.vllm_config.model_config.dtype)
        result = model.embed_multimodal((), pixel_values=pixel_values)
        assert result == {}

    def test_embed_multimodal_single_image(
        self, model: Qwen3VLMoeForConditionalGeneration, rng: PRNGKey
    ):
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

    @patch('tpu_inference.models.jax.qwen3_vl_moe.merge_multimodal_embeddings')
    def test_get_input_embeddings_without_multimodal(
        self, mock_merge: MagicMock, model: Qwen3VLMoeForConditionalGeneration, rng: PRNGKey
    ):
        input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        mock_text_embeds = jnp.ones((5, model.config.hidden_size))
        model.language_model.embed_tokens = MagicMock(
            return_value=mock_text_embeds)

        result = model.get_input_embeddings(input_ids, None)
        np.testing.assert_array_equal(np.array(result), np.array(mock_text_embeds))
        mock_merge.assert_not_called()

        empty_mm = jnp.empty((0, model.config.hidden_size), dtype=model.vllm_config.model_config.dtype)
        result = model.get_input_embeddings(input_ids, empty_mm)
        np.testing.assert_array_equal(np.array(result), np.array(mock_text_embeds))
        mock_merge.assert_not_called()

    @patch('tpu_inference.models.jax.qwen3_vl_moe.merge_multimodal_embeddings')
    def test_get_input_embeddings_with_multimodal(
        self, mock_merge: MagicMock, model: Qwen3VLMoeForConditionalGeneration, rng: PRNGKey
    ):
        input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        mock_text_embeds = jnp.ones((5, model.config.hidden_size))
        model.language_model.embed_tokens = MagicMock(
            return_value=mock_text_embeds)

        mm_embeds = jnp.ones((3, model.config.hidden_size))
        mock_merged = jnp.ones((5, model.config.hidden_size))
        mock_merge.return_value = mock_merged

        result = model.get_input_embeddings(input_ids, mm_embeds)
        np.testing.assert_array_equal(np.array(result), np.array(mock_merged))
        mock_merge.assert_called_once_with(
            input_ids, mock_text_embeds, mm_embeds,
            [model.config.image_token_id, model.config.video_token_id])

    def test_embed_input_ids_delegates_to_get_input_embeddings(
        self, model: Qwen3VLMoeForConditionalGeneration, rng: PRNGKey
    ):
        input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
        mock_embeds = jnp.ones((5, model.config.hidden_size))

        with patch.object(model, 'get_input_embeddings', return_value=mock_embeds) as mock_get:
            result = model.embed_input_ids(input_ids)
            np.testing.assert_array_equal(np.array(result), np.array(mock_embeds))
            mock_get.assert_called_once_with(input_ids, None)

    def test_call_delegates_to_language_model(
        self, model: Qwen3VLMoeForConditionalGeneration, rng: PRNGKey
    ):
        kv_caches = [MagicMock()]
        input_ids = jax.random.randint(rng, (10,), 0, model.config.vocab_size)
        attn_meta = MagicMock(spec=AttentionMetadata)
        mock_lm_output = ([MagicMock()], jnp.ones((10, model.config.hidden_size)))
        model.language_model.return_value = mock_lm_output

        new_kvs, x, aux_hidden_states = model(kv_caches, input_ids, attn_meta)
        model.language_model.assert_called_once()
        assert len(new_kvs) == 1
        assert x.shape == (10, model.config.hidden_size)

    def test_compute_logits_uses_lm_head_when_present(
        self, model: Qwen3VLMoeForConditionalGeneration
    ):
        hidden_states = jnp.ones((10, model.config.hidden_size))
        mock_logits = jnp.ones((10, model.config.vocab_size))
        model.lm_head = MagicMock(return_value=mock_logits)

        logits = model.compute_logits(hidden_states)
        np.testing.assert_array_equal(np.array(logits), np.array(mock_logits))
        model.lm_head.assert_called_once_with(hidden_states)

    @patch('tpu_inference.models.jax.qwen3_vl_moe.get_default_maps')
    @patch('tpu_inference.models.jax.qwen3_vl_moe.load_hf_weights')
    def test_load_weights(
        self, mock_load_weights: MagicMock, mock_get_default_maps: MagicMock,
        model: Qwen3VLMoeForConditionalGeneration,
        mock_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        with patch.object(model, '_load_moe_expert_weights') as mock_expert_load:
            model.load_weights(rng)
            mock_expert_load.assert_called_once()
            mock_load_weights.assert_called_once()
            kwargs = mock_load_weights.call_args.kwargs
            assert kwargs['vllm_config'] == mock_vllm_config
            assert kwargs['model'] is model
            assert kwargs['mesh'] is mesh
            # MoE load_weights filters out per-expert weights
            assert 'filter_regex' in kwargs


# --- Integration Tests (dense fallback, real layers) ---
class TestMoeServingIntegration:
    """Integration tests using num_experts=0 so all layers use dense MLP."""

    def test_text_only_forward_is_causal(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)
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

    def test_get_mrope_input_positions_wrapper_slicing(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)

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
        )
        assert positions.shape == (3, (len(tokens) - 1) - 2)
        assert isinstance(delta, int)

    def test_get_mrope_video_grid_expansion(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """MoE variant expands video (t,h,w) into t copies of (1,h,w)."""
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)

        vid_grid = (2, 4, 4)
        n_vid = _num_placeholders_for_grid(vid_grid, model.spatial_merge_size)
        tokens = (
            [model.vision_start_token_id]
            + [model.video_token_id] * n_vid
            + [1]
        )

        positions, _ = model.get_mrope_input_positions(
            input_tokens=tokens,
            hf_config=model.config,
            image_grid_thw=None,
            video_grid_thw=[vid_grid],
            context_len=0,
            seq_len=len(tokens),
        )
        assert positions.shape == (3, len(tokens))

        # Temporal positions within video should vary (t=2 -> 2 frames)
        t_vid = np.array(positions[0, 1 : 1 + n_vid])
        assert len(set(t_vid.tolist())) > 1

    def test_kv_cache_updates_after_forward(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)
        input_ids = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        attn_meta = _make_attention_metadata(input_ids.shape[0])

        kv_caches = _make_kv_caches(model, mesh)
        kv_caches = [cache * 0 for cache in kv_caches]
        before_norm = np.array(jnp.linalg.norm(kv_caches[0]))

        kv_caches, _, _ = model(kv_caches, input_ids, attn_meta)
        after_norm = np.array(jnp.linalg.norm(kv_caches[0]))

        assert before_norm == 0
        assert after_norm > 0

    def test_vision_encoder_and_embedding_merge_end_to_end(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)

        img_grid = (1, 4, 4)
        vid_grid = (2, 4, 4)
        n_img = _num_placeholders_for_grid(img_grid, model.spatial_merge_size)
        n_vid = _num_placeholders_for_grid(vid_grid, model.spatial_merge_size)

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

        mm_flat = jnp.concatenate(embeds, axis=0)
        assert mm_flat.shape[0] == (n_img + n_vid)

        input_ids_list = (
            [11, model.vision_start_token_id]
            + [model.image_token_id] * n_img
            + [12, model.vision_start_token_id]
            + [model.video_token_id] * n_vid
            + [13]
        )
        input_ids = jnp.array(input_ids_list, dtype=jnp.int32)

        base_text = model.language_model.embed_tokens(input_ids)
        merged = model.get_input_embeddings(input_ids, mm_flat)
        assert merged.shape == base_text.shape

        placeholder_mask = (np.array(input_ids) == model.image_token_id) | (
            np.array(input_ids) == model.video_token_id
        )
        merged_np = np.array(merged)
        base_np = np.array(base_text)
        mm_np = np.array(mm_flat)

        np.testing.assert_allclose(merged_np[placeholder_mask], mm_np, rtol=0, atol=0)
        np.testing.assert_allclose(merged_np[~placeholder_mask], base_np[~placeholder_mask], rtol=0, atol=0)

    def test_deepstack_injection_changes_placeholder_positions(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)
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

    def test_deepstack_injection_uses_global_layer_indices(
        self, dense_vllm_config: MockMoeVllmConfig, rng: PRNGKey, mesh: Mesh
    ):
        """DeepStack injection should use global decoder layer indices."""
        model = Qwen3VLMoeForConditionalGeneration(dense_vllm_config, rng, mesh)
        language_model = model.language_model
        hidden_size = model.config.hidden_size
        seq_len = 4

        base_hidden = jnp.zeros((seq_len, hidden_size),
                                dtype=model.vllm_config.model_config.dtype)
        language_model.start_layer = 1
        language_model.end_layer = 3
        language_model.embed_tokens = MagicMock(return_value=base_hidden)
        language_model.norm = MagicMock(side_effect=lambda x: x)
        language_model.layers = [
            MagicMock(side_effect=lambda kv, x, md: (kv, x))
            for _ in range(4)
        ]

        injected = []

        def record_injection(x, mask, visual_embeds):
            del mask
            injected.append(np.array(visual_embeds))
            return x

        language_model._inject_visual_features = MagicMock(
            side_effect=record_injection)

        kv_caches = [jnp.zeros((1,), dtype=jnp.int32) for _ in range(2)]
        attn_meta = _make_attention_metadata(seq_len)
        visual_mask = jnp.array([True, False, False, False], dtype=jnp.bool_)
        deepstack = [
            jnp.full((1, hidden_size), float(i),
                     dtype=model.vllm_config.model_config.dtype)
            for i in range(4)
        ]

        language_model(
            kv_caches=kv_caches,
            input_ids=jnp.arange(seq_len, dtype=jnp.int32),
            attention_metadata=attn_meta,
            inputs_embeds=base_hidden,
            visual_pos_mask=visual_mask,
            deepstack_visual_embeds=deepstack,
        )

        assert len(injected) == 2
        np.testing.assert_array_equal(injected[0], np.array(deepstack[1]))
        np.testing.assert_array_equal(injected[1], np.array(deepstack[2]))
