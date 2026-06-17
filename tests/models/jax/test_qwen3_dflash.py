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
"""Unit tests for the Qwen3 DFlash speculative decoding draft model on JAX/TPU."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen3Config

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.qwen3_dflash import (
    Qwen3DFlashAttention, Qwen3DFlashDecoderLayer, Qwen3DFlashForCausalLM,
    _build_target_layer_ids, _get_dflash_target_layer_ids)


# ----- Existing Tests for Layer IDs -----
def test_build_target_layer_ids_default_layout():
    assert _build_target_layer_ids(32, 1) == [16]
    assert _build_target_layer_ids(32, 4) == [1, 10, 20, 29]


def test_get_target_layer_ids_prefers_explicit_config():
    cfg = SimpleNamespace(
        dflash_config={"target_layer_ids": [2, 6, 10]},
        num_target_layers=32,
        num_hidden_layers=3,
    )
    assert _get_dflash_target_layer_ids(cfg, 32) == [2, 6, 10]


def test_get_target_layer_ids_fallback():
    cfg = SimpleNamespace(
        dflash_config=None,
        num_target_layers=32,
        num_hidden_layers=3,
    )
    assert _get_dflash_target_layer_ids(cfg, 32) == [1, 15, 29]


# ----- Fixtures & Mocks for Model Component Tests -----
@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device for testing."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def hf_configs():
    """Provides small dummy Qwen3Configs for draft and target models."""
    draft_cfg = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        head_dim=16,
        dflash_config={
            "mask_token_id": 0,
            "position_scheme": "incremental",
            "target_layer_ids": [0, 1],
        },
    )
    draft_cfg.rope_theta = 10000.0
    draft_cfg.rope_scaling = None

    target_cfg = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=4,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        head_dim=16,
    )
    target_cfg.rope_theta = 10000.0
    target_cfg.rope_scaling = None

    return draft_cfg, target_cfg


class MockVllmConfig:
    """Mock for VllmConfig containing the draft and target model configurations."""

    def __init__(self, draft_hf_config, target_hf_config):

        class MockDraftModelConfig:

            def __init__(self, hf_config):
                self.hf_config = hf_config
                self.is_multimodal_model = False

            def get_hidden_size(self):
                return self.hf_config.hidden_size

            def get_head_size(self):
                return getattr(
                    self.hf_config,
                    "head_dim",
                    self.hf_config.hidden_size //
                    self.hf_config.num_attention_heads,
                )

        class MockSpeculativeConfig:

            def __init__(self, hf_config):
                self.draft_model_config = MockDraftModelConfig(hf_config)
                self.num_speculative_tokens = 4

        class MockModelConfig:

            def __init__(self, hf_config):
                self.hf_config = hf_config
                self.dtype = jnp.bfloat16
                self.seed = 42
                self.model = "mock-model"

            def get_vocab_size(self):
                return self.hf_config.vocab_size

            def get_hidden_size(self):
                return self.hf_config.hidden_size

        class MockCacheConfig:

            def __init__(self):
                self.cache_dtype = "auto"

        class MockLoadConfig:

            def __init__(self):
                self.download_dir = None

        self.speculative_config = MockSpeculativeConfig(draft_hf_config)
        self.model_config = MockModelConfig(target_hf_config)
        self.cache_config = MockCacheConfig()
        self.load_config = MockLoadConfig()
        self.quant_config = None
        self.additional_config = {"dflash_attention_impl": "concat_dense"}


def dummy_attention(kv_cache, q, k, v, *args, **kwargs):
    """Dummy replacement for attention call."""
    del k, v, args, kwargs
    return kv_cache, q


def dummy_dflash_concat_attention(q, *args, **kwargs):
    """Dummy replacement for dflash_concat_attention."""
    del args, kwargs
    return q


def _make_attention_metadata(query_start_loc: list[int]) -> AttentionMetadata:
    """Helper to construct dummy AttentionMetadata."""
    query_start_loc = np.asarray(query_start_loc, dtype=np.int32)
    seq_lens = np.diff(query_start_loc)
    total_tokens = int(query_start_loc[-1])
    return AttentionMetadata(
        input_positions=jnp.arange(total_tokens, dtype=jnp.int32),
        block_tables=jnp.zeros((max(1, total_tokens), ), dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
        query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
        request_distribution=jnp.asarray([0, 0, len(seq_lens)],
                                         dtype=jnp.int32),
    )


# ----- Component tests -----
@pytest.mark.parametrize("dflash_impl", ["concat_dense", "additive_legacy"])
@patch(
    "tpu_inference.models.jax.qwen3_dflash.dflash_concat_attention",
    side_effect=dummy_dflash_concat_attention,
)
@patch(
    "tpu_inference.models.jax.qwen3_dflash.attention",
    side_effect=dummy_attention,
)
def test_qwen3_dflash_attention(mock_attn, mock_concat, dflash_impl,
                                hf_configs, mesh):
    """Verifies Qwen3DFlashAttention forward pass under different attention implementations."""
    draft_cfg, _ = hf_configs
    rng = nnx.Rngs(42)
    dtype = jnp.bfloat16

    with jax.set_mesh(mesh):
        attn = Qwen3DFlashAttention(
            config=draft_cfg,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype="auto",
            quant_config=None,
            dflash_attention_impl=dflash_impl,
            max_query_len=4,
        )

    T = 4
    hidden_size = draft_cfg.hidden_size
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    max_kv_len = 16

    hidden_states = jnp.ones((T, hidden_size), dtype=dtype)
    target_hidden = jnp.ones((T, hidden_size), dtype=dtype)
    kv_cache = jnp.zeros((T, num_heads, max_kv_len, head_dim), dtype=dtype)
    attention_metadata = _make_attention_metadata([0, T])

    with jax.set_mesh(mesh):
        new_kv_cache, output = attn(
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            target_hidden_states=target_hidden,
            attention_metadata=attention_metadata,
        )

    assert output.shape == (T, hidden_size)
    assert new_kv_cache.shape == (T, num_heads, max_kv_len, head_dim)

    if dflash_impl == "concat_dense":
        mock_concat.assert_called_once()
        mock_attn.assert_called_once()
    else:
        mock_concat.assert_not_called()
        mock_attn.assert_called_once()


@patch(
    "tpu_inference.models.jax.qwen3_dflash.dflash_concat_attention",
    side_effect=dummy_dflash_concat_attention,
)
@patch(
    "tpu_inference.models.jax.qwen3_dflash.attention",
    side_effect=dummy_attention,
)
def test_qwen3_dflash_decoder_layer(mock_attn, mock_concat, hf_configs, mesh):
    """Verifies Qwen3DFlashDecoderLayer integrates attention and MLP blocks."""
    draft_cfg, _ = hf_configs
    rng = nnx.Rngs(42)
    dtype = jnp.bfloat16

    with jax.set_mesh(mesh):
        layer = Qwen3DFlashDecoderLayer(
            config=draft_cfg,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype="auto",
            quant_config=None,
            dflash_attention_impl="concat_dense",
            max_query_len=4,
        )

    T = 4
    hidden_size = draft_cfg.hidden_size
    num_heads = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    max_kv_len = 16

    hidden_states = jnp.ones((T, hidden_size), dtype=dtype)
    target_hidden = jnp.ones((T, hidden_size), dtype=dtype)
    kv_cache = jnp.zeros((T, num_heads, max_kv_len, head_dim), dtype=dtype)
    attention_metadata = _make_attention_metadata([0, T])

    with jax.set_mesh(mesh):
        new_kv_cache, output = layer(
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            target_hidden_states=target_hidden,
            attention_metadata=attention_metadata,
        )

    assert output.shape == (T, hidden_size)
    assert new_kv_cache.shape == (T, num_heads, max_kv_len, head_dim)


@patch(
    "tpu_inference.models.jax.qwen3_dflash.dflash_concat_attention",
    side_effect=dummy_dflash_concat_attention,
)
@patch(
    "tpu_inference.models.jax.qwen3_dflash.attention",
    side_effect=dummy_attention,
)
def test_qwen3_dflash_for_causal_lm(mock_attn, mock_concat, hf_configs, mesh):
    """Validates full draft model's forward pass, logits calculation, and combined projection."""
    draft_cfg, target_cfg = hf_configs
    vllm_config = MockVllmConfig(draft_cfg, target_cfg)
    rng_key = jax.random.PRNGKey(0)

    with jax.set_mesh(mesh):
        model = Qwen3DFlashForCausalLM(
            vllm_config=vllm_config,
            rng_key=rng_key,
            mesh=mesh,
        )

    # Validate structure
    assert model.model.embed_tokens.embedding.shape == (
        target_cfg.vocab_size,
        draft_cfg.hidden_size,
    )
    assert len(model.model.layers) == draft_cfg.num_hidden_layers

    # Run forward pass
    T = 4
    hidden_size = draft_cfg.hidden_size
    num_heads = model.model.layers[0].self_attn.num_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    max_kv_len = 16

    input_ids = jnp.ones((T, ), dtype=jnp.int32)
    target_hidden = jnp.ones((T, hidden_size), dtype=jnp.bfloat16)
    attention_metadata = _make_attention_metadata([0, T])

    kv_caches = [
        jnp.zeros((T, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16)
        for _ in range(draft_cfg.num_hidden_layers)
    ]

    with jax.set_mesh(mesh):
        new_kv_caches, hidden_states, extra = model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            target_hidden_states=target_hidden,
            attention_metadata=attention_metadata,
        )

    assert len(new_kv_caches) == draft_cfg.num_hidden_layers
    assert hidden_states.shape == (T, hidden_size)
    assert len(extra) == 1
    assert extra[0].shape == (T, hidden_size)

    # Test compute_logits
    logits = model.compute_logits(hidden_states)
    assert logits.shape == (T, target_cfg.vocab_size)

    # Test combine_hidden_states
    # target_hidden_size = 64, target_layer_ids count = 2, combined shape = (T, 128)
    combined_input = jnp.ones((T, 128), dtype=jnp.bfloat16)
    with jax.set_mesh(mesh):
        combined_output = model.combine_hidden_states(combined_input)
    assert combined_output.shape == (T, hidden_size)


def test_qwen3_dflash_weight_loader(hf_configs, mesh):
    """Verifies that Qwen3DFlashWeightLoader correctly initializes and orchestrates weight loading."""
    draft_cfg, target_cfg = hf_configs
    vllm_config = MockVllmConfig(draft_cfg, target_cfg)
    rng_key = jax.random.PRNGKey(0)

    with jax.set_mesh(mesh):
        model = Qwen3DFlashForCausalLM(
            vllm_config=vllm_config,
            rng_key=rng_key,
            mesh=mesh,
        )

    with patch(
            "tpu_inference.models.jax.qwen3_dflash.load_hf_weights"
    ) as mock_load_hf, patch(
            "tpu_inference.models.jax.utils.weight_utils.model_weights_generator"
    ) as mock_generator:
        with jax.set_mesh(mesh):
            model.load_weights(rng_key)
        mock_load_hf.assert_called_once()
        mock_generator.assert_called_once()

        # Verify loader config propagation
        called_args, called_kwargs = mock_load_hf.call_args
        assert called_kwargs["vllm_config"] == vllm_config
        assert called_kwargs["model"] == model
        assert called_kwargs["mesh"] == mesh
        assert called_kwargs["is_draft_model"] is True
