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
"""Unit tests for the DFlash speculative decoding draft model on JAX/TPU."""
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen3Config

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.dflash import (DFlashAttention,
                                             DFlashDecoderLayer,
                                             DFlashForCausalLM, DFlashMLP)


def _make_attention_metadata(query_start_loc: list[int],
                             total_tokens: int) -> AttentionMetadata:
    """Helper to construct dummy AttentionMetadata."""
    query_start_loc = np.asarray(query_start_loc, dtype=np.int32)
    seq_lens = np.diff(query_start_loc)
    return AttentionMetadata(
        input_positions=jnp.arange(total_tokens, dtype=jnp.int32),
        block_tables=jnp.zeros((max(1, total_tokens), ), dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
        query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
        request_distribution=jnp.asarray([0, 0, len(seq_lens)],
                                         dtype=jnp.int32),
    )


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
def hf_config():
    """Provides a small dummy Qwen3Config for fast testing."""
    config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        head_dim=16,
        target_hidden_size=64,
        num_target_layers=2,
        dflash_config={
            "mask_token_id": 0,
            "position_scheme": "incremental",
            "target_layer_ids": [0, 1],
        },
    )
    config.rope_theta = 10000.0
    config.rope_scaling = None
    return config


class MockVllmConfig:
    """Mock for VllmConfig containing the draft model configuration."""

    def __init__(self, hf_config):

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

        class MockModelConfig:

            def __init__(self, hf_config):
                self.hf_config = hf_config
                self.seed = 42
                self.model = "mock-model"

            def get_vocab_size(self):
                return self.hf_config.vocab_size

            def get_hidden_size(self):
                return self.hf_config.hidden_size

        class MockLoadConfig:

            def __init__(self):
                self.download_dir = None

        class MockParallelConfig:

            def __init__(self):
                self.tensor_parallel_size = 1

        self.speculative_config = MockSpeculativeConfig(hf_config)
        self.model_config = MockModelConfig(hf_config)
        self.load_config = MockLoadConfig()
        self.parallel_config = MockParallelConfig()


def dummy_attention(kv_cache, q, k, v, *args, **kwargs):
    """Dummy replacement for attention to allow CPU-only unit testing."""
    del k, v, args, kwargs
    return kv_cache, q


@patch(
    "tpu_inference.models.jax.dflash.attention",
    side_effect=dummy_attention,
)
def test_dflash_attention(mock_attn, hf_config, mesh):
    """Verifies that DFlashAttention runs correctly and writes to the KV cache."""
    rng = nnx.Rngs(42)
    dtype = jnp.bfloat16
    with jax.set_mesh(mesh):
        attn = DFlashAttention(
            config=hf_config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
        )
    T_noise = 4
    T_padded = 8
    hidden_size = hf_config.hidden_size
    x_noise = jnp.ones((T_noise, hidden_size), dtype=dtype)
    target_hidden = jnp.ones((T_padded, hidden_size), dtype=dtype)
    target_query_start_loc = jnp.array([0, T_padded], dtype=jnp.int32)
    target_positions = jnp.arange(T_padded, dtype=jnp.int32)
    # Mock attention metadata
    attention_metadata = _make_attention_metadata([0, T_noise], T_noise)
    # kv_cache shape: [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache = jnp.zeros((10, 16, 4, 1, attn.head_dim), dtype=dtype)
    with jax.set_mesh(mesh):
        output, new_kv_cache = attn(
            x_noise=x_noise,
            target_hidden=target_hidden,
            attention_metadata=attention_metadata,
            target_query_start_loc=target_query_start_loc,
            target_positions=target_positions,
            kv_cache=kv_cache,
        )
    assert output.shape == (T_noise, hidden_size)
    assert new_kv_cache.shape == kv_cache.shape
    assert mock_attn.call_count == 2  # Step 1 (target) and Step 2 (noise)


def test_dflash_mlp(hf_config, mesh):
    """Tests the DFlashMLP layer's output shape and forward pass."""
    rng = nnx.Rngs(42)
    dtype = jnp.bfloat16
    with jax.set_mesh(mesh):
        mlp = DFlashMLP(config=hf_config, dtype=dtype, rng=rng)
        T = 5
        x = jnp.ones((T, hf_config.hidden_size), dtype=dtype)
        output = mlp(x)
    assert output.shape == (T, hf_config.hidden_size)


@patch(
    "tpu_inference.models.jax.dflash.attention",
    side_effect=dummy_attention,
)
def test_dflash_decoder_layer(mock_attn, hf_config, mesh):
    """Verifies that the DFlashDecoderLayer forward pass completes successfully."""
    rng = nnx.Rngs(42)
    dtype = jnp.bfloat16
    with jax.set_mesh(mesh):
        layer = DFlashDecoderLayer(
            config=hf_config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
        )
    T_noise = 4
    T_padded = 8
    hidden_size = hf_config.hidden_size
    x_noise = jnp.ones((T_noise, hidden_size), dtype=dtype)
    target_hidden = jnp.ones((T_padded, hidden_size), dtype=dtype)
    target_query_start_loc = jnp.array([0, T_padded], dtype=jnp.int32)
    target_positions = jnp.arange(T_padded, dtype=jnp.int32)
    attention_metadata = _make_attention_metadata([0, T_noise], T_noise)
    kv_cache = jnp.zeros((10, 16, 4, 1, layer.self_attn.head_dim), dtype=dtype)
    with jax.set_mesh(mesh):
        output, new_kv_cache = layer(
            x=x_noise,
            target_hidden=target_hidden,
            attention_metadata=attention_metadata,
            target_query_start_loc=target_query_start_loc,
            target_positions=target_positions,
            kv_cache=kv_cache,
        )
    assert output.shape == (T_noise, hidden_size)
    assert new_kv_cache.shape == kv_cache.shape
    assert mock_attn.call_count == 2


@patch(
    "tpu_inference.models.jax.dflash.attention",
    side_effect=dummy_attention,
)
def test_dflash_for_causal_lm(mock_attn, hf_config, mesh):
    """Validates the full draft model's forward pass, logits calculation, and state projection."""
    vllm_config = MockVllmConfig(hf_config)
    rng_key = jax.random.PRNGKey(0)
    with jax.set_mesh(mesh):
        model = DFlashForCausalLM(
            vllm_config=vllm_config,
            rng_key=rng_key,
            mesh=mesh,
        )
    assert model.model.embed_tokens.embedding.shape == (
        hf_config.vocab_size,
        hf_config.hidden_size,
    )
    assert len(model.model.layers) == hf_config.num_hidden_layers
    T_noise = 3
    T_padded = 6
    hidden_size = hf_config.hidden_size
    head_dim = model.model.layers[0].self_attn.head_dim
    input_ids = jnp.ones((T_noise, ), dtype=jnp.int32)
    ctx_hidden = jnp.ones((T_padded, hidden_size), dtype=jnp.bfloat16)
    target_query_start_loc = jnp.array([0, T_padded], dtype=jnp.int32)
    target_positions = jnp.arange(T_padded, dtype=jnp.int32)
    target_hidden_states = (ctx_hidden, target_query_start_loc,
                            target_positions)
    attention_metadata = _make_attention_metadata([0, T_noise], T_noise)
    # List of kv_cache tensors, one per layer
    kv_caches = [
        jnp.zeros((10, 16, 4, 1, head_dim), dtype=jnp.bfloat16)
        for _ in range(hf_config.num_hidden_layers)
    ]
    with jax.set_mesh(mesh):
        new_kv_caches, hidden_states, extra, _ = model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            attention_metadata=attention_metadata,
        )
    assert len(new_kv_caches) == hf_config.num_hidden_layers
    assert hidden_states.shape == (T_noise, hidden_size)
    assert extra == []
    # Test compute_logits
    logits = model.compute_logits(hidden_states)
    assert logits.shape == (T_noise, hf_config.vocab_size)
    # Test combine_hidden_states
    combined_input = jnp.ones(
        (
            T_noise,
            hf_config.num_target_layers * hf_config.target_hidden_size,
        ),
        dtype=jnp.bfloat16,
    )
    with jax.set_mesh(mesh):
        combined_output = model.combine_hidden_states(combined_input)
    assert combined_output.shape == (T_noise, hidden_size)


def test_dflash_weight_loader(hf_config, mesh):
    """Verifies that the DFlashWeightLoader maps and delegates weight loading correctly."""
    vllm_config = MockVllmConfig(hf_config)
    rng_key = jax.random.PRNGKey(0)
    with jax.set_mesh(mesh):
        model = DFlashForCausalLM(
            vllm_config=vllm_config,
            rng_key=rng_key,
            mesh=mesh,
        )
    with patch(
            "tpu_inference.models.jax.dflash.load_hf_weights"
    ) as mock_load_hf, patch(
            "tpu_inference.models.jax.utils.weight_utils.model_weights_generator"
    ) as mock_generator:
        with jax.set_mesh(mesh):
            model.load_weights(rng_key)
        mock_load_hf.assert_called_once()
        mock_generator.assert_called_once()
        called_args, called_kwargs = mock_load_hf.call_args
        assert called_kwargs["vllm_config"] == vllm_config
        assert called_kwargs["model"] == model
        assert called_kwargs["mesh"] == mesh
        assert called_kwargs["is_draft_model"] is True
