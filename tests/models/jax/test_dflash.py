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

from tpu_inference.models.jax.dflash import (DFlashAttention,
                                             DFlashDecoderLayer,
                                             DFlashForCausalLM, DFlashMLP)


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
    # Explicitly set rope_theta and rope_scaling on the config object to satisfy
    # direct attribute accesses in dflash.py.
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

            def __init__(self):
                self.seed = 42
                self.model = "mock-model"

        class MockLoadConfig:

            def __init__(self):
                self.download_dir = None

        self.speculative_config = MockSpeculativeConfig(hf_config)
        self.model_config = MockModelConfig()
        self.load_config = MockLoadConfig()


def dummy_flash_attention(
    q,
    k,
    v,
    segment_ids=None,
    causal=False,
    sm_scale=1.0,
    block_sizes=None,
    vmem_limit_bytes=None,
):
    """Dummy replacement for flash_attention to allow CPU-only unit testing."""
    del k, v, segment_ids, causal, sm_scale, block_sizes, vmem_limit_bytes
    return jnp.zeros_like(q)


@patch(
    "tpu_inference.models.jax.dflash.flash_attention",
    side_effect=dummy_flash_attention,
)
def test_dflash_attention(mock_fa, hf_config, mesh):
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
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    max_kv_len = 32

    x_noise = jnp.ones((T_noise, hidden_size), dtype=dtype)
    target_hidden = jnp.ones((T_padded, hidden_size), dtype=dtype)
    noise_positions = jnp.arange(T_noise, dtype=jnp.int32)
    ctx_positions = jnp.arange(T_padded, dtype=jnp.int32)
    kv_cache_k = jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=dtype)
    kv_cache_v = jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=dtype)
    cache_len = jnp.array(0, dtype=jnp.int32)
    actual_ctx_count = jnp.array(6, dtype=jnp.int32)

    with jax.set_mesh(mesh):
        output, new_k, new_v = attn(
            x_noise=x_noise,
            target_hidden=target_hidden,
            noise_positions=noise_positions,
            ctx_positions=ctx_positions,
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            cache_len=cache_len,
            actual_ctx_count=actual_ctx_count,
        )

    assert output.shape == (T_noise, hidden_size)
    assert new_k.shape == (1, num_heads, max_kv_len, head_dim)
    assert new_v.shape == (1, num_heads, max_kv_len, head_dim)
    mock_fa.assert_called_once()


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
    "tpu_inference.models.jax.dflash.flash_attention",
    side_effect=dummy_flash_attention,
)
def test_dflash_decoder_layer(mock_fa, hf_config, mesh):
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
    num_heads = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    max_kv_len = 32

    x_noise = jnp.ones((T_noise, hidden_size), dtype=dtype)
    target_hidden = jnp.ones((T_padded, hidden_size), dtype=dtype)
    noise_positions = jnp.arange(T_noise, dtype=jnp.int32)
    ctx_positions = jnp.arange(T_padded, dtype=jnp.int32)
    kv_cache_k = jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=dtype)
    kv_cache_v = jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=dtype)
    cache_len = jnp.array(0, dtype=jnp.int32)
    actual_ctx_count = jnp.array(6, dtype=jnp.int32)

    with jax.set_mesh(mesh):
        output, new_k, new_v = layer(
            x=x_noise,
            target_hidden=target_hidden,
            noise_positions=noise_positions,
            ctx_positions=ctx_positions,
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            cache_len=cache_len,
            actual_ctx_count=actual_ctx_count,
        )

    assert output.shape == (T_noise, hidden_size)
    assert new_k.shape == (1, num_heads, max_kv_len, head_dim)
    assert new_v.shape == (1, num_heads, max_kv_len, head_dim)
    mock_fa.assert_called_once()


@patch(
    "tpu_inference.models.jax.dflash.flash_attention",
    side_effect=dummy_flash_attention,
)
def test_dflash_for_causal_lm(mock_fa, hf_config, mesh):
    """Validates the full draft model's forward pass, logits calculation, and state projection."""
    vllm_config = MockVllmConfig(hf_config)
    rng_key = jax.random.PRNGKey(0)

    with jax.set_mesh(mesh):
        model = DFlashForCausalLM(
            vllm_config=vllm_config,
            rng_key=rng_key,
            mesh=mesh,
        )

    # Check structure
    assert model.model.embed_tokens.embedding.shape == (
        hf_config.vocab_size,
        hf_config.hidden_size,
    )
    assert len(model.model.layers) == hf_config.num_hidden_layers

    # Run forward pass
    T_noise = 3
    T_padded = 6
    hidden_size = hf_config.hidden_size
    num_heads = model.model.layers[0].self_attn.num_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    max_kv_len = 16

    input_ids = jnp.ones((T_noise, ), dtype=jnp.int32)
    ctx_hidden = jnp.ones((T_padded, hidden_size), dtype=jnp.bfloat16)
    cache_len_arr = jnp.array([2], dtype=jnp.int32)
    actual_ctx_count_arr = jnp.array([4], dtype=jnp.int32)

    target_hidden_states = (ctx_hidden, cache_len_arr, actual_ctx_count_arr)

    kv_caches = [
        jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16)
        for _ in range(2 * hf_config.num_hidden_layers)
    ]

    with jax.set_mesh(mesh):
        new_kv_caches, hidden_states, extra = model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            attention_metadata=None,
        )

    assert len(new_kv_caches) == 2 * hf_config.num_hidden_layers
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

        # Verify loader config propagation
        called_args, called_kwargs = mock_load_hf.call_args
        assert called_kwargs["vllm_config"] == vllm_config
        assert called_kwargs["model"] == model
        assert called_kwargs["mesh"] == mesh
        assert called_kwargs["is_draft_model"] is True
