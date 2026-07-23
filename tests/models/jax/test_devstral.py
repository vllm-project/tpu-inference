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
"""Tests for Devstral-Small-2507 (MistralForCausalLM, 24B parameters)."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.mistral import MistralForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches

DEVSTRAL_MODEL = "mistralai/Devstral-Small-2507"
NUM_LAYERS_FOR_TEST = 4


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Mistral model."""

    def __init__(self, model: str, kv_cache_dtype: str):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = LoadConfig(load_format="auto")
        self.load_config.download_dir = None
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None


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
def mock_model_inputs():
    """Create mock inputs for model forward pass testing."""
    num_tokens = 8
    num_reqs = 1
    max_num_blocks_per_req = 4
    input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
    positions = jnp.ones((num_tokens, ), dtype=jnp.int32)
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.ones((num_reqs, ), dtype=jnp.int32)
    query_start_loc = jnp.ones((num_reqs + 1, ), dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 0], dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    indices_do_sample = jnp.ones((num_reqs, ), dtype=jnp.int32)

    return (input_ids, attention_metadata, indices_do_sample)


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    """Mock pipeline parallelism group for single-device testing."""
    mock_pp = MagicMock(is_first_rank=True,
                        is_last_rank=True,
                        rank_in_group=0,
                        world_size=1)
    with patch("tpu_inference.models.jax.mistral.get_pp_group",
               return_value=mock_pp), \
         patch("tpu_inference.layers.jax.pp_utils.get_pp_group",
               return_value=mock_pp):
        yield


class TestDevstralSmall:
    """Tests for Devstral-Small-2507 (MistralForCausalLM architecture)."""

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
    def test_devstral_init(self, kv_cache_dtype, rng, mesh):
        """Test model initialization with Devstral-Small config.

        Uses reduced layers for faster testing.
        """
        mock_vllm_config = MockVllmConfig(DEVSTRAL_MODEL, kv_cache_dtype)
        # Reduce layers for faster init
        mock_vllm_config.model_config.hf_config.num_hidden_layers = (
            NUM_LAYERS_FOR_TEST)

        with jax.set_mesh(mesh):
            model = MistralForCausalLM(mock_vllm_config, rng, mesh)

        hf_config = mock_vllm_config.model_config.hf_config

        # Verify model structure
        layers = model.model.layers
        assert len(layers) == NUM_LAYERS_FOR_TEST

        # Verify attention configuration
        attn = layers[0].self_attn
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = 128  # padded head dim

        assert attn.hidden_size == hidden_size  # 5120
        assert attn.num_heads == num_heads  # 32
        assert attn.num_kv_heads == num_kv_heads  # 8
        assert attn.rope_theta == hf_config.rope_theta  # 1e9
        assert attn.head_dim_original == hf_config.head_dim  # 128
        assert attn.head_dim == head_dim

        # Verify projection shapes
        assert attn.q_proj.weight.shape == (hidden_size, num_heads, head_dim)
        assert attn.k_proj.weight.shape == (hidden_size, num_kv_heads,
                                            head_dim)
        assert attn.v_proj.weight.shape == (hidden_size, num_kv_heads,
                                            head_dim)
        assert attn.o_proj.weight.shape == (num_heads, head_dim, hidden_size)

        # Verify no bias in attention (Mistral architecture)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None

        # Verify MLP shapes
        mlp = layers[0].mlp
        intermediate_size = hf_config.intermediate_size
        assert mlp.gate_proj.weight.shape == (hidden_size, intermediate_size)
        assert mlp.up_proj.weight.shape == (hidden_size, intermediate_size)
        assert mlp.down_proj.weight.shape == (intermediate_size, hidden_size)

        # Verify lm_head (Devstral uses tie_word_embeddings=False)
        assert not hf_config.tie_word_embeddings
        assert hasattr(model, 'lm_head')
        assert model.lm_head.weight.shape == (hidden_size,
                                              hf_config.vocab_size)

        # Verify KV cache quantization config
        if kv_cache_dtype == "fp8":
            assert attn.kv_cache_quantized_dtype is not None
            assert attn.kv_cache_quantized_dtype == jnp.float8_e4m3fn
        else:
            assert attn.kv_cache_quantized_dtype is None

    def test_devstral_weight_loading(self, rng, mesh, mock_model_inputs):
        """Test weight loading from HF checkpoint.

        Uses SkipLayersModelLoaderForTest to only load first few layers.
        """
        mock_vllm_config = MockVllmConfig(DEVSTRAL_MODEL, "auto")
        # Reduce layers to speed up test
        mock_vllm_config.model_config.hf_config.num_hidden_layers = (
            NUM_LAYERS_FOR_TEST)
        mock_vllm_config.load_config.load_format = (
            "skip_layers_model_loader_for_test")
        mock_vllm_config.load_config.num_layers_to_load_for_test = (
            NUM_LAYERS_FOR_TEST)

        model_config = mock_vllm_config.model_config
        hf_config = model_config.hf_config
        hidden_size = hf_config.hidden_size
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = 128

        # Init and load weights
        with jax.set_mesh(mesh):
            model = MistralForCausalLM(mock_vllm_config, rng, mesh)
            loader = get_model_loader(mock_vllm_config.load_config)
            loader.load_weights(model, model_config)

        # Test forward pass with loaded weights
        kv_caches = create_kv_caches(num_blocks=4,
                                     block_size=32,
                                     num_kv_heads=num_kv_heads,
                                     head_size=head_dim,
                                     mesh=mesh,
                                     layer_names=["layer"] *
                                     NUM_LAYERS_FOR_TEST,
                                     cache_dtype=jnp.bfloat16)

        input_ids, attention_metadata, indices_do_sample = mock_model_inputs
        kv_caches, hidden_states, aux_hidden_states = model(
            kv_caches, input_ids, attention_metadata)
        assert hidden_states.shape == (8, hidden_size)
        assert len(aux_hidden_states) == 0

        hidden_states = hidden_states[indices_do_sample]
        assert hidden_states.shape == (1, hidden_size)

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (1, hf_config.vocab_size)
