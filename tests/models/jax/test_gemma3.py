# Copyright 2026 Google LLC
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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import CacheConfig, ModelConfig

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.jax.pp_utils import PPMissingLayer
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.gemma3 import (
    AttentionMetadata, Gemma3Attention, Gemma3DecoderLayer, Gemma3MLP,
    Gemma3Model, Gemma3ForCausalLM, GemmaRMSNorm
)


# --- Configuration Mocking ---
class MockGemma3Config:
    def __init__(self):
        self.hidden_size = 16
        self.intermediate_size = 32
        self.num_attention_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = 8
        self.num_hidden_layers = 2
        self.rms_norm_eps = 1e-6
        self.vocab_size = 32000
        self.tie_word_embeddings = True
        self.rope_theta = 10000.0
        self.sliding_window = 1024
        self.layer_types = ["global", "sliding_attention"]
        self.query_pre_attn_scalar = 144.0


class MockModelConfig:
    def __init__(self, hf_config, dtype):
        self.hf_config = hf_config
        self.dtype = dtype
        self.model = "mock_gemma3"
        self.tokenizer = "mock_tokenizer"
        self.tokenizer_mode = "auto"
        self.trust_remote_code = True
        self.seed = 0

    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_head_size(self):
        return self.hf_config.head_dim

    def get_vocab_size(self):
        return self.hf_config.vocab_size


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Gemma 3 model."""
    def __init__(self):
        hf_config = MockGemma3Config()
        self.model_config = MockModelConfig(hf_config, jnp.bfloat16)
        self.cache_config = MagicMock(spec=CacheConfig)
        self.cache_config.cache_dtype = "auto"
        self.load_config = MagicMock()
        self.extra_configs = {}
        self.additional_config = {}
        self.quant_config = None


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
    return jax.random.PRNGKey(42)

@pytest.fixture
def mock_vllm_config() -> MockVllmConfig:
    return MockVllmConfig()


@pytest.fixture
def rngs(rng: PRNGKey) -> nnx.Rngs:
    return nnx.Rngs(params=rng)


@pytest.fixture(autouse=True, scope="module")
def initialize_pp():
    init_pp_distributed_environment(
        ip="",
        rank=0,
        world_size=1,
        device=jax.devices()[0],
        need_pp=False,
    )


# --- Test Classes ---

class TestGemmaRMSNorm:
    def test_forward(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs, mesh: Mesh):
        dtype = mock_vllm_config.model_config.dtype
        with jax.set_mesh(mesh):
            norm = GemmaRMSNorm(16, epsilon=1e-6, dtype=dtype, rngs=rngs)
        x = jnp.ones((5, 16), dtype=dtype)
        y = norm(x)
        assert y.shape == (5, 16)
        assert y.dtype == dtype


class TestGemma3MLP:
    def test_forward(self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs, mesh: Mesh):
        config = mock_vllm_config.model_config.hf_config
        dtype = mock_vllm_config.model_config.dtype
        with jax.set_mesh(mesh):
            mlp = Gemma3MLP(config, dtype, rngs, quant_config=None)
        x = jnp.ones((5, config.hidden_size), dtype=dtype)
        y = mlp(x)
        assert y.shape == (5, config.hidden_size)
        assert y.dtype == dtype


class TestGemma3Attention:
    @patch('tpu_inference.models.jax.gemma3.attention')
    def test_forward_global_attention(
        self, mock_attention: MagicMock, mock_vllm_config: MockVllmConfig,
        rngs: nnx.Rngs, mesh: Mesh, rng: PRNGKey
    ):
        config = mock_vllm_config.model_config.hf_config
        dtype = mock_vllm_config.model_config.dtype
        with jax.set_mesh(mesh):
            # Prefix points to layer 0, which is "global" in our MockGemma3Config
            attn_module = Gemma3Attention(config, dtype, rngs, mesh, "auto", quant_config=None, prefix="model.layers.0.self_attn")
        
        assert not attn_module.is_sliding

        T, D = 10, config.hidden_size
        x = jax.random.normal(rng, (T, D))
        kv_cache = jnp.zeros((T, 2, config.num_key_value_heads, attn_module.head_dim))
        attn_meta = MagicMock(spec=AttentionMetadata)
        attn_meta.input_positions = jnp.arange(T)
        
        # Mocking the highly optimized Pallas kernel return
        mock_attention.return_value = (kv_cache, jnp.ones((T, config.num_attention_heads, attn_module.head_dim)))
        
        new_kv, y = attn_module(kv_cache, x, attn_meta)
        assert y.shape == (T, D)
        mock_attention.assert_called_once()

    @patch('tpu_inference.models.jax.gemma3.attention')
    def test_forward_sliding_attention(
        self, mock_attention: MagicMock, mock_vllm_config: MockVllmConfig,
        rngs: nnx.Rngs, mesh: Mesh, rng: PRNGKey
    ):
        config = mock_vllm_config.model_config.hf_config
        dtype = mock_vllm_config.model_config.dtype
        with jax.set_mesh(mesh):
            # Prefix points to layer 1, which is "sliding_attention" in our MockGemma3Config
            attn_module = Gemma3Attention(config, dtype, rngs, mesh, "auto", quant_config=None, prefix="model.layers.1.self_attn")
        
        assert attn_module.is_sliding


class TestGemma3DecoderLayer:
    @patch('tpu_inference.models.jax.gemma3.attention')
    def test_forward(
        self, mock_attention: MagicMock, mock_vllm_config: MockVllmConfig,
        rngs: nnx.Rngs, mesh: Mesh, rng: PRNGKey
    ):
        config = mock_vllm_config.model_config.hf_config
        dtype = mock_vllm_config.model_config.dtype
        with jax.set_mesh(mesh):
            layer = Gemma3DecoderLayer(config, dtype, rngs, mesh, "auto", quant_config=None, prefix="layer.0")
            
        T, D = 10, config.hidden_size
        x = jax.random.normal(rng, (T, D))
        residual = None
        kv_cache = jnp.zeros((T, 2, config.num_key_value_heads, layer.self_attn.head_dim))
        attn_meta = MagicMock(spec=AttentionMetadata)
        attn_meta.input_positions = jnp.arange(T)
        
        mock_attention.return_value = (kv_cache, jnp.ones((T, config.num_attention_heads, layer.self_attn.head_dim)))
        
        new_kv, hidden_states, new_residual = layer(kv_cache, x, residual, attn_meta)
        assert hidden_states.shape == (T, D)
        assert new_residual.shape == (T, D)
        
        # Test with existing residual
        new_kv, hidden_states, new_residual = layer(kv_cache, x, new_residual, attn_meta)
        assert hidden_states.shape == (T, D)


class TestGemma3ForCausalLM:
    @pytest.fixture
    def model(self, mock_vllm_config: MockVllmConfig, rng: PRNGKey, mesh: Mesh):
        with jax.set_mesh(mesh):
            model = Gemma3ForCausalLM(mock_vllm_config, rng, mesh)
            yield model

    def test_embed_input_ids(self, model: Gemma3ForCausalLM, rng: PRNGKey):
        input_ids = jnp.array([[1, 2, 3]])
        embeds = model.embed_input_ids(input_ids)
        assert embeds.shape == (1, 3, model.vllm_config.model_config.hf_config.hidden_size)

    def test_compute_logits(self, model: Gemma3ForCausalLM):
        hidden_states = jnp.ones((10, model.vllm_config.model_config.hf_config.hidden_size))
        logits = model.compute_logits(hidden_states)
        assert logits.shape == (10, model.vllm_config.model_config.hf_config.vocab_size)

    @patch('tpu_inference.models.jax.gemma3.attention')
    def test_call(self, mock_attention: MagicMock, model: Gemma3ForCausalLM, rng: PRNGKey):
        config = model.vllm_config.model_config.hf_config
        T, D = 10, config.hidden_size
        head_dim = model.model.layers[0].self_attn.head_dim
        
        input_ids = jnp.arange(T)
        kv_caches = [jnp.zeros((T, 2, config.num_key_value_heads, head_dim)) for _ in range(config.num_hidden_layers)]
        attn_meta = MagicMock(spec=AttentionMetadata)
        attn_meta.input_positions = jnp.arange(T)
        
        mock_attention.return_value = (kv_caches[0], jnp.ones((T, config.num_attention_heads, head_dim)))
        
        new_kvs, hidden_states, aux = model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attn_meta
        )
        
        assert len(new_kvs) == config.num_hidden_layers
        assert hidden_states.shape == (T, D)
        assert len(aux) == 0


class TestGemma3PipelineParallel:
    @pytest.fixture
    def mock_pp_group(self):
        with patch('tpu_inference.models.jax.gemma3.get_pp_group') as mock:
            yield mock

    def test_init_first_rank(self, mock_vllm_config, rng, mesh, mock_pp_group):
        mock_pp_group.return_value.is_first_rank = True
        mock_pp_group.return_value.is_last_rank = False
        with jax.set_mesh(mesh):
            model = Gemma3ForCausalLM(mock_vllm_config, rng, mesh)
            assert not isinstance(model.model.embed_tokens, PPMissingLayer)
            assert not hasattr(model, 'lm_head')

    def test_init_last_rank(self, mock_vllm_config, rng, mesh, mock_pp_group):
        mock_pp_group.return_value.is_first_rank = False
        mock_pp_group.return_value.is_last_rank = True
        with jax.set_mesh(mesh):
            model = Gemma3ForCausalLM(mock_vllm_config, rng, mesh)
            assert not isinstance(model.model.embed_tokens, PPMissingLayer)
            assert not hasattr(model, 'lm_head')

    @patch('tpu_inference.models.jax.gemma3.Gemma3Model')
    def test_call_non_last_rank(self, mock_gemma3_model, mock_vllm_config, rng, mesh, mock_pp_group):
        mock_pp_group.return_value.is_first_rank = True
        mock_pp_group.return_value.is_last_rank = False

        with jax.set_mesh(mesh):
            model = Gemma3ForCausalLM(mock_vllm_config, rng, mesh)
            
            kv_caches = [jnp.array([])]
            input_ids = jnp.array([1, 2, 3])
            attn_meta = MagicMock(spec=AttentionMetadata)

            # Simulate Gemma3Model returning hidden states and residuals
            model.model.return_value = (kv_caches, jnp.ones((3, 16)), jnp.ones((3, 16)))
            
            new_kvs, x, aux = model(kv_caches, input_ids, attn_meta, is_first_rank=True, is_last_rank=False)
            
            assert isinstance(x, JaxIntermediateTensors)
            assert "hidden_states" in x.tensors
            assert "residual" in x.tensors