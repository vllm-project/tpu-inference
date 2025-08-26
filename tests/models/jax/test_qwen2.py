from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM
from tpu_commons.runner.jax.kv_cache import create_kv_caches


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Qwen2 model."""

    def __init__(self, model: str):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock()
        self.load_config.download_dir = None


@pytest.fixture
def mock_vllm_config() -> MockVllmConfig:
    return MockVllmConfig(model="Qwen/Qwen2.5-1.5B")


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1))

    with Mesh(device_mesh, axis_names=('data', 'model')) as m:
        yield m


@pytest.fixture
def mock_model_inputs():
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


class TestQwen2ForCausalLM:
    """Tests for the main Qwen2ForCausalLM model class."""

    def test_qwen25_1_5b(self, mock_vllm_config, rng, mesh, mock_model_inputs):
        """Tests model init and model forward for the 8B model variant."""

        # Test model init
        model = Qwen2ForCausalLM(mock_vllm_config, rng, mesh)
        assert "1.5b" in model.vllm_config.model_config.model.lower()

        model_config = mock_vllm_config.model_config
        hf_config = model_config.hf_config

        assert model.mesh.shape == {"data": 1, "model": 1}

        layers = model.model.layers
        assert len(layers) == hf_config.num_hidden_layers
        assert isinstance(model.rng, nnx.Rngs)
        assert model.model.lm_head == model.model.embed.embedding

        attn = layers[0].self_attn
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        rope_theta = hf_config.rope_theta
        original_head_dim = hidden_size // num_heads
        head_dim = 128
        intermediate_size = hf_config.intermediate_size

        assert attn.hidden_size == hidden_size
        assert attn.num_heads == num_heads
        assert attn.num_kv_heads == num_kv_heads
        assert attn.rope_theta == rope_theta
        assert attn.head_dim_original == original_head_dim
        assert attn.head_dim == head_dim
        assert attn.q_proj.kernel.shape == (hidden_size, num_heads, head_dim)
        assert attn.k_proj.kernel.shape == (hidden_size, num_kv_heads,
                                            head_dim)
        assert attn.v_proj.kernel.shape == (hidden_size, num_kv_heads,
                                            head_dim)
        assert attn.o_proj.kernel.shape == (num_heads, head_dim, hidden_size)

        mlp = layers[0].mlp
        assert mlp.gate_proj.kernel.shape == (hidden_size, intermediate_size)
        assert mlp.up_proj.kernel.shape == (hidden_size, intermediate_size)
        assert mlp.down_proj.kernel.shape == (intermediate_size, hidden_size)

        # Test model load
        model.load_weights(rng)

        # Test model forward
        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=32,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            devices=mesh.devices[0],
        )
        # 1 seq with 16 tokens
        input_ids, attention_metadata, indices_do_sample = mock_model_inputs
        kv_caches, hidden_states = model(kv_caches, input_ids,
                                         attention_metadata)
        assert hidden_states.shape == (8, hidden_size)

        hidden_states = hidden_states[indices_do_sample]
        assert hidden_states.shape == (1, hidden_size)

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (1, hf_config.vocab_size)
