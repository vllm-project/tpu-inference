from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from transformers import Qwen2Config
from vllm.config import VllmConfig

from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Qwen2 model."""

    def __init__(self,
                 model_name: str,
                 tensor_parallelism: int = 1,
                 tie_word_embeddings: bool = False):
        self.model_config = MagicMock(spec=VllmConfig.model_config)
        self.model_config.hf_config = Qwen2Config(
            # Use small values for testing to avoid OOM
            vocab_size=1024,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            max_window_layers=2,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.model_config.get_vocab_size.return_value = self.model_config.hf_config.vocab_size
        self.model_config.get_hidden_size.return_value = self.model_config.hf_config.hidden_size
        self.model_config.model = model_name
        self.model_config.dtype = jnp.bfloat16

        self.load_config = MagicMock()
        self.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": tensor_parallelism
                }
            }
        }


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with all required axes for testing.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices())
    # Reshape devices into a 3D array to name 3 axes: data, model, and expert.
    # The 'model' and 'expert' axes will have a size of 1 for single-device tests.
    num_devices = len(devices)
    device_mesh = devices.reshape((num_devices, 1, 1))

    with Mesh(device_mesh, axis_names=('data', 'model', 'expert')) as m:
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_vllm_config_qwen2() -> MockVllmConfig:
    return MockVllmConfig(model_name="Qwen/Qwen2-1.5B-Instruct")


class TestQwen2ForCausalLM:
    """Tests for the main Qwen2ForCausalLM model class."""

    def test_init_qwen2(self, mock_vllm_config_qwen2, rng, mesh):
        """Tests correct initialization of the Qwen2 model."""
        model = Qwen2ForCausalLM(mock_vllm_config_qwen2, rng, mesh)
        assert model.vllm_config == mock_vllm_config_qwen2
        assert len(
            model.model.layers
        ) == mock_vllm_config_qwen2.model_config.hf_config.num_hidden_layers
        assert model.model.norm.num_features == mock_vllm_config_qwen2.model_config.hf_config.hidden_size
        assert not model.vllm_config.model_config.hf_config.tie_word_embeddings
        assert isinstance(model.lm_head, nnx.Param)

    def test_init_qwen2_tied_embeddings(self, rng, mesh):
        """Tests correct initialization when embeddings are tied."""
        vllm_config = MockVllmConfig(model_name="Qwen/Qwen2-1.5B-Instruct",
                                     tie_word_embeddings=True)
        model = Qwen2ForCausalLM(vllm_config, rng, mesh)
        assert model.vllm_config.model_config.hf_config.tie_word_embeddings
        # Check if lm_head is a reference to the embedding table
        assert model.lm_head is model.embed.embedding

    @patch("tpu_commons.models.jax.qwen2.load_hf_weights")
    def test_load_weights_called_correctly(self, mock_load_hf_weights,
                                           mock_vllm_config_qwen2, rng, mesh):
        """Tests that the weight loader utility is called correctly."""
        model = Qwen2ForCausalLM(mock_vllm_config_qwen2, rng, mesh)
        model.load_weights(rng)

        mock_load_hf_weights.assert_called_once()
        call_args = mock_load_hf_weights.call_args[1]
        assert call_args['vllm_config'] == mock_vllm_config_qwen2
        assert call_args['model'] is model
        assert "model.embed_tokens" in call_args['mappings']
        assert "lm_head" in call_args[
            'mappings']  # since tie_word_embeddings is False
        assert call_args['mesh'] is mesh

    @patch("tpu_commons.models.jax.qwen2.load_hf_weights")
    def test_load_weights_called_correctly_tied_embeddings(
            self, mock_load_hf_weights, rng, mesh):
        """Tests that the weight loader utility is called correctly with tied embeddings."""
        vllm_config = MockVllmConfig(model_name="Qwen/Qwen2-1.5B-Instruct",
                                     tie_word_embeddings=True)
        model = Qwen2ForCausalLM(vllm_config, rng, mesh)
        model.load_weights(rng)

        mock_load_hf_weights.assert_called_once()
        call_args = mock_load_hf_weights.call_args[1]
        assert "lm_head" not in call_args['mappings']
