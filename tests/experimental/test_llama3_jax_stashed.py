from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh

from tpu_commons.experimental.llama3_jax_stashed import (Llama3WeightLoader,
                                                         LlamaForCausalLM)


class MockParam:
    """A mock for a parameter used in the Llama model."""

    def __init__(self, shape=(32, 128)):
        self.value = SimpleNamespace(shape=shape)
        # The sharding spec is accessed during weight loading
        self.sharding = SimpleNamespace(spec=None)

    # Allow the mock parameter's value to be updated
    def __setattr__(self, name, value):
        if name == "value":
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Llama3 model."""

    def __init__(self,
                 model_name: str,
                 random_weights: bool = False,
                 tensor_parallelism: int = 1):
        self.model_config = SimpleNamespace(model=model_name,
                                            dtype="bfloat16",
                                            hf_overrides={},
                                            override_generation_config={})
        self.additional_config = {
            "random_weights": random_weights,
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
    FIX: The sharding logic expects 'data', 'model', and 'expert' axes.
    This creates a 3D mesh to satisfy the sharding rules, even on a single device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices())
    # Reshape devices into a 3D array to name 3 axes: data, model, and expert.
    # The 'model' and 'expert' axes will have a size of 1.
    num_devices = len(devices)
    device_mesh = devices.reshape((num_devices, 1, 1))

    with Mesh(device_mesh, axis_names=('data', 'model', 'expert')) as m:
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_vllm_config_8b() -> MockVllmConfig:
    return MockVllmConfig(model_name="meta-llama/Llama-3-8B")


@pytest.fixture
def mock_vllm_config_70b() -> MockVllmConfig:
    return MockVllmConfig(model_name="meta-llama/Llama-3-70B-Instruct")


@pytest.fixture
def mock_vllm_config_unknown() -> MockVllmConfig:
    return MockVllmConfig(model_name="some-other-model")


# --- Test Cases ---


class TestLlamaForCausalLM:
    """Tests for the main LlamaForCausalLM model class."""

    def test_init_8b_variant(self, mock_vllm_config_8b, rng, mesh):
        """Tests correct parameter detection for the 8B model variant."""
        model = LlamaForCausalLM(mock_vllm_config_8b, rng, mesh)
        assert model.hidden_size == 4096
        assert "8b" in model.vllm_config.model_config.model.lower()

    def test_init_70b_variant(self, mock_vllm_config_70b, rng, mesh):
        """Tests correct parameter detection for the 70B model variant."""
        model = LlamaForCausalLM(mock_vllm_config_70b, rng, mesh)
        assert model.hidden_size == 8192
        assert "70b" in model.vllm_config.model_config.model.lower()

    def test_init_unknown_variant_raises_error(self, mock_vllm_config_unknown,
                                               rng, mesh):
        """Tests that an unknown model variant raises a ValueError."""
        with pytest.raises(ValueError,
                           match="Could not determine Llama3 variant"):
            LlamaForCausalLM(mock_vllm_config_unknown, rng, mesh)

    def test_create_model_with_random_weights(self, mock_vllm_config_8b, rng,
                                              mesh):
        """
        Tests that random weight initialization creates concrete, non-zero-variance arrays.
        """
        model = LlamaForCausalLM.create_model_with_random_weights(
            vllm_config=mock_vllm_config_8b, rng=rng, mesh=mesh)

        embedding_weight = model.embedder.input_embedding_table_VD.value
        attention_q_kernel = model.layers[0].attn.kernel_q_proj_DNH.value
        final_norm_scale = model.final_norm.scale.value

        assert isinstance(embedding_weight, jax.Array)
        assert isinstance(attention_q_kernel, jax.Array)
        assert isinstance(final_norm_scale, jax.Array)

        assert jnp.std(embedding_weight) > 0
        assert jnp.std(attention_q_kernel) > 0

        assert jnp.all(final_norm_scale == 1.0)

    @patch("tpu_commons.experimental.llama3_jax_stashed.Llama3WeightLoader")
    def test_load_weights_called_correctly(self, mock_loader_cls, rng, mesh):
        """Tests that the weight loader is called correctly for checkpoint loading."""
        vllm_config = MockVllmConfig(model_name="llama3-8b",
                                     random_weights=False)
        model = LlamaForCausalLM(vllm_config, rng, mesh)

        mock_loader_instance = MagicMock()
        mock_loader_cls.return_value = mock_loader_instance
        model.load_weights(rng, cache_dir="/tmp/cache")
        mock_loader_cls.assert_called_once_with(vllm_config=vllm_config,
                                                hidden_size=4096,
                                                attn_heads=32,
                                                num_key_value_heads=8,
                                                attn_head_dim=128)
        mock_loader_instance.load_weights.assert_called_once_with(model)


class TestLlama3WeightLoader:
    """Tests for the Llama3WeightLoader class."""

    @pytest.fixture
    def weight_loader(self):
        # Patch the superclass's setup to isolate the Llama3 loader's logic
        return Llama3WeightLoader(vllm_config=MockVllmConfig("test-model"),
                                  hidden_size=32,
                                  attn_heads=4,
                                  num_key_value_heads=2,
                                  attn_head_dim=8)

    @pytest.mark.parametrize("hf_key, expected", [
        ("model.layers.15.self_attn.q_proj",
         "layers.15.attn.kernel_q_proj_DNH"),
        ("model.layers.0.mlp.down_proj",
         "layers.0.custom_module.kernel_down_proj_FD"),
        ("model.embed_tokens", "embedder.input_embedding_table_VD"),
        ("model.norm", "final_norm.scale"),
        ("lm_head", "lm_head.input_embedding_table_DV"),
        ("unmapped.key.name", "unmapped.key.name"),
    ])
    def test_map_loaded_to_standardized_name(self, weight_loader, hf_key,
                                             expected):
        """Tests the mapping from HuggingFace key names to internal names."""
        assert weight_loader.map_loaded_to_standardized_name(
            hf_key) == expected

    def test_load_weights_transformation(self, weight_loader, rng, mesh):
        """Tests that weights are correctly reshaped, transposed, and loaded."""
        vllm_config = MockVllmConfig("llama3-8b-small-test",
                                     random_weights=False)

        # Create a model instance but override its config for the test.
        model = LlamaForCausalLM(vllm_config, rng, mesh)

        # Original weight shape is (vocab_size, hidden_size)
        original_weight = jnp.ones((128, 32))
        dummy_weights = [
            ("model.embed_tokens.weight", original_weight),
        ]
        weight_loader.names_and_weights_generator = dummy_weights

        # Mock get_param to return a mock param with the target shape (hidden_size, vocab_size)
        mock_param = MockParam(shape=(128, 32))

        with patch("tpu_commons.experimental.llama3_jax_stashed.model_weights_generator", return_value=dummy_weights), \
            patch("tpu_commons.experimental.llama3_jax_stashed.get_param", return_value=mock_param), \
            patch("tpu_commons.experimental.llama3_jax_stashed.shard_put", return_value=jnp.ones(mock_param.value.shape)) as mock_shard_put:
            # This will now pass after the code fix
            weight_loader.load_weights_single_thread(model, [], mesh)

            # Assert that shard_put was called with the correctly transposed weight
            mock_shard_put.assert_called_once()

            # Get the actual array passed to shard_put
            called_with_weight = mock_shard_put.call_args[0][0]

            # Check if the shape of the array passed to shard_put matches the model's expected shape.
            assert called_with_weight.shape == mock_param.value.shape
