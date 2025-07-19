from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh

from tpu_commons.models.jax.recipes.llama3 import (Llama3, Llama3ModelConfig,
                                                   Llama3RecipeConfig,
                                                   Llama3WeightLoader)


class MockMesh:
    """A mock for jax.sharding.Mesh."""

    def __init__(self, devices=None, axis_names=None):
        self.devices = devices if devices is not None else np.array([[0]])
        self.axis_names = axis_names if axis_names is not None else ('dp',
                                                                     'mp')

    @property
    def size(self):
        return self.devices.size


class MockParam:
    """A mock for a parameter used in the Llama3 model."""

    def __init__(self):
        self.value = SimpleNamespace(shape=(32, 128))
        self.sharding = MagicMock()


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
    """Creates a 1D mesh with all available devices for testing."""
    devices = jax.local_devices()
    device_mesh = np.array(devices).reshape((len(devices), ))
    with Mesh(device_mesh, axis_names=('data', )) as m:
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


@pytest.fixture
def small_model_config() -> Llama3ModelConfig:
    """A small model configuration for fast and memory-efficient tests."""
    return Llama3ModelConfig(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        head_dim=8,  # hidden_size / num_attention_heads
        vocab_size=128,
        dtype=jnp.bfloat16)


# --- Test Cases ---


class TestLlama3Configs:
    """Tests for the various dataclass configurations."""

    def test_llama3_model_config_defaults(self):
        """Tests that default values are set correctly."""
        config = Llama3ModelConfig(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
        )
        assert config.hidden_size == 128
        assert config.vocab_size == 128256  # Default vocab size

    def test_llama3_model_config_post_init(self):
        """Tests that sub-configurations are created in __post_init__."""
        config = Llama3ModelConfig(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
        )
        # Check that default sub-configs are created
        assert config.emb is not None
        assert config.emb.vocab_size == config.vocab_size
        assert config.emb.hidden_size == config.hidden_size

        assert config.layers is not None
        assert config.layers.attention.hidden_size == config.hidden_size
        assert config.layers.dense_ffw.hidden_size == config.hidden_size

    def test_llama3_recipe_config(self, small_model_config):
        """Tests the composition of configurations in Llama3RecipeConfig."""
        recipe = Llama3RecipeConfig(model=small_model_config)
        assert recipe.model == small_model_config
        assert recipe.sharding is not None
        assert recipe.serving is not None


class TestLlama3Model:
    """Tests for the main Llama3 model class."""

    @patch("tpu_commons.models.jax.recipes.llama3.Llama3._init_layers",
           return_value=None)
    def test_init_8b_variant(self, _, mock_vllm_config_8b, rng, mesh):
        """Tests correct parameter detection for the 8B model variant."""
        model = Llama3(mock_vllm_config_8b, rng, mesh)
        assert model.cfg.model.hidden_size == 4096
        assert model.cfg.model.num_layers == 32
        assert "8b" in model.vllm_config.model_config.model.lower()

    @patch("tpu_commons.models.jax.recipes.llama3.Llama3._init_layers",
           return_value=None)
    def test_init_70b_variant(self, _, mock_vllm_config_70b, rng, mesh):
        """Tests correct parameter detection for the 70B model variant."""
        model = Llama3(mock_vllm_config_70b, rng, mesh)
        assert model.cfg.model.hidden_size == 8192
        assert model.cfg.model.num_layers == 80
        assert "70b" in model.vllm_config.model_config.model.lower()

    def test_init_unknown_variant_raises_error(self, mock_vllm_config_unknown,
                                               rng, mesh):
        """Tests that an unknown model variant raises a ValueError."""
        with pytest.raises(ValueError,
                           match="Could not determine Llama3 variant"):
            Llama3(mock_vllm_config_unknown, rng, mesh)

    @patch("tpu_commons.models.jax.recipes.llama3.Llama3._init_layers",
           return_value=None)
    def test_load_weights_with_random_init(self, _, rng, mesh):
        """Tests that the weight loader is not called when random_init is true."""
        vllm_config = MockVllmConfig(model_name="llama3-8b",
                                     random_weights=True)
        model = Llama3(vllm_config, rng, mesh)

        with patch("tpu_commons.models.jax.recipes.llama3.Llama3WeightLoader"
                   ) as mock_loader:
            model.load_weights(rng)
            mock_loader.assert_not_called()

    @patch("tpu_commons.models.jax.recipes.llama3.Llama3._init_layers",
           return_value=None)
    def test_load_weights_from_checkpoint(self, _, rng, mesh):
        """Tests that the weight loader is called correctly."""
        vllm_config = MockVllmConfig(model_name="llama3-8b",
                                     random_weights=False)
        model = Llama3(vllm_config, rng, mesh)

        with patch("tpu_commons.models.jax.recipes.llama3.Llama3WeightLoader"
                   ) as mock_loader_cls:
            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance

            model.load_weights(rng, cache_dir="/tmp/cache")

            mock_loader_cls.assert_called_once_with(
                vllm_config=vllm_config,
                model_config=model.cfg.model,
                cache_dir="/tmp/cache",
                sharding_cfg=model.cfg.sharding)
            mock_loader_instance.load_weights.assert_called_once_with(model)


class TestLlama3WeightLoader:
    """Tests for the Llama3WeightLoader class."""

    @pytest.fixture
    def weight_loader(self, small_model_config):
        with patch(
                'tpu_commons.models.jax.utils.weight_utils.WeightLoader.setup'
        ):
            loader = Llama3WeightLoader(
                vllm_config=MockVllmConfig("test-model"),
                model_config=small_model_config)
        return loader

    @pytest.mark.parametrize("hf_key, expected", [
        ("model.layers.15.self_attn.q_proj",
         "layers.15.attn.kernel_q_proj_DNH"),
        ("model.layers.0.mlp.down_proj", "layers.0.mlp.kernel_down_proj_FD"),
        ("model.embed_tokens", "embedder.input_embedding_table_DV"),
        ("model.norm", "final_norm.scale"),
        ("lm_head", "lm_head.input_embedding_table_DV"),
        ("unmapped.key.name", "unmapped.key.name"),
    ])
    def test_map_loaded_to_standardized_name(self, weight_loader, hf_key,
                                             expected):
        """Tests the mapping from HuggingFace key names to internal names."""
        assert weight_loader.map_loaded_to_standardized_name(
            hf_key) == expected

    @patch("tpu_commons.models.jax.recipes.llama3.Llama3._init_layers",
           return_value=None)
    def test_load_weights_transformation(self, _, small_model_config, rng,
                                         mesh):
        """Tests that weights are correctly reshaped, transposed, and loaded."""
        vllm_config = MockVllmConfig("llama3-8b", random_weights=False)
        model = Llama3(vllm_config, rng, mesh)
        cfg = Llama3RecipeConfig(model=small_model_config,
                                 sharding=None,
                                 serving=None)
        model.cfg = cfg
        model._init_layers()

        loader = Llama3WeightLoader(vllm_config=vllm_config,
                                    model_config=small_model_config)

        dummy_weights = [
            ("model.embed_tokens.weight", jnp.ones(
                (128, 32))),  # vocab, hidden
        ]
        loader.names_and_weights_generator = dummy_weights

        with patch("tpu_commons.models.jax.recipes.llama3.get_param", return_value=MockParam()), \
        patch("tpu_commons.models.jax.recipes.llama3.shard_put", return_value=None) as mock_shard_put, \
        patch("flax.nnx.update", return_value=None) as mock_update:
            loader.load_weights(model)

        mock_shard_put.assert_called_once()
        mock_update.assert_called_once()
