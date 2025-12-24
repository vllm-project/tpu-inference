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

from dataclasses import field
from types import SimpleNamespace
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_inference.models.jax.llama4 import (Llama4ForCausalLM,
                                             Llama4WeightLoader)


class MockParamLlama4:
    """A mock for a parameter used in the Llama4 model."""
    shape: Tuple[int, ...]
    dtype: jnp.dtype = jnp.bfloat16
    sharding_spec: Tuple[str | None, ...] | None = None
    value: Any = field(init=False)
    sharding: Any = field(init=False)

    def __init__(self, shape=(32, 128)):
        self.shape = shape
        self.value = jnp.zeros(self.shape, dtype=self.dtype)
        # The sharding spec is accessed during weight loading
        self.sharding = SimpleNamespace(spec=self.sharding_spec)

    # Allow the mock parameter's value to be updated
    def __setattr__(self, name, value):
        if name in ['value', 'shape', 'dtype', 'sharding', 'sharding_spec']:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Llama4 model."""

    def __init__(self,
                 model_name: str,
                 random_weights: bool = False,
                 tensor_parallelism: int = 1):
        self.model_config = MagicMock(spec=ModelConfig)
        self.load_config = MagicMock()
        self.load_config.download_dir = None

        # Choose small amount of layers to avoid OOM.
        self.model_config.get_vocab_size.return_value = 202048
        self.model_config.get_hidden_size.return_value = 32
        self.model_config.model = model_name

        self.additional_config = {
            "random_weights": random_weights,
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": tensor_parallelism
                }
            }
        }

        self.cache_config = MagicMock(cache_dtype="auto")

        text_config_mock = MagicMock()
        text_config_mock.interleave_moe_layer_step = 1
        text_config_mock.num_attention_heads = 40
        text_config_mock.num_key_value_heads = 8
        text_config_mock.head_dim = 128

        hf_config_mock = MagicMock()
        hf_config_mock.text_config = text_config_mock

        self.model_config.hf_config = hf_config_mock


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with all required axes for testing.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices())
    # Reshape devices into a 3D array to name 3 axes: data, model, and expert.
    # The 'model' and 'expert' axes will have a size of 1.
    num_devices = len(devices)
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'model', 'expert')) as m:
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_vllm_config_llama4() -> MockVllmConfig:
    return MockVllmConfig(model_name="meta-llama/Llama-4-Scout-17B-16E")


class TestLlama4ForCausalLM:
    """Tests for the main LlamaForCausalLM model class."""

    def test_init_llama4(self, mock_vllm_config_llama4, rng, mesh):
        """Tests correct parameter detection for the Llama4 model variant."""
        model = Llama4ForCausalLM(mock_vllm_config_llama4, rng, mesh)
        assert model.hidden_size == 32
        assert "llama-4" in model.vllm_config.model_config.model.lower()

    def test_create_model_with_random_weights(self, mock_vllm_config_llama4,
                                              rng, mesh):
        """
        Tests that random weight initialization creates concrete, non-zero-variance arrays.
        """
        with jax.set_mesh(mesh):
            model = Llama4ForCausalLM(vllm_config=mock_vllm_config_llama4,
                                      rng=rng,
                                      mesh=mesh,
                                      force_random_weights=True)
            embedding_weight = model.embedder.input_embedding_table_VD.value
            attention_q_kernel = model.layers[0].attn.kernel_q_proj_DNH.value
            final_norm_scale = model.final_norm.scale.value

            assert isinstance(embedding_weight, jax.Array)
            assert isinstance(attention_q_kernel, jax.Array)
            assert isinstance(final_norm_scale, jax.Array)

            assert jnp.std(embedding_weight) > 0
            assert jnp.std(attention_q_kernel) > 0

            assert jnp.all(final_norm_scale == 1.0)

    @patch("tpu_inference.models.jax.llama4.Llama4WeightLoader")
    def test_load_weights_called_correctly(self, mock_loader_cls, rng, mesh):
        """Tests that the weight loader is called correctly for checkpoint loading."""
        vllm_config = MockVllmConfig(model_name="llama4-scout",
                                     random_weights=False)
        model = Llama4ForCausalLM(vllm_config, rng, mesh)

        mock_loader_instance = MagicMock()
        mock_loader_cls.return_value = mock_loader_instance
        model.load_weights(rng)

        mock_loader_cls.assert_called_once_with(vllm_config=vllm_config,
                                                hidden_size=32,
                                                attn_heads=40,
                                                num_key_value_heads=8,
                                                attn_head_dim=128)
        mock_loader_instance.load_weights.assert_called_once_with(model)


class TestLlama4WeightLoader:
    """Tests for the Llama4WeightLoader class."""

    @pytest.fixture
    def weight_loader(self):
        # Patch the superclass's setup to isolate the Llama4 loader's logic
        return Llama4WeightLoader(vllm_config=MockVllmConfig("test-model"),
                                  hidden_size=32,
                                  attn_heads=40,
                                  num_key_value_heads=8,
                                  attn_head_dim=128)

    @pytest.mark.parametrize("hf_key, expected_num", [
        ("language_model.model.layers.15.self_attn.q_proj.weight", 15),
        ("layers.0.feed_forward.router.weight", 0),
        ("language_model.model.layers.99.norm.weight", 99),
        ("language_model.model.norm.weight", None),
        ("language_model.model.embed_tokens.weight", None),
    ])
    def test_get_layer_num(self, weight_loader, hf_key, expected_num):
        """Tests the private _get_layer_num utility function."""
        assert weight_loader._get_layer_num(hf_key) == expected_num

    @pytest.mark.parametrize("hf_key, expected_num", [
        ("language_model.model.layers.10.feed_forward.experts.4.down_proj.weight",
         4),
        ("language_model.model.layers.0.feed_forward.experts.0.gate_proj.weight_scale",
         0),
        ("language_model.model.layers.5.feed_forward.experts.128.up_proj.weight",
         128),
        ("language_model.model.norm.weight", None),
        ("language_model.model.layers.15.self_attn.q_proj.weight", None),
    ])
    def test_get_expert_num(self, weight_loader, hf_key, expected_num):
        """Tests the private _get_expert_num utility function to extract the expert index."""
        assert weight_loader._get_expert_num(hf_key) == expected_num

    @pytest.mark.parametrize("hf_key, expected", [
        ("language_model.model.layers.15.self_attn.q_proj.weight",
         "layers.15.attn.kernel_q_proj_DNH"),
        ("language_model.model.layers.0.feed_forward.shared_expert.down_proj.weight",
         "layers.0.shared_experts.kernel_down_proj_FD"),
        ("language_model.model.embed_tokens.weight",
         "embedder.input_embedding_table_VD"),
        ("language_model.model.norm.weight", "final_norm.scale"),
        ("language_model.lm_head.weight", "lm_head.input_embedding_table_DV"),
        ("unmapped.key.name", "unmapped.key.name"),
    ])
    def test_map_loaded_to_standardized_name(self, weight_loader, hf_key,
                                             expected):
        """Tests the mapping from HuggingFace key names to internal names."""
        assert weight_loader.map_loaded_to_standardized_name(
            hf_key) == expected

    def test_load_weights_transformation(self, weight_loader, rng, mesh):
        """Tests that weights are correctly reshaped, transposed, and loaded."""
        vllm_config = MockVllmConfig(model_name="llama4-small-test",
                                     random_weights=False)

        model = Llama4ForCausalLM(vllm_config, rng, mesh)

        # Original weight shape is (vocab_size, hidden_size)
        original_weight = jnp.ones((128, 32))
        dummy_weights = [
            ("language_model.model.embed_tokens.weight", original_weight),
        ]
        weight_loader.names_and_weights_generator = dummy_weights

        # Mock get_param to return a mock param with the target shape (vocab_size, hidden_size)
        mock_param = MockParamLlama4(shape=(128, 32))

        with patch("tpu_inference.models.jax.llama4.get_param", return_value=mock_param), \
            patch("tpu_inference.models.jax.llama4.shard_put", return_value=jnp.ones(mock_param.value.shape)) as mock_shard_put:

            # This will now pass after the code fix
            weight_loader.load_weights(model)

            # Assert that shard_put was called with the correctly transposed weight
            mock_shard_put.assert_called_once()

            # Get the actual array passed to shard_put
            called_with_weight = mock_shard_put.call_args[0][0]

            # Check if the shape of the array passed to shard_put matches the model's expected shape.
            assert called_with_weight.shape == mock_param.value.shape

    def test_map_llama4_gate_up_proj(self, weight_loader, rng, mesh):
        """Tests that gate_up_proj weights are correctly split, reshaped, transposed, and loaded."""
        # Set up a dummy model and its config
        model = Llama4ForCausalLM(MockVllmConfig("test-model"), rng, mesh)

        # Create a dummy fused gate_up_proj weight tensor
        hidden_size = 32
        intermediate_size_moe = 8192
        num_local_experts = 2
        dummy_weight = jnp.ones(
            (num_local_experts, hidden_size, 2 * intermediate_size_moe))

        # Set up mocks and patches
        mock_model_params = nnx.state(model)
        mock_param = MockParamLlama4(shape=(2, hidden_size,
                                            intermediate_size_moe))

        # Create a dummy WeightLoader and set up the necessary attributes
        weight_loader.is_verbose = False
        layer_num = 0
        weight_loader.names_and_weights_generator = [
            (f"language_model.model.layers.{layer_num}.feed_forward.experts.gate_up_proj.weight",
             dummy_weight),
        ]

        with patch("tpu_inference.models.jax.llama4.get_param", return_value=mock_param), \
            patch("tpu_inference.models.jax.llama4.shard_put", return_value=jnp.ones(mock_param.value.shape)) as mock_shard_put:

            # Call _map_llama4_gate_up_proj directly
            weight_loader._map_llama4_gate_up_proj(
                model, mock_model_params,
                f"language_model.model.layers.{layer_num}.feed_forward.experts.gate_up_proj.weight",
                dummy_weight)
            # Check if shard_put was called the correct number of times and with the correct weight shapes
            assert mock_shard_put.call_count == 2
            # call_args_list gives you a list of all the calls with their arguments.
            for call in mock_shard_put.call_args_list:
                assert call[0][0].shape == (num_local_experts, 32, 8192)
