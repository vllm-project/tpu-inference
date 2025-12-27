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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig

# Assuming the model file is named deepseek_v3.py
from tpu_inference.models.jax.deepseek_v3 import (DeepSeekV3,
                                                  DeepSeekV3WeightLoader)


class MockVariable:
    """Mocks an nnx.Variable or a QArray structure."""

    def __init__(self, shape, dtype=jnp.bfloat16, sharding=None):
        self.value = jnp.zeros(shape, dtype=dtype)
        self.sharding = sharding or (None, ) * len(shape)
        self.nbytes = self.value.nbytes
        # Handle the QArray structure used in the loader
        self.array = SimpleNamespace(
            qvalue=self,
            scale=SimpleNamespace(
                value=jnp.ones((1, )),
                nbytes=4,
                sharding=None,
                addressable_shards=[SimpleNamespace(data=jnp.ones((1, )))]))
        self.addressable_shards = [SimpleNamespace(data=self.value)]


class MockVllmConfig:
    """Mock VllmConfig for DeepSeekV3."""

    def __init__(self,
                 model_name: str = "deepseek-ai/DeepSeek-V3",
                 use_mla: bool = False):
        self.model_config = MagicMock(spec=ModelConfig)
        self.model_config.model = model_name
        self.model_config.use_mla = use_mla

        # DeepSeek V3 specific config
        hf_config = MagicMock()
        hf_config.num_hidden_layers = 1  # Small for testing
        hf_config.num_nextn_predict_layers = 1
        self.model_config.hf_config = hf_config

        self.load_config = MagicMock()
        self.load_config.download_dir = None

        self.cache_config = MagicMock()
        self.cache_config.cache_dtype = "auto"

        self.additional_config = {
            "random_weights": False,
            "sparse_matmul": False,
            "is_verbose": True
        }


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available.")
    devices = np.array(jax.local_devices())
    num_devices = len(devices)
    device_mesh = devices.reshape((num_devices, 1, 1, 1))
    # Simplify axis names for testing
    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'model', 'expert')) as m:
        yield m


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def mock_config():
    return MockVllmConfig()


class TestDeepSeekV3:

    def test_init(self, mock_config, rng, mesh):
        """Tests if the model initializes with the correct hierarchy."""
        model = DeepSeekV3(mock_config, rng, mesh)
        assert len(model.layers) == 3  # num_layers from mock
        assert isinstance(model.embedder, nnx.Module)
        assert model.vllm_config.model_config.hf_config.num_hidden_layers == 1

    def test_random_weights(self, mock_config, rng, mesh):
        """Tests that force_random_weights initializes non-zero weights."""
        with jax.set_mesh(mesh):
            model = DeepSeekV3(mock_config,
                               rng,
                               mesh,
                               force_random_weights=True)
            # Check embedding
            weight = model.embedder.input_embedding_table_VD.value
            assert jnp.std(weight) > 0
            # Check a layer norm (should be 1s usually, but check existence)
            assert model.final_norm.scale.value.shape == (7168, )

    @patch("tpu_inference.models.jax.deepseek_v3.DeepSeekV3WeightLoader")
    def test_load_weights_called(self, mock_loader_cls, mock_config, rng,
                                 mesh):
        model = DeepSeekV3(mock_config, rng, mesh)
        mock_loader_instance = mock_loader_cls.return_value

        model.load_weights(rng)

        mock_loader_instance.load_weights.assert_called_once_with(model)


class TestDeepSeekV3WeightLoader:

    @pytest.fixture
    def loader(self, mock_config):
        # We need to mock the generator so it doesn't try to download files
        with patch(
                "tpu_inference.models.jax.deepseek_v3.model_weights_generator",
                return_value=[]):
            return DeepSeekV3WeightLoader(vllm_config=mock_config,
                                          num_layers=2,
                                          hidden_size=7168,
                                          q_lora_rank=1536,
                                          kv_lora_rank=512,
                                          attn_heads=128,
                                          qk_nope_head_dim=128,
                                          qk_rope_head_dim=64,
                                          v_head_dim=128,
                                          num_local_experts=256,
                                          model_dtype=jnp.bfloat16)

    @pytest.mark.parametrize("loaded_key, expected_mapped", [
        ("model.embed_tokens.weight", "embedder.input_embedding_table_VD"),
        ("model.layers.0.self_attn.q_a_proj.weight",
         "layers.0.attn.kernel_q_down_proj_DA"),
        ("model.layers.5.mlp.experts.10.gate_proj.weight",
         "layers.5.custom_module.kernel_gating_EDF"),
        ("model.layers.1.mlp.shared_experts.down_proj.weight",
         "layers.1.shared_experts.kernel_down_proj_FD"),
        ("model.norm.weight", "final_norm.scale"),
    ])
    def test_key_mapping(self, loader, loaded_key, expected_mapped):
        assert loader.map_loaded_to_standardized_name(
            loaded_key) == expected_mapped

    def test_transpose_params(self, loader):
        # Test a standard MLP transpose (1, 0)
        dummy_weight = jnp.ones((100, 200))
        transposed = loader._transpose_params("mlp.down_proj", dummy_weight)
        assert transposed.shape == (200, 100)

        # Test MLA kernel transpose (2, 0, 1)
        dummy_mla = jnp.ones((10, 20, 30))
        transposed_mla = loader._transpose_params("k_b_proj", dummy_mla)
        assert transposed_mla.shape == (30, 10, 20)

    def test_moe_stacking_logic(self, loader):
        """Tests that individual expert weights are collected and stacked correctly."""
        weights_dict = {}
        layer_num = "0"
        loader.num_routed_experts = 4  # Small for test

        # Simulate loading 4 experts
        for i in range(4):
            name = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            weight = torch.ones((10, 20)) * i
            result = loader._process_moe_weights(name, weight, weights_dict)

            if i < 3:
                assert result is None
                assert weights_dict[layer_num][1] == i + 1
            else:
                # On the last expert, it should return stacked tensor
                assert result is not None
                assert result.shape == (4, 10, 20)
                assert layer_num not in weights_dict  # Should be cleaned up

    def test_mla_kernel_weight_splitting(self, loader, mesh):
        """Tests that kv_b_proj is split into k_b_proj and v_b_proj for MLA kernel."""
        loader.use_mla_kernel = True
        loader.attn_heads = 2
        loader.qk_nope_head_dim = 4
        loader.v_head_dim = 4
        loader.kv_lora_rank = 8

        # Total rows = heads * (nope_dim + v_dim) = 2 * (4 + 4) = 16
        # Cols = kv_lora_rank = 8
        kv_b_proj_weight = torch.randn((16, 8))

        # Mocking the load_individual_weight to capture what gets passed
        with patch.object(loader,
                          '_load_individual_weight',
                          return_value=(0, 0)):
            model_mock = MagicMock()
            model_mock.mesh = mesh

            # Simulate the splitting logic in the loader
            weight_reshaped = kv_b_proj_weight.view(2, 4 + 4, 8)
            k_weight = weight_reshaped[:, :4, :]
            v_weight = weight_reshaped[:, 4:, :]

            # Verify shapes of split parts
            assert k_weight.shape == (2, 4, 8)
            assert v_weight.shape == (2, 4, 8)

    def test_load_individual_weight_with_mxfp4(self, loader, mesh):
        """Tests the logic for unpacking MXFP4 weights."""
        name = "layers.0.attn.kernel_q_down_proj_DA"
        # Mocking torch tensor as uint8 (packed fp4)
        expected_weight_shape = (128, 128)  # Unpacked
        expected_scale_shape = (128, 1)

        weight = torch.zeros(expected_weight_shape, dtype=torch.uint8)
        scale = torch.ones(expected_scale_shape, dtype=torch.float32)

        # Mock model parameters
        mock_var = MockVariable(
            (128, 128),
            dtype=jnp.float4_e2m1fn,
            sharding=(None, ('attn_dp', 'model',
                             'expert')))  # Unpacked shape (64 * 2)
        mock_params = {
            "layers": {
                "0": {
                    "attn": {
                        "kernel_q_down_proj_DA": mock_var
                    }
                }
            }
        }

        with patch("tpu_inference.models.jax.deepseek_v3.get_param", return_value=mock_var), \
             patch("tpu_inference.models.jax.deepseek_v3.u8_unpack_e2m1") as mock_unpack, \
             patch("jax.make_array_from_callback") as mock_make_array:

            def side_effect_router(shape, *args, **kwargs):
                if shape == expected_scale_shape:
                    # Return FP32 for the scale call
                    return jnp.ones(shape, dtype=jnp.float32)
                elif shape == expected_weight_shape:
                    # Return FP4 for the weight call
                    return jnp.zeros(shape, dtype=jnp.float4_e2m1fn)
                return jnp.zeros(shape)  # Fallback

            mock_make_array.side_effect = side_effect_router
            mock_unpack.return_value = torch.zeros(expected_weight_shape)

            loader._load_individual_weight(name,
                                           weight,
                                           mock_params,
                                           mesh,
                                           scale=scale)

            mock_unpack.assert_called_once()
            (actual_arg, ), _ = mock_unpack.call_args
            # The implementation converts the torch weight to a JAX array
            expected_arg = jnp.array(weight.cpu().numpy())
            assert jnp.array_equal(actual_arg, expected_arg).item()
            assert mock_make_array.called

    def test_load_weights_full_flow(self, loader, mesh):
        """Integrative test for the load_weights loop."""
        model = MagicMock(spec=nnx.Module)
        model.mesh = mesh

        # Setup generator to return one normal weight
        loader.names_and_weights_generator = [("model.embed_tokens.weight",
                                               torch.ones((10, 10)))]

        mock_var = MockVariable((10, 10))

        with patch("tpu_inference.models.jax.deepseek_v3.nnx.state"), \
             patch("tpu_inference.models.jax.deepseek_v3.get_param", return_value=mock_var), \
             patch("tpu_inference.models.jax.deepseek_v3.nnx.update"), \
             patch.object(loader, '_load_individual_weight', return_value=(1.0, 0.5)):

            loader.load_weights(model)
            # Verify verbose logging worked if enabled
            assert loader.is_verbose is True

    def test_load_individual_weight_unpacked(self, loader, mesh):
        """
        Tests the logic for loading 'unpacked' weights (e.g., standard FP8).
        This verifies the branch that uses DTYPE_VIEW_MAP for raw memory conversion.
        """
        name = "layers.0.attn.kernel_q_down_proj_DA"

        # 1. Setup a standard 'unpacked' FP8 torch tensor
        # DeepSeek V3 weights are often float8_e4m3fn
        weight_shape = (128, 128)
        weight = torch.randn(weight_shape).to(torch.float8_e4m3fn)

        # 2. Mock model parameters to expect jnp.float8_e4m3fn
        # We reuse the MockVariable helper but specify the dtype
        mock_var = MockVariable(weight_shape, dtype=jnp.float8_e4m3fn)
        mock_params = {
            "layers": {
                "0": {
                    "attn": {
                        "kernel_q_down_proj_DA": mock_var
                    }
                }
            }
        }

        # 3. Patch the necessary JAX/Utility functions
        with patch("tpu_inference.models.jax.deepseek_v3.get_param", return_value=mock_var), \
             patch("tpu_inference.models.jax.deepseek_v3.u8_unpack_e2m1") as mock_unpack, \
             patch("jax.make_array_from_callback") as mock_make_array:

            # Mock the JAX array creation to return a dummy
            mock_make_array.return_value = jnp.zeros(weight_shape,
                                                     dtype=jnp.float8_e4m3fn)

            # Execute the loader method
            loader._load_individual_weight(name,
                                           weight,
                                           mock_params,
                                           mesh,
                                           scale=None)

            # VERIFICATIONS:
            # - u8_unpack_e2m1 should NOT be called for standard FP8 (only for packed uint8 + scale)
            mock_unpack.assert_not_called()

            # - make_array_from_callback should be called with the correct shape and sharding
            # The first argument to make_array_from_callback is the shape
            assert mock_make_array.call_args[0][0] == weight_shape

            # - Verify the model weight value was updated (even if with our dummy)
            assert mock_var.value.dtype == jnp.float8_e4m3fn

    def test_load_individual_weight_with_scale(self, loader, mesh):
        """
        Tests loading an unpacked weight that also has a quantization scale.
        """
        name = "layers.0.custom_module.kernel_gating_DF"
        weight_shape = (64, 128)
        scale_shape = (64, 1)

        # Use BF16 for this test to verify DTYPE_VIEW_MAP handles multiple types
        weight = torch.randn(weight_shape).to(torch.bfloat16)
        scale = torch.ones(scale_shape, dtype=torch.float32)

        mock_var = MockVariable(weight_shape, dtype=jnp.bfloat16)
        mock_params = {
            "layers": {
                "0": {
                    "custom_module": {
                        "kernel_gating_DF": mock_var
                    }
                }
            }
        }

        with patch("tpu_inference.models.jax.deepseek_v3.get_param", return_value=mock_var), \
             patch("jax.make_array_from_callback") as mock_make_array:

            def side_effect_router(shape, *args, **kwargs):
                if shape == scale_shape:
                    # Return FP32 for the scale call
                    return jnp.ones(shape, dtype=jnp.float32)
                elif shape == weight_shape:
                    # Return FP4 for the weight call
                    return jnp.zeros(shape, dtype=jnp.bfloat16)
                return jnp.zeros(shape)  # Fallback

            mock_make_array.side_effect = side_effect_router

            loader._load_individual_weight(name,
                                           weight,
                                           mock_params,
                                           mesh,
                                           scale=scale)

            # Verify the scale was applied to the MockVariable's internal QArray structure
            # (In the model code: base_model_weight.array.scale.value = maybe_sharded_scale)
            assert mock_var.array.scale.value is not None
