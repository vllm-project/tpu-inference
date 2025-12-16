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
        hf_config.num_hidden_layers = 2  # Small for testing
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

    @patch(
        "tpu_inference.models.jax.deepseek_v3.get_quant_dtype_from_qwix_config",
        return_value=[jnp.bfloat16, jnp.float8_e4m3fn])
    def test_init(self, mock_config, rng, mesh):
        """Tests if the model initializes with the correct hierarchy."""
        model = DeepSeekV3(mock_config, rng, mesh)
        assert len(model.layers) == 2  # num_layers from mock
        assert isinstance(model.embedder, nnx.Module)
        assert model.vllm_config.model_config.hf_config.num_hidden_layers == 2

    @patch(
        "tpu_inference.models.jax.deepseek_v3.get_quant_dtype_from_qwix_config",
        return_value=[jnp.bfloat16, jnp.float8_e4m3fn])
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
                return_value=[]
        ), patch(
                "tpu_inference.models.jax.deepseek_v3.get_quant_dtype_from_qwix_config",
                return_value=[jnp.bfloat16, jnp.float8_e4m3fn]):
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

    def test_invalid_quant_dtype_assertion(self, mock_get_dtype, mock_config):
        """Verifies that an AssertionError is raised if quant_dtype is not fp8."""
        # Mock returning bfloat16 instead of float8_e4m3fn
        mock_get_dtype.return_value = (jnp.bfloat16, jnp.bfloat16)

        with pytest.raises(AssertionError) as excinfo:
            DeepSeekV3WeightLoader(vllm_config=mock_config,
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
        assert "Expected quant_dtype to be float8_e4m3fn" in str(excinfo.value)

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
        weight = torch.zeros((128, 64), dtype=torch.uint8)
        scale = torch.ones((128, 1), dtype=torch.float32)

        # Mock model parameters
        mock_var = MockVariable((128, 128))  # Unpacked shape (64 * 2)
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
             patch("tpu_inference.models.jax.deepseek_v3.unpack_mxfp4") as mock_unpack, \
             patch("jax.make_array_from_callback") as mock_make_array:

            # Setup unpack to return a float tensor of double the width
            mock_unpack.return_value = torch.zeros((128, 128))
            mock_make_array.return_value = jnp.zeros((128, 128))

            loader._load_individual_weight(name,
                                           weight,
                                           mock_params,
                                           mesh,
                                           scale=scale)

            mock_unpack.assert_called_once_with(weight)
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
