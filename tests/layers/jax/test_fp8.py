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

import os
import tempfile
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from safetensors.torch import save_file
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.fp8 import (Fp8Config,
                                                       Fp8LinearMethod)
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator


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
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def rng():
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestFp8Linear:

    def test_fp8_linear_init(self, mesh):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))

        quant_config = get_tpu_quantization_config(vllm_config)
        assert isinstance(quant_config, Fp8Config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        assert isinstance(layer.quant_method, Fp8LinearMethod)
        assert hasattr(layer, "weight_scale")
        assert layer.weight_scale.value.shape == (output_dim, )
        assert layer.weight_scale.value.dtype == jnp.float32
        assert hasattr(layer.weight_scale, "weight_loader")

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            hidden_states = jnp.ones((batch_size, input_dim))
            out = layer(hidden_states)
            assert out.shape == (batch_size, output_dim)

    def test_fp8_linear_correctness(self, mesh, rng):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))
        quant_config = get_tpu_quantization_config(vllm_config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        k1, k2, k3 = jax.random.split(rng, 3)
        w_val = jax.random.normal(k1, (input_dim, output_dim),
                                  dtype=jnp.float32)
        s_val = jax.random.uniform(k2, (output_dim, ), dtype=jnp.float32)

        layer.weight.value = w_val
        layer.weight_scale.value = s_val

        hidden_states = jax.random.uniform(k3, (batch_size, input_dim),
                                           dtype=jnp.float32)

        effective_w = w_val * s_val
        expected = jnp.dot(hidden_states, effective_w)

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            out = layer(hidden_states)
            assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_fp8_linear_scale_loader_logic(self, mesh):
        """
        Unit test to verify the fp8 weight loader logic.
        
        This isolates the scale loading mechanism. The full loading process (including
        the main weight and file interaction) is covered by `test_fp8_linear_load_from_safetensors`.
        """
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))
        quant_config = get_tpu_quantization_config(vllm_config)

        input_dim = 16
        output_dim = 32
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Create a dummy torch tensor for scale
        torch_scale = torch.rand((output_dim, ), dtype=torch.float32)

        layer.weight_scale.mesh = mesh
        layer.weight_scale.weight_loader(layer.weight_scale, torch_scale)
        jax_scale = layer.weight_scale.value
        np_scale_from_torch = torch_scale.numpy()

        assert jnp.allclose(jax_scale, np_scale_from_torch)

    def test_fp8_linear_load_from_safetensors(self, mesh):
        """Load weights and scales from a safetensors file simulating a quantized checkpoint."""

        class TorchFP8Model(torch.nn.Module):
            """Simulated FP8 model in PyTorch (using Float32 for simplicity)."""

            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim,
                                              output_dim,
                                              bias=False)
                self.linear.weight_scale = torch.nn.Parameter(
                    torch.rand((output_dim, ), dtype=torch.float32))

        class JaxFP8Model(JaxModule, LoadableWithIterator):

            def __init__(self, rngs, quant_config):
                super().__init__()
                self.linear = JaxLinear(input_size=16,
                                        output_size=32,
                                        rngs=rngs,
                                        quant_config=quant_config)

        input_dim = 16
        output_dim = 32

        # Create dummy weights and scales in torch
        torch_model = TorchFP8Model(input_dim, output_dim)
        torch.nn.init.normal_(torch_model.linear.weight)

        # Save to safetensors
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, "model.safetensors")
            state_dict = torch_model.state_dict()
            save_file(state_dict, tmp_file_path)

            # Create a dummy config.json to satisfy ModelConfig validation
            import json
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "qwen2", "hidden_size": 16}, f)

            vllm_config = VllmConfig(
                model_config=ModelConfig(model=tmpdir, quantization="fp8"))
            quant_config = get_tpu_quantization_config(vllm_config)

            with mesh:
                jax_model = JaxFP8Model(rngs=nnx.Rngs(0),
                                        quant_config=quant_config)
                jax_model.linear.weight.mesh = mesh
                jax_model.linear.weight_scale.mesh = mesh

                loader = get_model_loader(
                    LoadConfig(load_format="safetensors"))

                mock_model_config = MagicMock()
                mock_model_config.model = tmpdir
                mock_model_config.quantization = "fp8"
                mock_model_config.revision = None

                loader.load_weights(jax_model, mock_model_config)

        # Verify Weight
        expected_weight = torch_model.linear.weight.detach().numpy().T
        np.testing.assert_allclose(jax_model.linear.weight.value,
                                   expected_weight,
                                   rtol=1e-5)

        # Verify Scale
        expected_scale = torch_model.linear.weight_scale.detach().numpy()
        np.testing.assert_allclose(jax_model.linear.weight_scale.value,
                                   expected_scale,
                                   rtol=1e-5)
