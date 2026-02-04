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
from torch import nn
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, load_blockwise_fp8_scale)


class TorchMLP(nn.Module):
    """MLP implemented with PyTorch."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2, 6)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(6, 2, bias=True)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class JaxMLP(JaxModule, LoadableWithIterator):
    """MLP implemented with JAX."""

    def __init__(self, rngs):
        super().__init__()
        self.w1 = JaxLinear(2, 6, rngs)
        self.act = nnx.relu
        self.w2 = JaxLinear(6, 2, rngs, use_bias=True)

    def __call__(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class TestJaxAutoWeightsLoader:

    def test_load_from_safetensors(self):
        """Load weights from a safetensors file saved from a PyTorch model.
        """
        torch_model = TorchMLP()
        with torch.no_grad():
            torch_model.w1.weight.fill_(1.1)
            torch_model.w2.weight.fill_(0.9)
            torch_model.w2.bias.fill_(0.1)

        # Save the PyTorch model weights to a safetensors file. Load them
        # into the JAX model.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, "model.safetensors")
            save_file(torch_model.state_dict(), tmp_file_path)

            devices = jax.local_devices()
            mesh = Mesh(devices, axis_names=('p', ))
            with jax.set_mesh(mesh):
                jax_model = JaxMLP(rngs=nnx.Rngs(0))

                model_config = MagicMock()
                model_config.quantization = None
                model_config.model = tmpdir
                model_config.revision = None

                loader = get_model_loader(
                    LoadConfig(load_format="safetensors"))
                loader.load_weights(jax_model, model_config)

        np.testing.assert_allclose(torch_model.w1.weight.T.detach().numpy(),
                                   jax_model.w1.weight.value)
        np.testing.assert_allclose(torch_model.w2.weight.T.detach().numpy(),
                                   jax_model.w2.weight.value)

        # Forward pass to verify correctness.
        input_values = [[0.1, 0.2], [0.3, 0.4]]
        torch_input = torch.tensor(input_values)
        jax_input = np.array(input_values)
        torch_output = torch_model(torch_input).detach().numpy()
        jax_output = jax_model(jax_input)
        np.testing.assert_allclose(torch_output,
                                   jax_output,
                                   rtol=1e-3,
                                   atol=1e-2)


class TestLoadBlockwiseFp8Scale:

    def test_load_blockwise_fp8_scale(self):
        output_dim = 4
        n_blocks = 2

        devices = jax.local_devices()
        mesh = Mesh(devices, axis_names=('p', ))

        with jax.set_mesh(mesh):
            # Case 1: 1D flattened input
            torch_scale_1d = torch.arange(n_blocks * output_dim,
                                          dtype=torch.float32)
            jax_param = nnx.Param(
                jnp.zeros((n_blocks, 1, output_dim), dtype=jnp.float32))

            load_blockwise_fp8_scale(jax_param, torch_scale_1d, output_dim,
                                     n_blocks, "test_param")

            expected_scale = jnp.arange(n_blocks * output_dim,
                                        dtype=jnp.float32).reshape(
                                            n_blocks, 1, output_dim)
            np.testing.assert_allclose(jax_param.value, expected_scale)

            # Case 2: 2D input
            torch_scale_2d = torch.arange(n_blocks * output_dim,
                                          dtype=torch.float32).reshape(
                                              n_blocks, output_dim)
            jax_param = nnx.Param(
                jnp.zeros((n_blocks, 1, output_dim), dtype=jnp.float32))

            load_blockwise_fp8_scale(jax_param, torch_scale_2d, output_dim,
                                     n_blocks, "test_param")

            np.testing.assert_allclose(jax_param.value, expected_scale)

            # Case 3: Size mismatch (1D)
            torch_scale_wrong_size = torch.arange(n_blocks * output_dim + 1,
                                                  dtype=torch.float32)
            with pytest.raises(ValueError,
                               match="Checkpoint scale size mismatch"):
                load_blockwise_fp8_scale(jax_param, torch_scale_wrong_size,
                                         output_dim, n_blocks, "test_param")

            # Case 4: Shape mismatch (2D)
            torch_scale_wrong_shape = torch.zeros((n_blocks + 1, output_dim))
            with pytest.raises(ValueError,
                               match="Checkpoint scale shape mismatch"):
                load_blockwise_fp8_scale(jax_param, torch_scale_wrong_shape,
                                         output_dim, n_blocks, "test_param")

            # Case 5: Wrong number of dimensions
            torch_scale_3d = torch.zeros((n_blocks, output_dim, 1))
            with pytest.raises(
                    ValueError,
                    match="Checkpoint scale has unexpected number of dimensions"
            ):
                load_blockwise_fp8_scale(jax_param, torch_scale_3d, output_dim,
                                         n_blocks, "test_param")