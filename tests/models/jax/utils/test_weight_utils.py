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
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.qwen3_moe import Qwen3MoeForCausalLM
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator


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

    @pytest.mark.parametrize(
        "model_name",
        [
            # Qwen3MoeForCausalLM has Linear/MoE layers which we care about.
            # NOTE: the linear layer does not have bias.
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        ])
    def test_process_weights_after_loading(
            self,
            model_name,
            # following are defined in conftest.py
            rng,
            mesh,
            mock_vllm_config):
        """Tests process_weights_after_loading are called.
        """
        kv_cache_type = "auto"
        vllm_config = mock_vllm_config(model_name, kv_cache_type)
        # No need to load full model.
        vllm_config.model_config.hf_config.num_hidden_layers = 2
        vllm_config.load_config.load_format = "skip_layers_model_loader_for_test"
        vllm_config.load_config.num_layers_to_load_for_test = 2
        # In extreme case, mimic each safetensor file only has 1 tensor
        vllm_config.load_config.safetensor_granularity_for_test = 1

        init_pp_distributed_environment(
            ip="",
            rank=0,
            world_size=1,
            device=jax.devices()[0],
            need_pp=False,
        )
        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        model_config = vllm_config.model_config

        with jax.set_mesh(mesh):
            model = Qwen3MoeForCausalLM(vllm_config, rng, mesh)
        # load weights from HF model
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            loader.load_weights(model, model_config)

        assert model is not None
