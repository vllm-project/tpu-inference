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

from unittest.mock import MagicMock, patch

import jax
import jax.sharding
import torch
import torchax

from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    reparametrize, shard_model_to_tpu)


def test_shard_model_to_tpu_simplification():
    # Setup dummy model with a parameter and a buffer
    class DummyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(10))
            self.register_buffer("my_buffer", torch.zeros(5))

    model = DummyModel()
    dummy_mesh = MagicMock()

    PREFIX = "tpu_inference.layers.vllm.process_weights.cleanup_sharding"
    # Mock jax device calls to prevent TPU multi-host initialization hangs
    with (
            patch(PREFIX + ".jax.devices") as mock_jax_devices,
            patch(PREFIX + ".jax.default_device"),
            patch(PREFIX + "._shard_module_to_tpu") as mock_shard_module,
            patch(PREFIX + "._tensor_is_in_cpu") as mock_is_cpu,
            patch(PREFIX + "._shard_tensor_to_tpu_replicated") as
            mock_shard_replicated,
    ):
        mock_jax_devices.return_value = ["mock_cpu_device"]

        # Pretend the parameter is on CPU (needs sharding) but buffer is not
        def mock_is_in_cpu_side_effect(tensor):
            return tensor is model.weight

        mock_is_cpu.side_effect = mock_is_in_cpu_side_effect

        # Pretend sharded tensor returns a special string
        mock_shard_replicated.return_value = "sharded_tensor"

        result = shard_model_to_tpu(model, dummy_mesh)

        # Check that special sharding was called for any LoRA/custom layers
        mock_shard_module.assert_called_once_with(model, dummy_mesh)

        # Verify exactly the expected dictionary keys
        assert set(result.keys()) == {"weight", "my_buffer"}

        # Verify the parameter was sharded (because is_in_cpu returned True)
        assert result["weight"] == "sharded_tensor"

        # Verify the buffer passed straight through (because is_in_cpu returned False)
        assert result["my_buffer"] is model.my_buffer

        # Verify _shard_tensor_to_tpu_replicated was only called once
        mock_shard_replicated.assert_called_once_with(model.weight, dummy_mesh)


def test_extract_and_reparametrize():

    class DummyModel(torch.nn.Module):

        def __init__(self, param: int):
            super().__init__()
            self.some_param = torch.nn.Parameter(torch.tensor(param))
            self.tied_param = self.some_param
            self.some_buffer = torch.nn.Buffer(torch.tensor(1.))

        def forward(self, val: torch.Tensor) -> torch.Tensor:
            """Return val * param + 1"""
            return val * self.tied_param + self.some_buffer

    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("x", ))
    model_foo = DummyModel(param=3.)
    model_bar = DummyModel(param=5.)

    params_and_buffers = shard_model_to_tpu(model_foo, mesh)

    with (torchax.default_env(), reparametrize(model_bar, params_and_buffers)):
        # 7 = 2 * 3 + 1
        out = model_bar(torch.tensor(2.))
        assert torch.allclose(torch.tensor(7.), out)
