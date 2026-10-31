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

import torch
import torchax

from tpu_inference.layers.common import utils as common_utils
from tpu_inference.layers.common.utils import local_cpu_devices
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    _tensor_is_in_cpu, shard_model_to_tpu)


def test_shard_model_to_tpu_simplification():
    # Setup dummy model with a parameter and a buffer
    class DummyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(10))
            self.register_buffer("my_buffer", torch.zeros(5))

    model = DummyModel()
    dummy_mesh = MagicMock()

    # Mock jax device calls to prevent TPU multi-host initialization hangs
    with (
            patch(
                "tpu_inference.layers.vllm.process_weights.cleanup_sharding.jax.devices"
            ) as mock_jax_devices,
            patch(
                "tpu_inference.layers.vllm.process_weights.cleanup_sharding.jax.default_device"
            ),
    ):
        mock_jax_devices.return_value = ["mock_cpu_device"]

        # Mock the internal functions that do the actual sharding/device checks
        with patch(
                "tpu_inference.layers.vllm.process_weights.cleanup_sharding._shard_module_to_tpu"
        ) as mock_shard_module:
            with patch(
                    "tpu_inference.layers.vllm.process_weights.cleanup_sharding._tensor_is_in_cpu"
            ) as mock_is_cpu:
                with patch(
                        "tpu_inference.layers.vllm.process_weights.cleanup_sharding._shard_tensor_to_tpu_replicated"
                ) as mock_shard_replicated:
                    # Pretend the parameter is on CPU (needs sharding) but buffer is not
                    def mock_is_in_cpu_side_effect(tensor):
                        return tensor is model.weight

                    mock_is_cpu.side_effect = mock_is_in_cpu_side_effect

                    # Pretend sharded tensor returns a special string
                    mock_shard_replicated.return_value = "sharded_tensor"

                    result = shard_model_to_tpu(model, dummy_mesh)

                    # Check that special sharding was called for any LoRA/custom layers
                    mock_shard_module.assert_called_once_with(
                        model, dummy_mesh)

                    # Verify exactly the expected dictionary keys
                    assert set(result.keys()) == {"weight", "my_buffer"}

                    # Verify the parameter was sharded (because is_in_cpu returned True)
                    assert result["weight"] == "sharded_tensor"

                    # Verify the buffer passed straight through (because is_in_cpu returned False)
                    assert result["my_buffer"] is model.my_buffer

                    # Verify _shard_tensor_to_tpu_replicated was only called once
                    mock_shard_replicated.assert_called_once_with(
                        model.weight, dummy_mesh)


def test_local_cpu_devices_scoping_and_mock_fallback():
    fake_local_cpu = MagicMock(name="local_cpu_4")
    fake_global_cpu = MagicMock(name="global_cpu_0")
    with patch.object(common_utils.jax,
                      "local_devices",
                      return_value=[fake_local_cpu]) as mock_local:
        assert local_cpu_devices(common_utils.jax) == [fake_local_cpu]
        mock_local.assert_called_once_with(backend="cpu")

    # When a unit test mocks jax.devices without mocking jax.local_devices,
    # local_cpu_devices falls back to the mocked jax.devices("cpu").
    with patch.object(common_utils.jax,
                      "devices",
                      return_value=[fake_global_cpu]) as mock_devices:
        assert local_cpu_devices(common_utils.jax) == [fake_global_cpu]
        mock_devices.assert_called_once_with("cpu")


def test_tensor_is_in_cpu_multihost_non_zero_process():
    fake_local_cpu = MagicMock(name="local_cpu_4")
    fake_global_cpu = MagicMock(name="global_cpu_0")
    fake_tpu_device = MagicMock(name="tpu_4")

    cpu_torchax_tensor = MagicMock(spec=torchax.tensor.Tensor)
    cpu_torchax_tensor.jax_device = fake_local_cpu

    tpu_torchax_tensor = MagicMock(spec=torchax.tensor.Tensor)
    tpu_torchax_tensor.jax_device = fake_tpu_device

    with (
            patch(
                "tpu_inference.layers.vllm.process_weights.cleanup_sharding.jax.local_devices",
                return_value=[fake_local_cpu],
            ),
            patch(
                "tpu_inference.layers.vllm.process_weights.cleanup_sharding.jax.devices",
                return_value=[fake_global_cpu, fake_local_cpu],
            ),
    ):
        assert _tensor_is_in_cpu(cpu_torchax_tensor) is True
        assert _tensor_is_in_cpu(tpu_torchax_tensor) is False
        assert _tensor_is_in_cpu(torch.ones(4)) is True
