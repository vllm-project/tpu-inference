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

from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.linear import LinearBase

from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig


def test_weight_only_int8_reports_unsupported_scheme():
    config = VllmCompressedTensorsConfig.from_config({
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                },
                "input_activations": None,
            }
        },
    })
    config.vllm_config = MagicMock()
    config.mesh = MagicMock()
    layer = MagicMock(spec=LinearBase)
    layer.input_size = 16
    layer.output_size = 16

    with pytest.raises(NotImplementedError,
                       match="No compressed-tensors compatible scheme"):
        config.get_scheme(layer, layer_name="Linear")
