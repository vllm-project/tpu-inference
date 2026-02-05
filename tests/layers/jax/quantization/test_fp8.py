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

import jax
import numpy as np
import pytest
from flax import nnx

from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.quantization.fp8 import Fp8Config


@pytest.fixture
def rngs():
    return nnx.Rngs(42)


class TestFp8BlockwiseJaxLinear:

    @pytest.mark.parametrize("in_features,out_features", [(128, 64),
                                                          (256, 128)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_linear_forward_correctness(self, in_features, out_features,
                                        use_bias, batch_size, rngs):
        hf_quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
        quant_config = Fp8Config(hf_quant_config)

        # Initialize quantized layer
        layer = JaxLinear(
            input_size=in_features,
            output_size=out_features,
            rngs=rngs,
            use_bias=use_bias,
            quant_config=quant_config,
        )

        # Use a dummy mesh for testing
        devices = jax.devices()
        mesh = jax.sharding.Mesh(np.array(devices), ('device', ))
        with jax.set_mesh(mesh):
            # Process weights in mesh context
            layer.quant_method.process_weights_after_loading(layer)

            # Prepare input
            x = jax.random.normal(rngs.params(), (batch_size, in_features))

            # Forward pass
            output = layer(x)

        assert output.shape == (batch_size, out_features)

    @pytest.mark.parametrize("kernel_shape", [(128, 8, 16), (256, 32, 32)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_einsum_forward_correctness(self, kernel_shape, use_bias,
                                        batch_size, rngs):
        hf_quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [8, 16],
        }
        quant_config = Fp8Config(hf_quant_config)

        layer = JaxEinsum(
            einsum_str='TD,DNH->TNH',
            kernel_shape=kernel_shape,
            rngs=rngs,
            bias_shape=kernel_shape[1:] if use_bias else None,
            quant_config=quant_config,
        )

        # Use a dummy mesh for testing
        devices = jax.devices()
        mesh = jax.sharding.Mesh(np.array(devices), ('device', ))
        with jax.set_mesh(mesh):
            # Process weights in mesh context
            layer.quant_method.process_weights_after_loading(layer)

            # Prepare input (B, D)
            x = jax.random.normal(rngs.params(), (batch_size, kernel_shape[0]))

            # Forward pass
            output = layer(x)

        # Output shape should be (B, N, H)
        expected_shape = (batch_size, ) + kernel_shape[1:]
        assert output.shape == expected_shape
