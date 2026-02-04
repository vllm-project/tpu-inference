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

from typing import Optional

import jax

from tpu_inference.layers.common.quantization import unquantized as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


class UnquantizedLinearMethod(QuantizeMethodBase,
                              jax_common.UnquantizedLinearMethod):
    """Unquantized method for JAX Linear layer.
    """

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(
                    x,
                    layer.weight.value,
                    layer.bias.value if layer.bias else None,
                    einsum_str=layer.einsum_str)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(
                enable_sp=False, output_sizes=[layer.kernel_shape[1]])
            return UnquantizedLinearMethod(linear_config)
        return None
