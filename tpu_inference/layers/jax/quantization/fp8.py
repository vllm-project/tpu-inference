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
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


class UnquantizedLinearMethod(QuantizeMethodBase,
                              jax_common.UnquantizedLinearMethod):
    """Unquantized method for JAX Linear layer.
    """

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer.__name__):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(
                    x, layer.weight.value,
                    layer.bias.value if layer.bias else None)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class Fp8MoEMethod(QuantizeMethodBase, jax_common.UnquantizedFusedMoEMethod):
    """Unquantized method for JAX FusedMoELayer.
    """

    def create_weight_jax(self, layer: JaxModule) -> jax.Array:
        assert isinstance(layer, JaxMoE)
        # TODO: rngs, shape, sharding, dtype
        layer.w13_weight_scale_inv = create_param()
        layer.w2_weight_scale_inv = create_param()

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer.__name__):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(
                    x, layer.weight.value,
                    layer.bias.value if layer.bias else None)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class Fp8QuantizationConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxMoE):
            return Fp8MoEMethod(layer.moe_config)
        return None
