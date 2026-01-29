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

from tpu_inference.layers.common.fused_moe import MoEBackend, moe_apply
from tpu_inference.layers.common.quantization import unquantized as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (QuantFusedMoEConfig,
                                                           QuantizationConfig)

if TYPE_CHECKING:
    from tpu_inference.layers.vllm.process_weights.fused_moe_weights import \
        FusedMoEWeights


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


import jax.numpy as jnp
from flax import nnx


class UnquantizedFusedMoEMethod(QuantizeMethodBase,
                                jax_common.UnquantizedFusedMoEMethod):
    """Unquantized method for JAX FusedMoELayer.
    """

    def process_weights_after_loading(layer):
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            raise ValueError
            # if self.edf_sharding:
            #     self.e2df_sharding = (self.edf_sharding[0], None,
            #                           self.edf_sharding[1],
            #                           self.edf_sharding[2])
            # self.kernel_gating_upproj_E2DF = create_param(
            #     rngs,
            #     shape=(E, 2, D, F),
            #     dtype=self.dtype,
            #     sharding=self.e2df_sharding,
            #     random_init=self.random_init)
            # self.kernel_down_proj_EFD = create_param(
            #     rngs,
            #     shape=(E, F, D),
            #     dtype=self.dtype,
            #     sharding=self.efd_sharding,
            #     random_init=self.random_init)
            # self.block_size = {
            #     "bt": 32,
            #     "bf": 512,
            #     "bd1": 512,
            #     "bd2": 512,
            #     "btc": 64,
            #     "bfc": 256,
            #     "bd1c": 256,
            #     "bd2c": 256,
            # }
        elif layer.moe_backend == MoEBackend.VLLM_MOE:
            # TODO (jacobplatin): the current GMM kernel expects that w1/w2 have the second and third
            # dimensions transposed, but this is likely not optimal for DeepSeek, so we will
            # need to fix this in the future
            pass
            # self.kernel_gating_upproj_EFD = create_param(
            #     rngs,
            #     shape=(E, D, 2 * F),
            #     dtype=self.dtype,
            #     sharding=self.efd_sharding,
            #     random_init=self.random_init)
            # self.kernel_down_proj_EDF = create_param(
            #     rngs,
            #     shape=(E, F, D),
            #     dtype=self.dtype,
            #     sharding=self.edf_sharding,
            #     random_init=self.random_init)

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxMoE)

        if layer.moe_backend == MoEBackend.VLLM_MOE:
            x_TD = jnp.asarray(x, layer.dtype)
            x_TD = nnx.with_sharding_constraint(x_TD, layer.activation_ffw_td)

            # TODO
            router_logits_TE = layer.router(x_TD)

            # TODO; unfused too
            weights = FusedMoEWeights(
                w13_weight=layer.kernel_gating_upproj_EFD.value,
                w13_weight_scale=None,
                w13_bias=None,  # TODO?
                w2_weight=layer.kernel_down_proj_EDF.value,
                w2_weight_scale=None,
                w2_bias=None,  # TODO?
            )
        else:
            raise ValueError

        return moe_apply(layer, x_TD, router_logits_TE, weights,
                         self.moe_backend, self.mesh,
                         self.extra_backend_kwargs)


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer)
            return UnquantizedLinearMethod(linear_config)
        if isinstance(layer, JaxMoE):
            # TODO: pass a config
            moe_config = QuantFusedMoEConfig()
            return UnquantizedFusedMoEMethod()
        return None
