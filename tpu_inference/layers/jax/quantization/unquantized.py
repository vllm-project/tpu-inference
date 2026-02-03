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
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.common.quantization import unquantized as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
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


class UnquantizedFusedMoEMethod(QuantizeMethodBase):
    """
    Unquantized method for JAXMoE layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}

    def process_weights_after_loading(self, layer):
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            if layer.edf_sharding:
                e2df_sharding = (layer.edf_sharding[0], None,
                                 layer.edf_sharding[1], layer.edf_sharding[2])
            # fuse the weights into w13: [Gate, Up]
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # stack to create a 4d matrix
            w13_val = jnp.stack([w_gate, w_up], axis=1)

            layer.kernel_gating_upproj_E2DF = nnx.Param(w13_val,
                                                        sharding=e2df_sharding)

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

            ep_axis_name = layer.efd_sharding[0]

            self.extra_backend_kwargs = {
                "ep_axis_name": ep_axis_name,
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 64,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            }

        elif layer.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # Fuse the weights into w13: [Gate, Up]
            w13_val = jnp.concatenate([w_gate, w_up], axis=-1)

            layer.kernel_gating_upproj_EFD = nnx.Param(w13_val)

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

            # Rename for consistency
            w_down = layer.kernel_down_proj_EFD.value
            layer.kernel_down_proj_EDF = nnx.Param(w_down)
            del layer.kernel_down_proj_EFD

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxMoE)

        # Fused weight backends
        if layer.moe_backend in [
                MoEBackend.FUSED_MOE, MoEBackend.GMM_EP, MoEBackend.GMM_TP
        ]:
            x_TD = jnp.asarray(x, layer.dtype)
            x_TD = nnx.with_sharding_constraint(x_TD, layer.activation_ffw_td)
            router_logits_TE = layer.router(x_TD)

            # TODO; unfused too
            w13_weight = layer.kernel_gating_upproj_E2DF.value if layer.moe_backend == MoEBackend.FUSED_MOE else layer.kernel_gating_upproj_EFD.value
            w2_weight = layer.kernel_down_proj_EFD.value if layer.moe_backend == MoEBackend.FUSED_MOE else layer.kernel_down_proj_EDF.value
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=None,
                w13_bias=None,  # TODO?
                w2_weight=w2_weight,
                w2_weight_scale=None,
                w2_bias=None,  # TODO?
            )
        else:
            raise ValueError
        return moe_apply(layer, x_TD, router_logits_TE, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer)
            return UnquantizedLinearMethod(linear_config)
        if isinstance(layer, JaxMoE):
            # TODO (jacobplatin): do we need to pass a config here?
            return UnquantizedFusedMoEMethod()
        return None
