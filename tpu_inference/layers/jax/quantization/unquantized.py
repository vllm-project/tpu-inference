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

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
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


import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.jax.moe.dense_moe import (
    dense_moe_fwd, dense_moe_fwd_preapply_router_weights)
from tpu_inference.layers.jax.moe.utils import MoEBackend


class UnquantizedFusedMoEMethod(QuantizeMethodBase,
                                jax_common.UnquantizedFusedMoEMethod):
    """Unquantized method for JAX FusedMoELayer.
    """

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, layer.activation_ffw_td)
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            raise ValueError
            # router_logits_TE = layer.router(x_TD)
            # ep_axis_name = layer.efd_sharding[0]
            # output_TD = fused_ep_moe(
            #     mesh=layer.mesh,
            #     tokens=x_TD,
            #     w1=layer.kernel_gating_upproj_E2DF.value,
            #     w2=layer.kernel_down_proj_EFD.value,
            #     gating_output=router_logits_TE,
            #     top_k=layer.router.num_experts_per_tok,
            #     ep_axis_name=ep_axis_name,
            #     renormalize_topk_logits=layer.renormalize,
            #     act_fn=layer.hidden_act,
            #     **layer.block_size,
            # )
            # return output_TD
        elif layer.moe_backend == MoEBackend.VLLM_MOE:
            router_logits_TE = layer.router(x_TD)
            output_TD = fused_moe_func(
                hidden_states=x_TD,
                w1=layer.kernel_gating_upproj_EFD.value,
                w2=layer.kernel_down_proj_EDF.value,
                w1_bias=layer.w1_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w1_scale,
                w2_scale=layer.w2_scale,
                gating_output=router_logits_TE,
                topk=layer.router.num_experts_per_tok,
                renormalize=layer.renormalize,
                mesh=layer.mesh,
                use_ep=layer.num_expert_parallelism > 1,
                activation=layer.hidden_act,
            )
            return output_TD
        else:
            weights_TX, indices_TX = layer.router(x_TD)

            # if layer.moe_backend == MoEBackend.MEGABLX_GMM or layer.moe_backend == MoEBackend.RAGGED_DOT:
            #     in_specs = (
            #         PartitionSpec(),  # replicated MoE instance
            #         PartitionSpec(*layer.activation_ffw_td),  # Sharded x_TD
            #         PartitionSpec(),  # Replicated router_weights_TX
            #         PartitionSpec(),  # Replicated selected_experts_TX
            #         PartitionSpec(*layer.edf_sharding),  # Sharded gating kernel
            #         PartitionSpec(
            #             *layer.edf_sharding),  # Sharded up-projection kernel
            #         PartitionSpec(
            #             *layer.efd_sharding),  # Sharded down-projection kernel
            #     )
            #     out_specs = PartitionSpec(*layer.activation_ffw_td)

            #     mapped_moe_fwd = partial(
            #         jax.experimental.shard_map.shard_map,
            #         mesh=layer.mesh,
            #         in_specs=in_specs,
            #         out_specs=out_specs,
            #         check_rep=False)(sparse_moe_distributed_fwd)

            #     kernel_gating_EDF = layer._process_weight_for_qwix(
            #         layer.kernel_gating_EDF,
            #         channelwise_axes=[0, 2],
            #         tiled_axes={})
            #     kernel_up_proj_EDF = layer._process_weight_for_qwix(
            #         layer.kernel_up_proj_EDF,
            #         channelwise_axes=[0, 2],
            #         tiled_axes={})
            #     kernel_down_proj_EFD = layer._process_weight_for_qwix(
            #         layer.kernel_down_proj_EFD,
            #         channelwise_axes=[0, 2],
            #         tiled_axes={})

            #     return mapped_moe_fwd(layer, x_TD, weights_TX, indices_TX,
            #                           kernel_gating_EDF, kernel_up_proj_EDF,
            #                           kernel_down_proj_EFD)

            # Dense Matmul, elif
            if layer.moe_backend == MoEBackend.DENSE_MAT:
                one_hot_indices_TXE = jax.nn.one_hot(
                    indices_TX,
                    num_classes=layer.num_local_experts,
                    dtype=layer.dtype)
                full_weights_TE = jnp.sum(one_hot_indices_TXE *
                                          weights_TX[..., None],
                                          axis=1)
                # Some models use the routing scores to weight the data instead of
                # weighting the expert outputs.
                if layer.apply_expert_weight_before_computation:
                    with jax.named_scope("pre_computing_weight"):
                        return dense_moe_fwd_preapply_router_weights(
                            layer, x_TD, full_weights_TE)
                else:
                    return dense_moe_fwd(layer, x_TD, full_weights_TE)
            else:
                raise ValueError


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer)
            return UnquantizedLinearMethod(linear_config)
        if isinstance(layer, JaxMoE):
            # TODO: pass a config
            return UnquantizedFusedMoEMethod()
        return None
