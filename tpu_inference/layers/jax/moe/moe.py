# Copyright 2025 Google LLC
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
from dataclasses import InitVar, dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.sharding import PartitionSpec
from jaxtyping import Float
from qwix._src.providers import ptq

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.moe import MoEBackend, fused_moe_func
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.moe.dense_moe import (
    dense_moe_fwd, dense_moe_fwd_preapply_router_weights)
from tpu_inference.layers.jax.moe.sparse_moe import sparse_moe_distributed_fwd
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.models.jax.utils.qwix.qwix_utils import \
    manually_quantize_qwix_weight

modeling_flax_utils = FlaxUtils()


@dataclass(kw_only=True)
class CombineExperts(nnx.Module):
    """Combines expert outputs with router weights.

    Supports `TED,TE -> TD` when passed expert outputs, using float32
    accumulation for numerical stability, then casting back to the target
    dtype.
    """

    dtype: jnp.dtype

    def __call__(self, expert_outputs_TED: Float, weights_TE: Float) -> Float:
        with jax.named_scope("combine_experts"):
            output_TD = jnp.einsum(
                "TED,TE -> TD",
                expert_outputs_TED.astype(jnp.float32),
                weights_TE.astype(jnp.float32),
                precision="float32",
            )

        return output_TD.astype(self.dtype)


@dataclass(kw_only=True)
class Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
    """
    dtype: jnp.dtype
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    router_act: str
    rngs: InitVar[nnx.Rngs]
    activation_ffw_td: Sharding
    ed_sharding: Sharding
    random_init: bool = False
    moe_backend: MoEBackend = MoEBackend.DENSE_MAT

    def __call__(self, x_TD: Float):
        """Routes tokens to experts.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).

        Returns:
            A tuple containing:
                - normalized_weights_TX: Normalized weights for selected experts, shape (sequence_length, num_experts_per_tok).
                - selected_experts_TX: Indices of selected experts, shape (sequence_length, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        router_act = modeling_flax_utils.ACT2FN[self.router_act]
        router_logits_TE = jnp.einsum('TD,DE -> TE', x_TD,
                                      self.kernel_DE.value)

        #TODO: Refactor the Router so that it will always only return router_logits_TE
        if self.moe_backend in [
                MoEBackend.FUSED_MOE, MoEBackend.GMM_EP, MoEBackend.GMM_TP
        ]:
            return router_logits_TE
        else:
            weights_TX, selected_experts_TX = jax.lax.top_k(
                router_logits_TE, self.num_experts_per_tok)
            if self.router_act != "sigmoid":  # sigmoid does not accept axis argument.
                normalized_weights_TX = router_act(weights_TX.astype(
                    self.dtype),
                                                   axis=-1)
            else:
                normalized_weights_TX = router_act(
                    weights_TX.astype(self.dtype))
            return normalized_weights_TX, selected_experts_TX

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights) for routing."""
        shape = (self.hidden_size, self.num_experts)
        self.kernel_DE = create_param(rngs,
                                      shape=shape,
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)


# --- Main Class for MoE ---
@dataclass(kw_only=True)
class JaxMoE(JaxModule):
    """Mixture-of-Experts (MoE) Routed MLP Layer.

    This module implements a MoE layer with a router and multiple expert MLPs.

    Attributes:
        router: The Router module.
    """
    dtype: jnp.dtype
    num_local_experts: int
    hidden_size: int
    intermediate_size_moe: int
    hidden_act: str
    rngs: InitVar[nnx.Rngs]
    router: nnx.Module
    mesh: jax.sharding.Mesh
    # --- Sharding Config ---
    activation_ffw_td: Sharding
    activation_ffw_ted: Sharding
    edf_sharding: Sharding
    efd_sharding: Sharding
    e2df_sharding: Sharding = ()

    # --- Flags & Configs ---
    apply_expert_weight_before_computation: bool
    expert_axis_name: str
    num_expert_parallelism: int
    use_ep: bool
    random_init: bool = False
    moe_backend: MoEBackend = MoEBackend.DENSE_MAT

    # --- Sparse MoE Specific Attributes ---
    num_experts_per_tok: int = 1  # Required for Sparse, optional/derived for Dense
    tile_size: tuple[int, int, int] = (128, 128, 128)
    # NOTE: this is only needed for SparseMoE
    qwix_quantized_weight_dtype: Optional[jnp.dtype] = None

    # --- MoE Kernel Specific Attributes ---
    renormalize: bool = True

    # ---- Quantization Specific Attributes ----
    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""

    def __call__(self, x_TD: Float):
        """Performs the forward pass of the MoE layer.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).

        Returns:
            Output array of shape (sequence_length, d_model) after passing through MoE.
        """
        # TODO (jacobplatin): wire this up
        if self.quant_method is not None:
            return self.quant_method.apply_jax(self, x_TD)

        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        if self.moe_backend == MoEBackend.FUSED_MOE:
            router_logits_TE = self.router(x_TD)
            ep_axis_name = self.efd_sharding[0]
            output_TD = fused_ep_moe(
                mesh=self.mesh,
                tokens=x_TD,
                w1=self.kernel_gating_upproj_E2DF.value,
                w2=self.kernel_down_proj_EFD.value,
                gating_output=router_logits_TE,
                top_k=self.router.num_experts_per_tok,
                ep_axis_name=ep_axis_name,
                renormalize_topk_logits=self.renormalize,
                act_fn=self.hidden_act,
                **self.block_size,
            )
            return output_TD
        elif self.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            router_logits_TE = self.router(x_TD)
            # TODO (jacobplatin): the current GMM kernel expects that w1/w2 have the second and third dimensions
            # transposed, but this is likely not optimal for DeepSeek, so we will need to fix this
            # in the future
            output_TD = fused_moe_func(
                hidden_states=x_TD,
                w1=self.kernel_gating_upproj_EDF.value,
                w2=self.kernel_down_proj_EFD.value,
                w1_bias=self.w1_bias,
                w2_bias=self.w2_bias,
                w1_scale=self.w1_scale,
                w2_scale=self.w2_scale,
                gating_output=router_logits_TE,
                topk=self.router.num_experts_per_tok,
                renormalize=self.renormalize,
                mesh=self.mesh,
                use_ep=self.use_ep,
                activation=self.hidden_act,
            )
            return output_TD
        else:
            weights_TX, indices_TX = self.router(x_TD)

            if self.moe_backend == MoEBackend.MEGABLX_GMM or self.moe_backend == MoEBackend.RAGGED_DOT:
                # NOTE: for the qwix_quantized_weight_dtype case, we make the spec a tuple of 2 PartitionSpecs
                # since the first entry corresponds to the weight and the second entry corresponds to the scale.
                # For the scale, we don't shard on the "D" dimmension because this is the subchannel dimmension
                if self.qwix_quantized_weight_dtype:
                    gating_up_proj_spec = (PartitionSpec(*self.edf_sharding),
                                           PartitionSpec(
                                               self.edf_sharding[0], None,
                                               self.edf_sharding[2]))
                    down_proj_spec = (PartitionSpec(*self.efd_sharding),
                                      PartitionSpec(self.efd_sharding[0], None,
                                                    self.efd_sharding[2]))
                else:
                    gating_up_proj_spec = PartitionSpec(*self.edf_sharding)
                    down_proj_spec = PartitionSpec(*self.efd_sharding)

                in_specs = (
                    PartitionSpec(),  # replicated MoE instance
                    PartitionSpec(*self.activation_ffw_td),  # Sharded x_TD
                    PartitionSpec(),  # Replicated router_weights_TX
                    PartitionSpec(),  # Replicated selected_experts_TX
                    gating_up_proj_spec,  # Sharded gating kernel
                    gating_up_proj_spec,  # Sharded up-projection kernel
                    down_proj_spec,  # Sharded down-projection kernel
                )
                out_specs = PartitionSpec(*self.activation_ffw_td)

                mapped_moe_fwd = partial(
                    jax.experimental.shard_map.shard_map,
                    mesh=self.mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_rep=False)(sparse_moe_distributed_fwd)

                # TODO (jacobplatin): this is needed because of issues with Qwix quantizing the `shard_map` in SpraseMatmul
                # Basically, during the abstract pass, we need to manually quantize the weights here for Qwix, but we'll
                # override the actual weight/scale during loading (we just need to make sure Qwix quantizes the weight
                # in the first place).
                kernel_gating_EDF = self._process_weight_for_qwix(
                    "kernel_gating_EDF",
                    self.kernel_gating_EDF,
                    channelwise_axes=[0, 2],
                    tiled_axes={})
                kernel_up_proj_EDF = self._process_weight_for_qwix(
                    "kernel_up_proj_EDF",
                    self.kernel_up_proj_EDF,
                    channelwise_axes=[0, 2],
                    tiled_axes={})
                kernel_down_proj_EFD = self._process_weight_for_qwix(
                    "kernel_down_proj_EFD",
                    self.kernel_down_proj_EFD,
                    channelwise_axes=[0, 2],
                    tiled_axes={})

                return mapped_moe_fwd(self, x_TD, weights_TX, indices_TX,
                                      kernel_gating_EDF, kernel_up_proj_EDF,
                                      kernel_down_proj_EFD)

            # Dense Matmul
            elif self.moe_backend == MoEBackend.DENSE_MAT:
                one_hot_indices_TXE = jax.nn.one_hot(
                    indices_TX,
                    num_classes=self.num_local_experts,
                    dtype=self.dtype)
                full_weights_TE = jnp.sum(one_hot_indices_TXE *
                                          weights_TX[..., None],
                                          axis=1)
                # Some models use the routing scores to weight the data instead of
                # weighting the expert outputs.
                if self.apply_expert_weight_before_computation:
                    with jax.named_scope("pre_computing_weight"):
                        return dense_moe_fwd_preapply_router_weights(
                            self, x_TD, full_weights_TE)
                else:
                    return dense_moe_fwd(self, x_TD, full_weights_TE)

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the kernels (weights) for the router and experts (gating, up-projection, and down-projection layers)."""
        # TODO (jacobplatin): wire this up
        if self.quant_config is None:
            self.quant_method = None
        elif (quant_method :=
              self.quant_config.get_quant_method(self, prefix=self.prefix)):
            assert isinstance(quant_method, QuantizeMethodBase)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self)
        else:
            self.quant_method = None

        E = self.num_local_experts
        D = self.hidden_size
        F = self.intermediate_size_moe

        if self.moe_backend == MoEBackend.FUSED_MOE:
            if self.edf_sharding:
                self.e2df_sharding = (self.edf_sharding[0], None,
                                      self.edf_sharding[1],
                                      self.edf_sharding[2])
            self.kernel_gating_upproj_E2DF = create_param(
                rngs,
                shape=(E, 2, D, F),
                dtype=self.dtype,
                sharding=self.e2df_sharding,
                random_init=self.random_init)
            self.kernel_down_proj_EFD = create_param(
                rngs,
                shape=(E, F, D),
                dtype=self.dtype,
                sharding=self.efd_sharding,
                random_init=self.random_init)
            self.block_size = {
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 64,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            }
        elif self.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            # TODO (jacobplatin): the current GMM kernel expects that w1/w2 have the second and third
            # dimensions transposed, but this is likely not optimal for DeepSeek, so we will
            # need to fix this in the future
            self.kernel_gating_upproj_EDF = create_param(
                rngs,
                shape=(E, D, 2 * F),
                dtype=self.dtype,
                sharding=self.efd_sharding,
                random_init=self.random_init)
            self.kernel_down_proj_EFD = create_param(
                rngs,
                shape=(E, F, D),
                dtype=self.dtype,
                sharding=self.edf_sharding,
                random_init=self.random_init)
        else:
            self.kernel_gating_EDF = create_param(rngs,
                                                  shape=(E, D, F),
                                                  dtype=self.dtype,
                                                  sharding=self.edf_sharding,
                                                  random_init=self.random_init)
            self.kernel_up_proj_EDF = create_param(
                rngs,
                shape=(E, D, F),
                dtype=self.dtype,
                sharding=self.edf_sharding,
                random_init=self.random_init)
            self.kernel_down_proj_EFD = create_param(
                rngs,
                shape=(E, F, D),
                dtype=self.dtype,
                sharding=self.efd_sharding,
                random_init=self.random_init)

        # Default MoE has no bias vectors
        self.w1_bias, self.w2_bias = (None, None)

        # TODO: Add quantization scale params for VLLM MoE kernel
        self.w1_scale, self.w2_scale = (None, None)

        # Derive if data is sharded by expert
        self.data_axis_name = self.activation_ffw_td[0]
        self.is_batch_sharded_by_expert = (
            self.expert_axis_name is not None) and (self.expert_axis_name
                                                    == self.data_axis_name)

    def _process_weight_for_qwix(self,
                                 name,
                                 weight_param,
                                 channelwise_axes=[],
                                 tiled_axes={}):
        """
        Extracts weight value, applies quantization if needed,
        and returns the underlying array.
        """
        weight = weight_param.value

        if self.qwix_quantized_weight_dtype:
            if not isinstance(weight, ptq.WithAux):
                weight = manually_quantize_qwix_weight(
                    name, weight, self.qwix_quantized_weight_dtype,
                    channelwise_axes, tiled_axes, "absmax")
            return (weight.array.qvalue, weight.array.scale)

        return weight
