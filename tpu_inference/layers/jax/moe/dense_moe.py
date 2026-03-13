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
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Sharding
from jaxtyping import Float

from tpu_inference.layers.common.process_weights.moe_weights import \
    UnfusedMoEWeights
from tpu_inference.layers.jax.moe.utils import modeling_flax_utils


def dense_moe_fwd(weights: UnfusedMoEWeights, x_TD: Float,
                  cast_dtype: jnp.dtype, activation_ffw_td: Sharding,
                  hidden_act: str, full_weights_TE: jax.Array) -> jax.Array:
    """Forward pass of the dense Moe layer where we don't pre-apply the weights.

    TODO (jacobplatin): we probably want to support quantization at some point.

    Args:
        weights: The weights of the dense Moe layer.
        x_TD: Input array of shape (sequence_length, d_model).
        cast_dtype: The dtype to cast the input to.
        activation_ffw_td: The sharding of the activation.
        hidden_act: The activation function to use.
        full_weights_TE: The full weights of the dense Moe layer.

    Returns:
        The output of the dense Moe layer.
    """
    x_TD = jnp.asarray(x_TD, cast_dtype)
    x_TD = nnx.with_sharding_constraint(x_TD, activation_ffw_td)
    with jax.named_scope("gating"):
        gating_TEF = jnp.einsum('TD,EDF -> TEF', x_TD, weights.w1_weight)
        activated_gating_TEF = modeling_flax_utils.ACT2FN[hidden_act](
            gating_TEF)
    with jax.named_scope("up_projection"):
        up_proj_TEF = jnp.einsum('TD,EDF -> TEF', x_TD, weights.w2_weight)
    fuse_TEF = activated_gating_TEF * up_proj_TEF
    with jax.named_scope("down_projection"):
        down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                   weights.w3_weight)
    with jax.named_scope("sum"):
        output_TD = jnp.einsum('TED,TE -> TD', down_proj_TED, full_weights_TE)
    return output_TD.astype(cast_dtype)


def dense_moe_fwd_preapply_router_weights(
        weights: UnfusedMoEWeights, x_TD: Float, cast_dtype: jnp.dtype,
        activation_ffw_ted: Sharding, hidden_act: str,
        full_weights_TE: jax.Array) -> jax.Array:
    """
    Forward pass of the dense Moe layer where we pre-apply the weights.

    TODO (jacobplatin): we probably want to support quantization at some point.

    Args:
        weights: The weights of the dense Moe layer.
        x_TD: Input array of shape (sequence_length, d_model).
        cast_dtype: The dtype to cast the input to.
        activation_ffw_td: The sharding of the activation.
        hidden_act: The activation function to use.
        full_weights_TE: The weights of the router.

    Returns:
        The output of the dense Moe layer.
    """

    num_experts = full_weights_TE.shape[-1]
    x_TED = jnp.repeat(x_TD[:, None, :], num_experts, 1)
    x_TED = jnp.asarray(x_TED, cast_dtype) * full_weights_TE[..., None]
    x_TED = nnx.with_sharding_constraint(x_TED, activation_ffw_ted)

    with jax.named_scope("gating"):
        gating_TEF = jnp.einsum('TED,EDF -> TEF', x_TED, weights.w1_weight)
        activated_gating_TEF = modeling_flax_utils.ACT2FN[hidden_act](
            gating_TEF)
    with jax.named_scope("up_projection"):
        up_proj_TEF = jnp.einsum('TED,EDF -> TEF', x_TED, weights.w2_weight)

    fuse_TEF = activated_gating_TEF * up_proj_TEF
    with jax.named_scope("down_projection"):
        down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                   weights.w3_weight)
    return down_proj_TED.sum(axis=1).astype(cast_dtype)


def dense_moe_func(weights: UnfusedMoEWeights, x_TD: jax.Array,
                   gating_output: Tuple[jax.Array, jax.Array],
                   cast_dtype: jnp.dtype, num_local_experts: int,
                   apply_expert_weight_before_computation: bool,
                   activation_ffw_td: Sharding, activation_ffw_ted: Sharding,
                   hidden_act: str) -> jax.Array:
    """
    Forward pass of the dense MoE layer.  This is a naive implementation
    and thus should not be used in production.

    TODO (jacobplatin): we probably want to support quantization at some point.

    Args:
        weights: The weights of the dense Moe layer.
        x_TD: Input array of shape (sequence_length, d_model).
        gating_output: The gating output of the dense Moe layer.
        indices_TX: The indices of the experts to use.
        cast_dtype: The dtype to cast the input to.
        num_local_experts: The number of local experts.
        apply_expert_weight_before_computation: Whether to apply the expert weights before computing the output.
        activation_ffw_td: The sharding of the activation.
        activation_ffw_ted: The sharding of the activation, used for the
            pre-apply weights case.
        hidden_act: The activation function to use.

    Returns:
        The output of the dense Moe layer.
    """
    assert isinstance(
        weights,
        UnfusedMoEWeights), "Expected unfused weights for DENSE_MAT backend"

    weights_TX, indices_TX = gating_output
    one_hot_indices_TXE = jax.nn.one_hot(indices_TX,
                                         num_classes=num_local_experts,
                                         dtype=cast_dtype)
    full_weights_TE = jnp.sum(one_hot_indices_TXE * weights_TX[..., None],
                              axis=1)
    # Some models use the routing scores to weight the data instead of
    # weighting the expert outputs.
    if apply_expert_weight_before_computation:
        with jax.named_scope("pre_computing_weight"):
            return dense_moe_fwd_preapply_router_weights(
                weights=weights,
                x_TD=x_TD,
                cast_dtype=cast_dtype,
                activation_ffw_ted=activation_ffw_ted,
                hidden_act=hidden_act,
                full_weights_TE=full_weights_TE)
    else:
        return dense_moe_fwd(weights=weights,
                             x_TD=x_TD,
                             cast_dtype=cast_dtype,
                             activation_ffw_td=activation_ffw_td,
                             hidden_act=hidden_act,
                             full_weights_TE=full_weights_TE)
