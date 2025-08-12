from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jaxtyping import Float

from tpu_commons.models.jax.common.base import create_param
from tpu_commons.models.jax.common.layers import FlaxUtils

modeling_flax_utils = FlaxUtils()


@dataclass(kw_only=True)
class Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
        mesh: The JAX device mesh for distributed computation.
        quant: Optional configuration for quantization.
    """
    mesh: Mesh
    dtype: jnp.dtype
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    router_act: str
    activation_ffw_td: NamedSharding
    ed_sharding: NamedSharding
    random_init: bool = False
    quant: Any | None = None

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
        weights_TX, selected_experts_TX = jax.lax.top_k(
            router_logits_TE, self.num_experts_per_tok)
        if self.router_act != "sigmoid":  # sigmoid does not accept axis argument.
            normalized_weights_TX = router_act(weights_TX.astype(self.dtype),
                                               axis=-1)
        else:
            normalized_weights_TX = router_act(weights_TX.astype(self.dtype))
        return normalized_weights_TX, selected_experts_TX

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights) for routing."""
        shape = (self.hidden_size, self.num_experts)
        self.kernel_DE = create_param(rngs,
                                      shape=shape,
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)


@dataclass(kw_only=True)
class MoE(nnx.Module):
    """Mixture-of-Experts (MoE) Routed MLP Layer.

    This module implements a MoE layer with a router and multiple expert MLPs.

    Attributes:
        mesh: The JAX mesh for device sharding.
        router: The Router module.
        quant: Optional configuration for quantization.
    """
    mesh: Mesh
    dtype: jnp.dtype
    num_local_experts: int
    apply_expert_weight_before_computation: bool
    hidden_size: int
    intermediate_size_moe: int
    hidden_act: str
    router: nnx.Module
    activation_ffw_td: NamedSharding
    activation_ffw_ted: NamedSharding
    edf_sharding: NamedSharding
    efd_sharding: NamedSharding
    random_init: bool = False
    quant: Any | None = None

    def __call__(self, x_TD: Float):
        """Performs the forward pass of the MoE layer.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).

        Returns:
            Output array of shape (sequence_length, d_model) after passing through MoE.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        weights_TX, indices_TX = self.router(x_TD)
        one_hot_indices_TXE = jax.nn.one_hot(
            indices_TX, num_classes=self.num_local_experts, dtype=self.dtype)
        full_weights_TE = jnp.sum(one_hot_indices_TXE * weights_TX[..., None],
                                  axis=1)

        # Some models use the routing scores to weight the data instead of
        # weighting the expert outputs.
        if self.apply_expert_weight_before_computation:
            with jax.named_scope("pre_computing_weight"):
                return self._moe_fwd_preapply_router_weights(
                    x_TD, full_weights_TE)
        else:
            return self._moe_fwd(x_TD, full_weights_TE)

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the kernels (weights) for the router and experts (gating, up-projection, and down-projection layers)."""

        # Generate router kernels
        self.router.generate_kernel(rngs)

        D = self.hidden_size
        F = self.intermediate_size_moe
        shape_gating = (self.num_local_experts, D, F)
        shape_up = (self.num_local_experts, D, F)
        shape_down = (self.num_local_experts, F, D)

        self.kernel_gating_EDF = create_param(rngs,
                                              shape=shape_gating,
                                              dtype=self.dtype,
                                              sharding=self.edf_sharding,
                                              random_init=self.random_init)
        self.kernel_up_proj_EDF = create_param(rngs,
                                               shape=shape_up,
                                               dtype=self.dtype,
                                               sharding=self.edf_sharding,
                                               random_init=self.random_init)
        self.kernel_down_proj_EFD = create_param(rngs,
                                                 shape=shape_down,
                                                 dtype=self.dtype,
                                                 sharding=self.efd_sharding,
                                                 random_init=self.random_init)

    def _moe_fwd_preapply_router_weights(self, x_TD: jax.Array, weights_TE):
        """Performs the forward pass of the MoE experts with router weights pre-applied to the inputs.

        Args:
            x_TD: Input array for the experts, shape (sequence_length, hidden_size).
            weights_TE: Router weights, shape (sequence_length, num_experts).

        Returns:
            Output array of shape (sequence_length, d_model).
        """
        # Data needs to be replicated since it will be weighted by the router
        # scores before being passed to each expert.
        num_experts = weights_TE.shape[-1]
        x_TED = jnp.repeat(x_TD[:, None, :], num_experts, 1)
        weights_TED = weights_TE[..., None]
        x_TED = jnp.asarray(x_TED, self.dtype)

        with jax.named_scope("activation_expert_weighting"):
            x_TED = x_TED * weights_TED

        x_TED = nnx.with_sharding_constraint(x_TED, self.activation_ffw_ted)
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TED,EDF -> TEF', x_TED,
                                    self.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TED,EDF -> TEF', x_TED,
                                     self.kernel_up_proj_EDF.value)

        fuse_TEF = activated_gating_TEF * up_proj_TEF

        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                       self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_TD = down_proj_TED.sum(axis=1)
        return output_TD.astype(self.dtype)

    def _moe_fwd(self, x_TD: Float, weights):
        """Performs the basic forward pass of the MoE experts without dropping or megablocks.

        Args:
            x_TD: Input array for the experts, shape (sequence_length, d_model).
            weights: Weights for combining expert outputs, shape (sequence_length, num_experts).

        Returns:
            Output array of shape (sequence_length, d_model).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TD,EDF -> TEF', x_TD,
                                    self.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TD,EDF -> TEF', x_TD,
                                     self.kernel_up_proj_EDF.value)

        fuse_TEF = activated_gating_TEF * up_proj_TEF

        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                       self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_TD = jnp.einsum('TED,TE -> TD', down_proj_TED, weights)
        return output_TD.astype(self.dtype)
