from dataclasses import dataclass, field, make_dataclass
from typing import Any, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import DTypeLike, Float
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import (HuggingFaceArgNames,
                                                     RouterType)
from tpu_commons.models.jax.common.layers import FlaxUtils
from tpu_commons.models.jax.common.moe.deepseek_moe import \
    DeepSeekV3RoutingConfig
from tpu_commons.models.jax.common.sharding import ShardingConfig

modeling_flax_utils = FlaxUtils()

RouterConfig = make_dataclass(
    "RouterConfig",
    [(HuggingFaceArgNames.HIDDEN_SIZE.value, int),
     (HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value, int),
     (HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value, int),
     ("router_type", RouterType), ("expert_capacity", int),
     ("dtype", DTypeLike),
     ("vllm_config", VllmConfig, field(repr=False, default=None)),
     ("router_act", str, "softmax")],
    bases=(Config, ))
RouterConfig.__doc__ = f"""Configuration for the Router module.

     Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value}: The total number of experts.
        {HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value}: The number of experts each token is routed to.
         router_type: The type of router to use (e.g., 'top_k').
         router_act: The activation function to use for normalizing router scores.
         expert_capacity: The maximum number of tokens an expert can process. Defaults to -1 (no capacity limit).
        dtype: The data type to use for computations.
        vllm_config: The VLLM config containing any overrides to apply."""


@dataclass
class Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
        cfg: The RouterConfig object.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """
    mesh: Mesh
    dtype: jnp.dtype
    hidden_size: int
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    num_experts: int
    num_experts_per_tok: int
    router_act: str
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the Router module by creating sharding configurations and generating the router kernel."""
        self.create_sharding()

    def __call__(self, x_TD: Float, op_mode):
        """Routes tokens to experts.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            A tuple containing:
                - normalized_weights_TX: Normalized weights for selected experts, shape (sequence_length, num_experts_per_tok).
                - selected_experts_TX: Indices of selected experts, shape (sequence_length, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD,
                                            self.activation_ffw_td[op_mode])
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
        self.kernel_DE = self.param_factory.create_kernel_param(
            rngs, shape=shape, dtype=self.dtype, sharding=self.ed_sharding)

    def create_sharding(self):
        """Creates sharding configurations for activations and kernel."""
        mode_dependent_attrs = [
            "activation_ffw_td",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_rules,
                                              attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(*generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.ed_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_router_de))

        return


MoEConfig = make_dataclass(
    "MoEConfig",
    [(HuggingFaceArgNames.HIDDEN_SIZE.value, int),
     (HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value, int),
     (HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value, int),
     (HuggingFaceArgNames.HIDDEN_ACT.value, int),
     ("apply_expert_weight_before_computation", bool),
     ("router", Union[RouterConfig, DeepSeekV3RoutingConfig]),
     ("dtype", DTypeLike),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
MoEConfig.__doc__ = f"""Configuration for the Mixture-of-Experts (MoE) layer.

     Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value}: The hidden size of each expert's MLP.
        {HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value}: The total number of experts.
        {HuggingFaceArgNames.HIDDEN_ACT.value}: The activation function to use within the experts.
        apply_expert_weight_before_computation: Whether to apply expert weights before computation. Defaults to False.
        router: The config for the Router module.
        dtype: The data type to use for computations.
        vllm_config: The VLLM config containing any overrides to apply."""


@dataclass
class MoE(nnx.Module):
    """Mixture-of-Experts (MoE) Routed MLP Layer.

    This module implements a MoE layer with a router and multiple expert MLPs.

    Attributes:
        cfg: The MoEConfig object.
        mesh: The JAX mesh for device sharding.
        param_factory: A factory for creating and initializing model parameters.
        router: The Router module.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    dtype: jnp.dtype
    num_local_experts: int
    apply_expert_weight_before_computation: bool
    hidden_size: int
    intermediate_size_moe: int
    hidden_act: str
    router: nnx.Module
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the MoE module by creating sharding configurations and generating expert kernels."""
        self.router.create_sharding()
        self.create_sharding()

    def __call__(self, x_TD: Float, op_mode):
        """Performs the forward pass of the MoE layer.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            Output array of shape (sequence_length, d_model) after passing through MoE.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD,
                                            self.activation_ffw_td[op_mode])
        weights_TX, indices_TX = self.router(x_TD, op_mode)
        one_hot_indices_TXE = jax.nn.one_hot(
            indices_TX, num_classes=self.num_local_experts, dtype=self.dtype)
        full_weights_TE = jnp.sum(one_hot_indices_TXE * weights_TX[..., None],
                                  axis=1)

        # Some models use the routing scores to weight the data instead of
        # weighting the expert outputs.
        if self.apply_expert_weight_before_computation:
            with jax.named_scope("pre_computing_weight"):
                return self._moe_fwd_preapply_router_weights(
                    x_TD, full_weights_TE, op_mode)
        else:
            return self._moe_fwd(x_TD, full_weights_TE, op_mode)

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the kernels (weights) for the router and experts (gating, up-projection, and down-projection layers)."""

        # Generate router kernels
        self.router.generate_kernel(rngs)

        D = self.hidden_size
        F = self.intermediate_size_moe
        shape_gating = (self.num_local_experts, D, F)
        shape_up = (self.num_local_experts, D, F)
        shape_down = (self.num_local_experts, F, D)

        self.kernel_gating_EDF = self.param_factory.create_kernel_param(
            rngs,
            shape=shape_gating,
            dtype=self.dtype,
            sharding=self.edf_sharding)
        self.kernel_up_proj_EDF = self.param_factory.create_kernel_param(
            rngs, shape=shape_up, dtype=self.dtype, sharding=self.edf_sharding)
        self.kernel_down_proj_EFD = self.param_factory.create_kernel_param(
            rngs,
            shape=shape_down,
            dtype=self.dtype,
            sharding=self.efd_sharding)

    def _moe_fwd_preapply_router_weights(self, x_TD: jax.Array, weights_TE,
                                         op_mode):
        """Performs the forward pass of the MoE experts with router weights pre-applied to the inputs.

        Args:
            x_TD: Input array for the experts, shape (sequence_length, hidden_size).
            weights_TE: Router weights, shape (sequence_length, num_experts).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

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

        x_TED = nnx.with_sharding_constraint(x_TED,
                                             self.activation_ffw_ted[op_mode])
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

    def _moe_fwd(self, x_TD: Float, weights, op_mode):
        """Performs the basic forward pass of the MoE experts without dropping or megablocks.

        Args:
            x_TD: Input array for the experts, shape (sequence_length, d_model).
            weights: Weights for combining expert outputs, shape (sequence_length, num_experts).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            Output array of shape (sequence_length, d_model).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD,
                                            self.activation_ffw_td[op_mode])
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

    def create_sharding(self):
        """Creates sharding configurations for activations and expert kernels."""
        mode_dependent_attrs = [
            "activation_ffw_td",
            "activation_ffw_ted",
        ]

        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_rules,
                                              attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(*generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.edf_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_weights_edf))
        self.efd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_weights_efd))

        return
