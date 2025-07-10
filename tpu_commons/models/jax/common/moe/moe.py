from dataclasses import dataclass, field, make_dataclass
from typing import Any

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
from tpu_commons.models.jax.common.sharding import ShardingConfig

modeling_flax_utils = FlaxUtils()

RouterConfig = make_dataclass(
    "RouterConfig",
    [(HuggingFaceArgNames.HIDDEN_SIZE.value, int),
     (HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value, int),
     (HuggingFaceArgNames.INTERMEDIATE_SIZE.value, int),
     (HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value, int),
     (HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value, int),
     ("router_type", RouterType), (HuggingFaceArgNames.HIDDEN_ACT.value, str),
     ("expert_capacity", int), ("routed_bias", bool),
     ("routed_scaling_factor", float), ("dtype", DTypeLike),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
RouterConfig.__doc__ = f"""Configuration for the Router module.

     Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value}: The hidden size of the expert.
        {HuggingFaceArgNames.INTERMEDIATE_SIZE.value}: The hidden size of MLP layers.
        {HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value}: The total number of experts.
        {HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value}: The number of experts each token is routed to.
         router_type: The type of router to use (e.g., 'top_k').
        {HuggingFaceArgNames.HIDDEN_ACT.value}: The activation function to use.
         expert_capacity: The maximum number of tokens an expert can process. Defaults to -1 (no capacity limit).
         routed_bias: Whether to use a bias in the router. Defaults to False. # DeepSeek related. Could be removed
         routed_scaling_factor: Scaling factor for routed weights. Defaults to 1.0.
        dtype: The data type to use for computations. Defaults to jnp.float32.
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
    cfg: RouterConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the Router module by creating sharding configurations and generating the router kernel."""
        self.create_sharding()

    def __call__(self, x: Float, op_mode):
        """Routes tokens to experts.

        Args:
            x: Input array of shape (sequence_length, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            A tuple containing:
                - normalized_weights_TX: Normalized weights for selected experts, shape (sequence_length, num_experts_per_tok).
                - selected_experts_TX: Indices of selected experts, shape (sequence_length, num_experts_per_tok).
        """
        x = jnp.asarray(x, self.cfg.dtype)
        x_TD = nnx.with_sharding_constraint(x, self.activation_ffw_td[op_mode])
        num_experts_per_tok = getattr(
            self.cfg, HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN)
        router_logits_TE = jnp.einsum('TD,DE -> TE', x_TD,
                                      self.kernel_DE.value)
        activated_gating_TF = nnx.softmax(router_logits_TE.astype(jnp.float32),
                                          axis=-1)
        weights_TX, selected_experts_TX = jax.lax.top_k(
            activated_gating_TF, num_experts_per_tok)
        normalized_weights_TX = nnx.softmax(weights_TX.astype(self.cfg.dtype),
                                            axis=-1)
        return normalized_weights_TX, selected_experts_TX

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights) for routing."""
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        num_experts = getattr(self.cfg,
                              HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value)
        shape = (D, num_experts)
        self.kernel_DE = self.param_factory.create_kernel_param(
            rngs, shape=shape, dtype=self.cfg.dtype, sharding=self.ed_sharding)

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
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
MoEConfig.__doc__ = f"""Configuration for the Mixture-of-Experts (MoE) layer.

     Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value}: The hidden size of each expert's MLP.
        {HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value}: The total number of experts.
        {HuggingFaceArgNames.HIDDEN_ACT.value}: The activation function to use within the experts.
        apply_expert_weight_before_computation: Whether to apply expert weights before computation. Defaults to False.
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
    cfg: MoEConfig
    mesh: Mesh
    param_factory: ParamFactory
    router: Router
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the MoE module by creating sharding configurations and generating expert kernels."""
        self.create_sharding()

    def __call__(self, x: Float, op_mode):
        """Performs the forward pass of the MoE layer.

        Args:
            x: Input array of shape (sequence_length, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            Output array of shape (sequence_length, d_model) after passing through MoE.
        """
        x = jnp.asarray(x, jnp.float32)
        x_TD = nnx.with_sharding_constraint(x, self.activation_ffw_td[op_mode])
        weights_TX, indices_TX = self.router(x_TD, op_mode)
        num_experts = getattr(self.cfg,
                              HuggingFaceArgNames.NUM_LOCAL_EXPERTS.value)
        one_hot_indices_TXE = jax.nn.one_hot(indices_TX,
                                             num_classes=num_experts,
                                             dtype=self.cfg.dtype)
        full_weights_TE = jnp.sum(one_hot_indices_TXE * weights_TX[..., None],
                                  axis=2)

        if self.cfg.apply_expert_weight_before_computation:
            with jax.named_scope("pre_computing_weight"):
                # need optimization for the out-product
                weighted_x_TXD = jnp.einsum('TD,TX->TXD', x_TD, weights_TX)
            # TODO: need fix the mod_fwd call for pre_weighting
            return self._moe_fwd(weighted_x_TXD,
                                 jnp.ones_like(full_weights_TE), op_mode)
        else:
            return self._moe_fwd(x_TD, full_weights_TE, op_mode)

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the kernels (weights) for the gating, up-projection, and down-projection layers of each expert."""
        num_experts = getattr(self.cfg,
                              HuggingFaceArgNames.NUM_LOCAL_EXPERT.value)
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        F = getattr(self.cfg, HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value)
        shape_gating = (num_experts, D, F)
        shape_up = (num_experts, D, F)
        shape_down = (num_experts, F, D)

        self.kernel_gating_EDF = self.param_factory.create_kernel_param(
            rngs,
            shape=shape_gating,
            dtype=self.cfg.dtype,
            sharding=self.edf_sharding)
        self.kernel_up_proj_EDF = self.param_factory.create_kernel_param(
            rngs,
            shape=shape_up,
            dtype=self.cfg.dtype,
            sharding=self.edf_sharding)
        self.kernel_down_proj_EFD = self.param_factory.create_kernel_param(
            rngs,
            shape=shape_down,
            dtype=self.cfg.dtype,
            sharding=self.efd_sharding)

    def _moe_fwd(self, x: Float, weights, op_mode):
        """Performs the basic forward pass of the MoE experts without dropping or megablocks.

        Args:
            x: Input array for the experts, shape (sequence_length, d_model) or (sequence_length, num_experts_per_tok, d_model) if pre-weighted.
            weights: Weights for combining expert outputs, shape (sequence_length, num_experts).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            Output array of shape (batch_size, sequence_length, d_model).
        """
        x = jnp.asarray(x, self.cfg.dtype)
        x_TD = nnx.with_sharding_constraint(x, self.activation_ffw_td[op_mode])
        act = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_ACT.value)
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TD,EDF -> TEF', x_TD,
                                    self.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[act](gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TD,EDF -> TEF', x_TD,
                                     self.kernel_up_proj_EDF.value)
        fuse_TEF = activated_gating_TEF * up_proj_TEF
        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                       self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_TD = jnp.einsum('TED,TE -> TD', down_proj_TED, weights)
        return output_TD.astype(self.cfg.dtype)

    def get_cfg(self) -> MoEConfig:
        """Returns the configuration object for the MoE layer."""
        return self.cfg

    def create_sharding(self):
        """Creates sharding configurations for activations and expert kernels."""
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
        self.edf_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_weights_edf))
        self.efd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_weights_efd))

        return
