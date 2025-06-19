from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Float

from tpu_commons.models.jax.common.base import (Config,
                                                ParamFactory)
from tpu_commons.models.jax.common.constants import RouterType
from tpu_commons.models.jax.common.sharding import (ShardingConfig,
                                                    NamedSharding)
from tpu_commons.models.jax.common.layers import (FFWConfig,
                                                  FlaxUtils)

modeling_flax_utils = FlaxUtils()


@dataclass
class RoutingConfig(Config):
    """Configuration for the Router module.

    Attributes:
        d_model: The dimension of the model.
        hidden_size: The hidden size of the expert.
        num_experts: The total number of experts.
        num_experts_per_tok: The number of experts each token is routed to.
        router_type: The type of router to use (e.g., 'top_k').
        act: The activation function to use.
        expert_capacity: The maximum number of tokens an expert can process. Defaults to -1 (no capacity limit).
        routed_bias: Whether to use a bias in the router. Defaults to False. # DeepSeek related. Could be removed
        routed_scaling_factor: Scaling factor for routed weights. Defaults to 1.0.
        dtype: The data type to use for computations. Defaults to jnp.float32.
    """
    d_model: int
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    router_type: RouterType
    act: str
    expert_capacity: int = -1
    routed_bias: bool = False
    routed_scaling_factor: float = 1.0
    dtype: Any = jnp.float32


@dataclass
class Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
        cfg: The RoutingConfig object.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """
    cfg: RoutingConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the Router module by creating sharding configurations and generating the router kernel."""
        self.create_sharding()
        self._generate_kernel()

    def __call__(self, x: Float, op_mode):
        """Routes tokens to experts.

        Args:
            x: Input array of shape (batch_size, sequence_length, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            A tuple containing:
                - normalized_weights_BTX: Normalized weights for selected experts, shape (batch_size, sequence_length, num_experts_per_tok).
                - selected_experts_BTX: Indices of selected experts, shape (batch_size, sequence_length, num_experts_per_tok).
        """
        x = jnp.asarray(x, self.cfg.dtype)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_btd[op_mode])
        router_logits_BTE = jnp.einsum('BTD,DE -> BTE', x,
                                       self.kernel_DE.value)
        activated_gating_BTF = nnx.softmax(router_logits_BTE.astype(
            jnp.float32),
                                           axis=-1)
        weights_BTX, selected_experts_BTX = jax.lax.top_k(
            activated_gating_BTF, self.cfg.num_experts_per_tok)
        normalized_weights_BTX = nnx.softmax(weights_BTX.astype(
            self.cfg.dtype),
                                             axis=-1)
        return normalized_weights_BTX, selected_experts_BTX

    def _generate_kernel(self):
        """Generates the router kernel (weights) for routing."""
        shape = (self.cfg.d_model, self.cfg.num_experts)
        self.kernel_DE = self.param_factory.create_kernel_init(
            shape=shape, dtype=self.cfg.dtype, sharding=self.ed_sharding)

    def create_sharding(self):
        """Creates sharding configurations for activations and kernel."""
        mode_dependent_attrs = [
            "activation_ffw_btd",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         prefill_sharding_config),
                'generate': NamedSharding(self.mesh,
                                          generate_sharding_config)
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.ed_sharding = NamedSharding(
            self.mesh,
            self.sharding_cfg.generate_sharding_cfg.moe_router_de)

        return


@dataclass(kw_only=True)
class MoEConfig(FFWConfig):
    """Configuration for the Mixture-of-Experts (MoE) layer.

    Attributes:
        d_model: The dimension of the model.
        expert_hidden_size: The hidden size of each expert's MLP.
        num_experts: The total number of experts.
        sequence_len: The maximum sequence length.
        act: The activation function to use within the experts.
        dtype: The data type to use for computations. Defaults to jnp.float32.
        apply_expert_weight_before_computation: Whether to apply expert weights before computation. Defaults to False.
    """
    d_model: int
    expert_hidden_size: int
    num_experts: int
    expert_act: str
    router_config: RoutingConfig
    apply_expert_weight_before_computation: bool = False


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
        self._generate_kernel()

    def __call__(self, x: Float, op_mode):
        """Performs the forward pass of the MoE layer.

        Args:
            x: Input array of shape (batch_size, sequence_length, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            Output array of shape (batch_size, sequence_length, d_model) after passing through MoE.
        """
        x = jnp.asarray(x, jnp.float32)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_btd[op_mode])
        weights_BTX, indices_BTX = self.router(x, op_mode)
        one_hot_indices_BTXE = jax.nn.one_hot(indices_BTX,
                                              num_classes=self.cfg.num_experts,
                                              dtype=self.cfg.dtype)
        full_weights_BTE = jnp.sum(one_hot_indices_BTXE *
                                   weights_BTX[..., None],
                                   axis=2)

        if self.cfg.apply_expert_weight_before_computation:
            with jax.named_scope("pre_computing_weight"):
                # need optimization for the out-product
                weighted_x_BTXD = jnp.einsum('BTD,BTX->BTXD', x, weights_BTX)
            return self._moe_fwd(weighted_x_BTXD,
                                 jnp.ones_like(full_weights_BTE), op_mode)
        else:
            return self._moe_fwd(x, full_weights_BTE, op_mode)

    def _generate_kernel(self):
        """Generates the kernels (weights) for the gating, up-projection, and down-projection layers of each expert."""

        shape_gating = (self.cfg.num_experts, self.cfg.d_model,
                        self.cfg.expert_hidden_size)
        shape_up = (self.cfg.num_experts, self.cfg.d_model,
                    self.cfg.expert_hidden_size)
        shape_down = (self.cfg.num_experts, self.cfg.expert_hidden_size,
                      self.cfg.d_model)

        self.kernel_gating_EDF = self.param_factory.create_kernel_init(
            shape=shape_gating,
            dtype=self.cfg.dtype,
            sharding=self.edf_sharding)
        self.kernel_up_proj_EDF = self.param_factory.create_kernel_init(
            shape=shape_up, dtype=self.cfg.dtype, sharding=self.edf_sharding)
        self.kernel_down_proj_EFD = self.param_factory.create_kernel_init(
            shape=shape_down, dtype=self.cfg.dtype, sharding=self.efd_sharding)

    def _moe_fwd(self, x: Float, weights, op_mode):
        """Performs the basic forward pass of the MoE experts without dropping or megablocks.

        Args:
            x: Input array for the experts, shape (batch_size, sequence_length, d_model) or (batch_size, sequence_length, num_experts_per_tok, d_model) if pre-weighted.
            weights: Weights for combining expert outputs, shape (batch_size, sequence_length, num_experts).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            Output array of shape (batch_size, sequence_length, d_model).
        """
        x = jnp.asarray(x, self.cfg.dtype)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_btd[op_mode])

        with jax.named_scope("gating"):
            gating_BTEF = jnp.einsum('BTD,EDF -> BTEF', x,
                                     self.kernel_gating_EDF.value)
            activated_gating_BTEF = modeling_flax_utils.ACT2FN[self.cfg.act](
                gating_BTEF)
        with jax.named_scope("up_projection"):
            up_proj_BTEF = jnp.einsum('BTD,EDF -> BTEF', x,
                                      self.kernel_up_proj_EDF.value)
        fuse_BTEF = activated_gating_BTEF * up_proj_BTEF
        with jax.named_scope("down_projection"):
            down_proj_BTED = jnp.einsum('BTEF,EFD -> BTED', fuse_BTEF,
                                        self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_BTD = jnp.einsum('BTED,BTE -> BTD', down_proj_BTED, weights)
        return output_BTD.astype(self.cfg.dtype)

    def get_cfg(self) -> MoEConfig:
        """Returns the configuration object for the MoE layer."""
        return self.cfg

    def create_sharding(self):
        """Creates sharding configurations for activations and expert kernels."""
        mode_dependent_attrs = [
            "activation_ffw_btd",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         prefill_sharding_config),
                'generate': NamedSharding(self.mesh,
                                          generate_sharding_config)
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        jax.debug.print("Sharding config = {sharding_config}", sharding_config=self.sharding_cfg)
        self.edf_sharding = NamedSharding(
            self.mesh,
            self.sharding_cfg.generate_sharding_cfg.moe_weights_edf)
        self.efd_sharding = NamedSharding(
            self.mesh,
            self.sharding_cfg.generate_sharding_cfg.moe_weights_efd)

        return
