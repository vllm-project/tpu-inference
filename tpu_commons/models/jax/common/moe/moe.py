from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from tpu_commons.models.jax.common.constants import *
from tpu_commons.models.jax.common.layers import *
from tpu_commons.models.jax.common.sharding import *


@dataclass
class RoutingConfig(Config):
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
    cfg: RoutingConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        self.create_sharding()
        self._generate_kernel()

    def __call__(self, x: Float[Array, 'B S D'], op_mode):
        x = jnp.asarray(x, self.cfg.dtype)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_bsd[op_mode])
        router_logits_BSE = jnp.einsum('BSD,DE -> BSE', x,
                                       self.kernel_DE.value)
        activated_gating_BSF = nnx.softmax(router_logits_BSE.astype(
            jnp.float32),
                                           axis=-1)
        weights_BSK, selected_experts_BSK = jax.lax.top_k(
            activated_gating_BSF, self.cfg.num_experts_per_tok)
        normalized_weights_BSK = nnx.softmax(weights_BSK.astype(
            self.cfg.dtype),
                                             axis=-1)
        return normalized_weights_BSK, selected_experts_BSK

    def _generate_kernel(self):
        shape = (self.cfg.d_model, self.cfg.num_experts)
        self.kernel_DE = self.param_factory.create_kernel_init(
            shape=shape, dtype=self.cfg.dtype, sharding=self.ed_sharding)

    def create_sharding(self):
        mode_dependent_attrs = [
            "activation_ffw_bsd",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.ed_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.moe_router_de))

        return


@dataclass
class MoEConfig(Config):
    d_model: int
    expert_hidden_size: int
    num_experts: int
    sequence_len: int
    act: str
    dtype: Any = jnp.float32
    apply_expert_weight_before_computation: bool = False


@dataclass
class MoE(nnx.Module):
    """Moe Routed MLP Layer"""
    cfg: MoEConfig
    mesh: Mesh
    param_factory: ParamFactory
    router: Router
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        self.create_sharding()
        self._generate_kernel()

    def __call__(self, x: Float[Array, 'B S D'], op_mode):
        x = jnp.asarray(x, jnp.float32)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_bsd[op_mode])
        weights_BSK, indices_BSK = self.router(x, op_mode)
        one_hot_indices_BSKE = jax.nn.one_hot(indices_BSK,
                                              num_classes=self.cfg.num_experts,
                                              dtype=self.cfg.dtype)
        full_weights_BSE = jnp.sum(one_hot_indices_BSKE *
                                   weights_BSK[..., None],
                                   axis=2)

        if self.cfg.apply_expert_weight_before_computation:
            with jax.named_scope("pre_computing_weight"):
                # need optimization for the out-product
                weighted_x_BSKD = jnp.einsum('BSD,BSK->BSKD', x, weights_BSK)
            return self._moe_fwd(weighted_x_BSKD,
                                 jnp.ones_like(full_weights_BSE), op_mode)
        else:
            return self._moe_fwd(x, full_weights_BSE, op_mode)

    def _generate_kernel(self):

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

    def _moe_fwd(self, x: Float[Array, 'B S D'], weights, op_mode):
        """
        basic moe forward without dropping, megablx etc
        """
        x = jnp.asarray(x, self.cfg.dtype)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_bsd[op_mode])

        with jax.named_scope("gating"):
            gating_BSEF = jnp.einsum('BSD,EDF -> BSEF', x,
                                     self.kernel_gating_EDF.value)
            activated_gating_BSEF = modeling_flax_utils.ACT2FN[self.cfg.act](
                gating_BSEF)
        with jax.named_scope("up_projection"):
            up_proj_BSEF = jnp.einsum('BSD,EDF -> BSEF', x,
                                      self.kernel_up_proj_EDF.value)
        fuse_BSEF = activated_gating_BSEF * up_proj_BSEF
        with jax.named_scope("down_projection"):
            down_proj_BSED = jnp.einsum('BSEF,EFD -> BSED', fuse_BSEF,
                                        self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_BSD = jnp.einsum('BSED,BSE -> BSD', down_proj_BSED, weights)
        return output_BSD.astype(self.cfg.dtype)

    def get_cfg(self) -> MoEConfig:
        return self.cfg

    def create_sharding(self):
        mode_dependent_attrs = [
            "activation_ffw_bsd",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.edf_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.moe_weights_edf))
        self.efd_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.moe_weights_efd))

        return
