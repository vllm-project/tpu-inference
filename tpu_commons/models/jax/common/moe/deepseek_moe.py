from dataclasses import dataclass, field, make_dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import DTypeLike, Float
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.sharding import ShardingConfig

DeepSeekV3RoutingConfig = make_dataclass(
    "RoutingConfig",
    [
        (HuggingFaceArgNames.HIDDEN_SIZE.value, int),
        (HuggingFaceArgNames.NUM_ROUTED_EXPERTS.value, int),
        (HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value, int),
        (HuggingFaceArgNames.NUM_GROUPS.value, int),
        (HuggingFaceArgNames.ROUTED_SCALING_FACTOR.value, float),
        (HuggingFaceArgNames.TOPK_GROUP.value, int),
        (HuggingFaceArgNames.NORM_TOPK_PROB.value, bool),
        ("dtype", DTypeLike),
        ("vllm_config", VllmConfig, field(repr=False, default=None)),
    ],
    bases=(Config, ),
)

DeepSeekV3RoutingConfig.__doc__ = f"""Configuration for the Router module.

     Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.NUM_ROUTED_EXPERTS.value}: The number of routed experts.
        {HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value}: The number of experts each token is routed to.
        {HuggingFaceArgNames.NUM_GROUPS.value}: The number of groups.
        {HuggingFaceArgNames.ROUTED_SCALING_FACTOR.value}: The scaling factor for routed weights.
        {HuggingFaceArgNames.TOPK_GROUP.value}: The number of top k groups to consider.
        {HuggingFaceArgNames.NORM_TOPK_PROB.value}: Whether to normalize the top k groups."""


@dataclass
class DeepSeekV3Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
        cfg: The RoutingConfig object.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """

    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    n_groups: int
    topk_groups: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    dtype: jnp.dtype
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the Router module by creating sharding configurations and generating the router kernel."""

        self.create_sharding()

    def get_topk_indices(self, scores_TE: Float) -> Float:
        """Get the topk indices of the scores.

        Args:
            scores_TE: The scores to get the topk indices of. Shape (sequence, num_experts).

        Returns:
            The topk indices of the scores. Shape (sequence, num_experts_per_tok).
        """

        scores_TE = scores_TE + self.bias_E
        if self.n_groups > 1:
            experts_per_group = self.num_experts // self.n_groups
            group_scores_TGM = jnp.reshape(
                scores_TE, (-1, self.n_groups, experts_per_group))
            group_scores_TG2 = jax.lax.top_k(group_scores_TGM, k=2)[0]
            group_scores_TG = jnp.sum(group_scores_TG2, axis=-1)
            indices = jax.lax.top_k(group_scores_TG, k=self.topk_groups)[1]

            mask_TG = jnp.any(jnp.arange(
                self.n_groups)[:, None] == indices[..., None, :],
                              axis=-1)
            mask_TE = jnp.repeat(mask_TG,
                                 scores_TE.shape[-1] // mask_TG.shape[-1], -1)
            scores_TE = jnp.where(mask_TE, scores_TE, 0.0)

        indices_TX = jax.lax.top_k(scores_TE, k=self.num_experts_per_tok)[1]

        return indices_TX

    def __call__(self, x_TD: Float, op_mode: str) -> Tuple[Float, Float]:
        """Routes tokens to top k experts.

        Args:
            x_TD: Input array of shape (sequence, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            A tuple containing:
                - weights: Normalized weights for selected experts, shape (sequence, num_experts_per_tok).
                - indices: Indices of selected experts, shape (sequence, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD,
                                            self.activation_ffw_td[op_mode])

        scores_TE = jnp.einsum("TD,DE -> TE", x_TD, self.kernel_DE.value)
        scores_TE = nnx.sigmoid(scores_TE)

        original_scores_TE = scores_TE
        topk_indices_TX = self.get_topk_indices(scores_TE)
        weights_TX = jnp.take_along_axis(original_scores_TE,
                                         topk_indices_TX,
                                         axis=-1)

        if self.norm_topk_prob:
            weights_TX /= jnp.sum(weights_TX, axis=-1)[..., None] + 1e-20

        weights_TX *= self.routed_scaling_factor

        return weights_TX, topk_indices_TX

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights and bias) for routing."""
        D = self.hidden_size
        E = self.num_experts
        self.kernel_DE = self.param_factory.create_kernel_param(
            rngs, shape=(D, E), dtype=self.dtype, sharding=self.ed_sharding)
        self.bias_E = self.param_factory.create_scale_param(
            rngs,
            shape=(E, ),
            dtype=self.dtype,
            sharding=self.e_sharding,
        )

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
                "prefill": NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                "generate": NamedSharding(self.mesh,
                                          P(*generate_sharding_config)),
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.ed_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_router_de))

        self.e_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_router_bias_e))

        return
