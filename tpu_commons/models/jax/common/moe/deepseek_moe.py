from dataclasses import field, make_dataclass, dataclass
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
        (HuggingFaceArgNames.NORM_TOPK_PROB.value, float),
        ("dtype", DTypeLike),
        ("vllm_config", VllmConfig, field(repr=False, default=None)),
    ],
    bases=(Config,),
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

    cfg: DeepSeekV3RoutingConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None


    def __post_init__(self):
        """Initializes the Router module by creating sharding configurations and generating the router kernel."""
        
        self.hidden_size = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        self.num_experts = getattr(
            self.cfg, HuggingFaceArgNames.NUM_ROUTED_EXPERTS.value
        )
        self.num_experts_per_tok = getattr(
            self.cfg, HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN.value
        )
        self.n_groups = getattr(self.cfg, HuggingFaceArgNames.NUM_GROUPS.value)
        self.topk_groups = getattr(self.cfg, HuggingFaceArgNames.TOPK_GROUP.value)
        self.norm_topk_prob = getattr(
            self.cfg, HuggingFaceArgNames.NORM_TOPK_PROB.value
        )
        self.routed_scaling_factor = getattr(
            self.cfg, HuggingFaceArgNames.ROUTED_SCALING_FACTOR.value
        )
        self.dtype = self.cfg.dtype

        self.create_sharding()

    def get_topk_indices(self, scores: Float) -> Float:
        """Get the topk indices of the scores.

        Args:
            scores: The scores to get the topk indices of. Shape (sequence, num_experts).

        Returns:
            The topk indices of the scores. Shape (sequence, num_experts_per_tok).
        """

        scores = scores + self.bias_E
        if self.n_groups > 1:
            experts_per_group = self.num_experts // self.n_groups
            group_scores = jnp.reshape(scores, (-1, self.n_groups, experts_per_group))
            group_scores = jax.lax.top_k(group_scores, k=2)[0]
            group_scores = jnp.sum(group_scores, axis=-1)
            indices = jax.lax.top_k(group_scores, k=self.topk_groups)[1]

            mask = jnp.any(
                jnp.arange(self.n_groups)[:, None] == indices[..., None, :], axis=-1
            )
            mask = jnp.repeat(mask, scores.shape[-1] // mask.shape[-1], -1)
            scores = jnp.where(mask, scores, 0.0)

        indices = jax.lax.top_k(scores, k=self.num_experts_per_tok)[1]

        return indices

    def __call__(self, x: Float, op_mode: str) -> Tuple[Float, Float]:
        """Routes tokens to top k experts.

        Args:
            x: Input array of shape (sequence, d_model).
            op_mode: The operation mode ('prefill' or 'generate') to determine sharding.

        Returns:
            A tuple containing:
                - weights: Normalized weights for selected experts, shape (sequence, num_experts_per_tok).
                - indices: Indices of selected experts, shape (sequence, num_experts_per_tok).
        """
        x = jnp.asarray(x, self.dtype)
        x_td = nnx.with_sharding_constraint(x, self.activation_ffw_td[op_mode])

        scores_te = jnp.einsum("TD,DE -> TE", x_td, self.kernel_DE.value)
        scores_te = nnx.sigmoid(scores_te)

        original_scores = scores_te
        topk_indices_tk = self.get_topk_indices(scores_te)
        weights_tk = jnp.take_along_axis(original_scores, topk_indices_tk, axis=-1)

        if self.norm_topk_prob:
            weights_tk /= jnp.sum(weights_tk, axis=-1)[..., None] + 1e-20

        weights_tk *= self.routed_scaling_factor

        return weights_tk, topk_indices_tk

    def generate_kernel(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights and bias) for routing."""
        D = self.hidden_size
        E = self.num_experts
        self.kernel_DE = self.param_factory.create_kernel_param(
            rngs, shape=(D, E), dtype=self.cfg.dtype, sharding=self.ed_sharding
        )
        self.bias_E = self.param_factory.create_scale_param(
            rngs,
            shape=(E,),
            dtype=self.cfg.dtype,
            sharding=None,  # TODO: check if this is correct
        )

    def create_sharding(self):
        """Creates sharding configurations for activations and kernel."""
        mode_dependent_attrs = [
            "activation_ffw_td",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_rules, attr_name
            )
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name
            )

            sharding_dict = {
                "prefill": NamedSharding(self.mesh, P(*prefill_sharding_config)),
                "generate": NamedSharding(self.mesh, P(*generate_sharding_config)),
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.ed_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.moe_router_de)
        )

        return
