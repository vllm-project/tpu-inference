from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jaxtyping import Float

from tpu_commons.models.jax.common.base import create_param


@dataclass
class DeepSeekV3Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
        mesh: The JAX device mesh for distributed computation.
        quant: Optional configuration for quantization.
    """

    mesh: Mesh
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    n_groups: int
    topk_groups: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    dtype: jnp.dtype

    # Sharding Attributes
    activation_ffw_td: NamedSharding
    ed_sharding: NamedSharding
    e_sharding: NamedSharding

    random_init: bool = False
    quant: Any | None = None

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

    def __call__(self, x_TD: Float) -> Tuple[Float, Float]:
        """Routes tokens to top k experts.

        Args:
            x_TD: Input array of shape (sequence, d_model).

        Returns:
            A tuple containing:
                - weights: Normalized weights for selected experts, shape (sequence, num_experts_per_tok).
                - indices: Indices of selected experts, shape (sequence, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

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
        self.kernel_DE = create_param(rngs,
                                      shape=(D, E),
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)
        self.bias_E = create_param(rngs,
                                   shape=(E, ),
                                   dtype=self.dtype,
                                   sharding=self.e_sharding,
                                   random_init=self.random_init)
