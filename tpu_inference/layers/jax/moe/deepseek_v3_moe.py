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

import enum
from dataclasses import InitVar, dataclass
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.sharding import PartitionSpec
from jaxtyping import Float
from qwix._src.core.ragged_dot import ragged_dot as qwix_ragged_dot
from qwix._src.providers import ptq

from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.moe.moe import CombineExperts, MoE
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
    manually_quantize_qwix_activation, manually_quantize_qwix_weight)

modeling_flax_utils = FlaxUtils()


@dataclass
class DeepSeekV3Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    """

    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    n_groups: int
    topk_groups: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    dtype: jnp.dtype
    rngs: InitVar[nnx.Rngs]

    # Sharding Attributes
    activation_ffw_td: Sharding = ()
    ed_sharding: Sharding = ()
    e_sharding: Sharding = ()

    random_init: bool = False

    router_bias_dtype: jnp.dtype = jnp.float32

    use_moe_kernel: bool = False

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

        if self.use_moe_kernel:
            return scores_TE
        
        else:
            original_scores_TE = scores_TE
            topk_indices_TX = self.get_topk_indices(scores_TE)
            weights_TX = jnp.take_along_axis(original_scores_TE,
                                            topk_indices_TX,
                                            axis=-1)

            if self.norm_topk_prob:
                weights_TX /= jnp.sum(weights_TX, axis=-1)[..., None] + 1e-20

            weights_TX *= self.routed_scaling_factor

            return weights_TX, topk_indices_TX

    def __post_init__(self, rngs: nnx.Rngs):
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
                                   dtype=self.router_bias_dtype,
                                   sharding=self.e_sharding,
                                   random_init=self.random_init)


