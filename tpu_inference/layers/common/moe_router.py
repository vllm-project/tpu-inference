# Copyright 2026 Google LLC
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
"""Expert selection shared by the MoE backends: score the router output and pick
each token's top-k experts."""

import functools

import jax
from jax import numpy as jnp

import tpu_inference.envs as envs
from tpu_inference.kernels.router_topk import router_topk


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case "sqrtsoftplus":
            return jnp.sqrt(jax.nn.softplus(x))
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {scoring_fn} scoring function")


def select_experts(
    gating_output: jax.Array,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    hash_based_topk_indices: jax.Array | None = None,
    expert_score_correction_bias: jax.Array | None = None,
    num_valid_tokens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Pick each token's experts from the router output.

    Args:
        gating_output: router logits [num_tokens, num_experts].
        topk: number of experts per token.
        renormalize: make each token's top-k weights sum to 1.
        scoring_fn: scoring function applied to gating_output.
        hash_based_topk_indices: if set, the experts to use [num_tokens, topk].
        expert_score_correction_bias: added to the scores to pick the experts
            by exact top-k, but not to their weights [num_experts].
        num_valid_tokens: if set, the tokens from this index on are padding.

    Returns:
        (topk_weights, topk_indices), both [num_tokens, topk].
    """
    num_tokens = gating_output.shape[0]
    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    if hash_based_topk_indices is not None:
        topk_indices = hash_based_topk_indices
        topk_weights = jnp.take_along_axis(topk_weights, topk_indices, axis=-1)
    elif envs.MOE_APPROX_TOPK:
        topk_weights, topk_indices = jax.lax.approx_max_k(
            topk_weights,
            k=topk,
            recall_target=envs.MOE_APPROX_TOPK_RECALL_TARGET)
    else:
        # TPU_MOE_ROUTER_TOPK=rowmax: k row-max passes instead of a sort.
        select_topk = (functools.partial(router_topk, k=topk)
                       if envs.TPU_MOE_ROUTER_TOPK == "rowmax" else
                       functools.partial(jax.lax.top_k, k=topk))
        if expert_score_correction_bias is not None:
            _, topk_indices = select_topk(
                topk_weights + expert_score_correction_bias[None, :])
            topk_weights = jnp.take_along_axis(topk_weights,
                                               topk_indices,
                                               axis=-1)
        else:
            topk_weights, topk_indices = select_topk(topk_weights)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    # Route padding tokens to expert 0 instead of picking a selected expert. This
    # is especially useful when we have a low number of tokens (e.g. low
    # concurrency), where padding tokens may activate unnecessary expert weights
    # and slow down the gmm kernel.
    if num_valid_tokens is not None:
        token_valid = (jnp.arange(num_tokens) < num_valid_tokens)[:, None]
        topk_indices = jnp.where(token_valid, topk_indices, 0)
        topk_weights = jnp.where(token_valid, topk_weights, 0.0)
    return topk_weights, topk_indices
