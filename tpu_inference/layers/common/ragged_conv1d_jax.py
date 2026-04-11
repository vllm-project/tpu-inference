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
"""Ragged convolution Jax implementation."""

import jax.numpy as jnp


def ragged_conv1d(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    kernel_size,
):
    """Applies 1D convolution over ragged sequences and updates state.

    Args:
      x: Input tensor of shape `(num_tokens, dim)`.
      conv_state: Combined convolutional state of shape `(max_reqs, kernel_size -
        1, dim)`.
      conv_weight: Convolutional weight of shape `(dim, 1, kernel_size)`.
      conv_bias: Optional convolutional bias of shape `(dim,)`.
      query_start_loc: Tensor of shape `(num_seqs + 1,)` containing the start
        indices of each sequence, with the last element being the total number of
        valid tokens.
      state_indices: Tensor of shape `(max_reqs,)` mapping request index to state
        index.
      kernel_size: The size of the convolution kernel.
  
    Returns:
      A tuple containing:
      - output: The output tensor of shape `(num_tokens, dim)`.
      - updated_conv_state: The updated convolutional state of shape `(max_reqs,
        kernel_size - 1, dim)`.
    """
    num_tokens = x.shape[0]
    max_reqs = state_indices.shape[0]
    token_idx = jnp.arange(num_tokens)

    req_indices = (
        jnp.sum(token_idx[:, None] >= query_start_loc[None, :], axis=1) - 1)
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
    local_indices = token_idx - query_start_loc[req_indices]

    lengths = query_start_loc[1:] - query_start_loc[:-1]

    # 1. Compute Convolution
    out = jnp.zeros_like(x)
    w = conv_weight[:, 0, :].T  # (K, C)

    gathered_state = conv_state[state_indices]  # (max_reqs, K-1, C)

    for k in range(kernel_size):
        mask = local_indices >= k

        idx_x = jnp.clip(token_idx - k, 0, num_tokens - 1)
        idx_state_t = (kernel_size - 1) + (local_indices - k)

        x_tokens = x[idx_x]
        state_tokens = gathered_state[req_indices, idx_state_t]

        token_k = jnp.where(mask[:, None], x_tokens, state_tokens)
        out = out + token_k * w[kernel_size - 1 - k]

    if conv_bias is not None:
        out = out + conv_bias[jnp.newaxis, :]

    # 2. Update State
    padded_lengths = jnp.zeros(max_reqs, dtype=jnp.int32)
    padded_lengths = padded_lengths.at[:lengths.shape[0]].set(lengths)

    padded_q_loc = jnp.zeros(max_reqs + 1, dtype=jnp.int32)
    padded_q_loc = padded_q_loc.at[:query_start_loc.shape[0]].set(
        query_start_loc)
    padded_q_loc = padded_q_loc.at[query_start_loc.shape[0]:].set(
        query_start_loc[-1])

    r_grid = jnp.arange(max_reqs)[:, None]
    j_grid = jnp.arange(kernel_size - 1)[None, :]

    v_i = padded_lengths[:, None] + j_grid
    is_from_old_state = v_i < kernel_size - 1

    idx_state = jnp.where(is_from_old_state, v_i, 0)
    idx_x = padded_q_loc[r_grid + 1] + j_grid - (kernel_size - 1)
    idx_x = jnp.clip(idx_x, 0, num_tokens - 1)

    new_state_extracted = jnp.where(is_from_old_state[..., None],
                                    gathered_state[r_grid,
                                                   idx_state], x[idx_x])

    valid_seq_mask = jnp.arange(max_reqs) < lengths.shape[0]
    updated_conv_state = conv_state.at[state_indices].set(
        jnp.where(
            valid_seq_mask[:, None, None],
            new_state_extracted,
            conv_state[state_indices],
        ))

    return out, updated_conv_state
