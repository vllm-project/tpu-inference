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
"""Ragged convolution Jax implementation.

This file provides highly optimized 1D convolutions over ragged sequences on
TPUs in JAX.

Key design ideas:
1. Convolution via standard operators: Instead of mapping large index loops,
   it applies standard `lax.conv_general_dilated` on flat tokens to perform a
   depthwise 1-D convolution.
2. Ragged boundary fixup: To prevent cross-sequence pollution at padded
   boundaries in consecutive batch groups, we compute slice segments right where
   sequences transition and resolve border overlaps accurately.
3. State update: We update the convolutional state by selecting either the
   previous state or the current input based on whether the current position is
   within the state's range or not.
"""

import jax
import jax.numpy as jnp
from jax import lax


def _fix_query_start_loc(query_start_loc, num_valid_seqs):
    """Fixes query_start_loc to be non-decreasing for invalid sequences."""
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    return jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)


def _get_boundary_indices(starts, lengths, kernel_size, num_valid_seqs):
    """Computes indices for boundary fixup."""
    valid_mask = jnp.arange(starts.shape[0]) < num_valid_seqs
    starts = jnp.where(valid_mask, starts, 1)[:, None]
    lengths = lengths[:, None]
    k_range = jnp.arange(kernel_size - 1)[None, :]
    gather_indices = starts + jnp.minimum(k_range, lengths - 1)
    scatter_indices = jnp.where(
        (k_range < lengths) & valid_mask[:, None],
        starts + k_range,
        -1,
    )
    return gather_indices, scatter_indices


def _get_state_update_indices(query_start_loc, kernel_size, num_tokens):
    """Computes indices for updating the convolutional state."""
    lengths = query_start_loc[1:] - query_start_loc[:-1]

    k_range = jnp.arange(kernel_size - 1)

    safe_idx_x = (query_start_loc[1:, None] -
                  jnp.arange(kernel_size - 1, 0, -1)[None, :])
    safe_idx_x = jnp.clip(safe_idx_x, 0, num_tokens - 1)

    is_from_old_state = k_range[None, :] < (kernel_size - 1 - lengths)[:, None]

    idx_g = k_range[None, :] + lengths[:, None]
    idx_g = jnp.clip(idx_g, 0, kernel_size - 2)

    return safe_idx_x, is_from_old_state, idx_g


def _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias):
    """Depthwise 1D convolution using loops over kernel size.

  Note that an alternative is to use `lax.conv_general_dilated`. However we use
  loops to enable fusing bias addition.
  """
    num_tokens = x.shape[0]
    kernel_size = conv_weight.shape[-1]
    out = None

    # Pad x on the left with kernel_size - 1 zeros
    padded_x = jnp.pad(x, ((kernel_size - 1, 0), (0, 0)))

    # Accumulate over kernel size
    for k in range(kernel_size):
        # Accumulation needs to be done in float32 to avoid accuracy loss
        x_slice = padded_x[k:k + num_tokens, :].astype(jnp.float32)
        weight_slice = conv_weight[:, 0, k].astype(jnp.float32)
        if out is None:
            if conv_bias is None:
                out = x_slice * weight_slice
            else:
                out = x_slice * weight_slice + conv_bias[jnp.newaxis, :]
        else:
            out += x_slice * weight_slice

    assert out is not None
    return out.astype(x.dtype)


def ragged_conv1d_mixed_prefill(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
    """Applies 1D convolution, optimized for prefill."""
    num_tokens = x.shape[0]
    max_blocks = state_indices.shape[0]
    num_valid_seqs = distribution[2]

    # 1. Compute Convolution
    out = _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias)

    # 2. Fixup Boundary Tokens
    query_start_loc = _fix_query_start_loc(query_start_loc, num_valid_seqs)
    starts = query_start_loc[:-1]
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    gather_indices, scatter_indices = _get_boundary_indices(
        starts, lengths, kernel_size, num_valid_seqs)
    x_first = x[gather_indices]  # (max_blocks, K-1, dim)

    # Concatenate state and x_first along the spatial dimension
    gathered_state = conv_state[state_indices]  # (max_blocks, K-1, dim)

    # Mask the gathered conv state to zero for sequences without initial
    # state, so brand-new prefills see zeros instead of whatever a reused
    # slot still held.
    gathered_state = jnp.where(
        has_initial_state[:, None, None],
        gathered_state,
        jnp.zeros_like(gathered_state),
    )

    combined_tokens = jnp.concatenate([gathered_state, x_first],
                                      axis=1)  # (max_blocks, 2K - 2, dim)

    # Depthwise Convolution for Fixup
    b_out = lax.conv_general_dilated(
        combined_tokens,
        conv_weight,
        window_strides=(1, ),
        padding="VALID",
        dimension_numbers=("NWC", "OIW", "NWC"),
        feature_group_count=x.shape[-1],
        precision=lax.Precision.HIGHEST,
    ).reshape(-1, x.shape[-1])
    if conv_bias is not None:
        b_out += conv_bias[jnp.newaxis, :]

    # Scatter the updates. Note that scatter indices may contain -1, which will be
    # ignored by the scatter operation with mode="drop".
    out = out.at[scatter_indices.flatten()].set(b_out.astype(out.dtype),
                                                mode="drop",
                                                wrap_negative_indices=False)
    # Mask invalid tokens to 0
    total_valid_tokens = query_start_loc[num_valid_seqs]
    valid_token_mask = jnp.arange(num_tokens) < total_valid_tokens
    out = jnp.where(valid_token_mask[:, jnp.newaxis], out, 0.0)
    # 3. Update State
    true_valid_seq_mask = jnp.arange(max_blocks) < num_valid_seqs
    safe_idx_x, is_from_old_state, idx_g = _get_state_update_indices(
        query_start_loc, kernel_size, num_tokens)

    x_tokens = x[safe_idx_x]
    r_grid = jnp.arange(max_blocks)[:, None]
    state_tokens = gathered_state[r_grid, idx_g]

    new_state_extracted = jnp.where(is_from_old_state[..., None], state_tokens,
                                    x_tokens)

    updated_conv_state = conv_state.at[state_indices].set(
        jnp.where(
            true_valid_seq_mask[:, None, None],
            new_state_extracted,
            conv_state[state_indices],
        ))

    return out.astype(x.dtype), updated_conv_state


def ragged_conv1d_decode_only(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
    """Apply conv1d for decode-only case (All valid reqs have seq_len=1)."""
    num_tokens = x.shape[0]

    token_idx = jnp.arange(num_tokens)
    req_state_indices = state_indices[token_idx]
    gathered_state = conv_state[
        req_state_indices]  # (num_tokens, kernel_size - 1, dim)

    # Concat old state and new token to form (num_tokens, kernel_size, dim)
    lhs = jnp.concatenate([gathered_state, x[:, jnp.newaxis, :]], axis=1)

    out = jnp.einsum(
        "nkd,dk->nd",
        lhs,
        conv_weight[:, 0, :],
        precision=lax.Precision.HIGHEST,
    )

    if conv_bias is not None:
        out = out + conv_bias

    num_valid_seqs = distribution[2]

    # Drop oldest state and append new state
    new_state_extracted = jnp.concatenate(
        [gathered_state[:, 1:, :], x[:, jnp.newaxis, :]], axis=1)

    token_idx = jnp.arange(num_tokens)
    valid_mask = token_idx < num_valid_seqs
    states_to_set = jnp.where(
        valid_mask[:, jnp.newaxis, jnp.newaxis],
        new_state_extracted,
        gathered_state,
    )

    updated_conv_state = conv_state.at[req_state_indices].set(states_to_set)

    out = jnp.where(valid_mask[:, jnp.newaxis], out, 0.0)

    return out.astype(x.dtype), updated_conv_state


# Donate conv_state to avoid "copy" op by XLA
@jax.jit(donate_argnames=("conv_state", ), static_argnames=("kernel_size", ))
@jax.named_scope("ragged_conv1d_jax")
def ragged_conv1d(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Applies 1D convolution over ragged sequences and updates state.

    Args:
      x: Input tensor of shape `(num_tokens, dim)`.
      conv_state: Combined convolutional state of shape `(max_blocks, kernel_size
        - 1, dim)`.
      conv_weight: Convolutional weight of shape `(dim, 1, kernel_size)`.
      conv_bias: Optional convolutional bias of shape `(dim,)`.
      query_start_loc: Tensor of shape `(num_seqs + 1,)` containing the start
        indices of each sequence, with the last element being the total number of
        valid tokens.
      state_indices: Tensor of shape `(max_blocks,)` mapping request index to
        state index.
      distribution: Distribution tensor containing number of valid sequences at
        index 2.
      has_initial_state: Boolean tensor of shape `(max_reqs,)`. ``True`` when the
        request has prior conv state to use (chunked-prefill continuation or
        prefix-cache hit). ``False`` for brand-new prefills, in which case the
        gathered conv state is treated as zeros — matching GPU's
        ``causal_conv1d_fn(has_initial_state=...)`` semantics. Without this
        masking the conv would consume whatever a reused mamba slot still held
        from a prior request, silently corrupting the first ``kernel_size - 1``
        outputs of every new request.
      kernel_size: The size of the convolution kernel.

    Returns:
      A tuple containing:
      - output: The output tensor of shape `(num_tokens, dim)`.
      - updated_conv_state: The updated convolutional state of shape `(max_blocks,
        kernel_size - 1, dim)`.
    """
    is_decode_only = distribution[0] == distribution[2]

    def decode_only_branch(_):
        return ragged_conv1d_decode_only(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )

    def mixed_prefill_branch(_):
        return ragged_conv1d_mixed_prefill(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )

    return jax.lax.cond(is_decode_only,
                        decode_only_branch,
                        mixed_prefill_branch,
                        operand=None)
