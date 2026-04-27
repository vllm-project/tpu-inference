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

import jax
import jax.numpy as jnp


# Donate conv_state to avoid "copy" op by XLA
@jax.jit(donate_argnames=("conv_state", ), static_argnames=("kernel_size", ))
@jax.named_scope("ragged_conv1d_jax")
def ragged_conv1d(
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
    """Applies 1D convolution over ragged sequences and updates state.

    Args:
      x: Input tensor of shape `(num_tokens, dim)`.
      conv_state: Combined convolutional state of shape `(num_blocks, kernel_size
        - 1, dim)`. `num_blocks` is always equal or larger than `max_seqs + 1`.
        The first block is a null_block and only used for padded / invalid tokens.
      conv_weight: Convolutional weight of shape `(dim, 1, kernel_size)`.
      conv_bias: Optional convolutional bias of shape `(dim,)`.
      query_start_loc: Tensor of shape `(num_seqs + 1,)` containing the start
        indices of each sequence, with the last element being the total number of
        valid tokens.
      state_indices: Tensor of shape `(max_reqs,)` mapping request index to state
        index.
      kernel_size: The size of the convolution kernel.
      distribution: Distribution tensor containing number of valid sequences at
        index 2.
      has_initial_state: Boolean tensor of shape `(max_reqs,)`. ``True`` when
        the request has prior conv state to use (chunked-prefill continuation
        or prefix-cache hit). ``False`` for brand-new prefills, in which case
        the gathered conv state is treated as zeros — matching GPU's
        ``causal_conv1d_fn(has_initial_state=...)`` semantics. Without this
        masking the conv would consume whatever a reused mamba slot still
        held from a prior request, silently corrupting the first
        ``kernel_size - 1`` outputs of every new request.

    Returns:
      A tuple containing:
      - output: The output tensor of shape `(num_tokens, dim)`.
      - updated_conv_state: The updated convolutional state of shape `(num_blocks,
        kernel_size - 1, dim)`.
    """
    num_tokens = x.shape[0]
    max_reqs = state_indices.shape[0]
    token_idx = jnp.arange(num_tokens)

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    # Handle case where query_start_loc has trailing zeros by filling them with the last valid location.
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc,
                                          last_valid_loc)

    req_indices = (jnp.sum(
        token_idx[:, None] >= effective_query_start_loc[None, :], axis=1) - 1)
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
    local_indices = token_idx - effective_query_start_loc[req_indices]

    lengths = effective_query_start_loc[1:] - effective_query_start_loc[:-1]

    # 1. Compute Convolution.
    # Accumulate in fp32 to match GPU's `causal_conv1d_fn`
    # (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`), which
    # loads x and weights as fp32 before the kernel-size-element sum.
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    w = conv_weight[:, 0, :].T.astype(jnp.float32)  # (K, C)

    gathered_state = conv_state[state_indices]  # (max_reqs, K-1, C)
    # Mask the gathered conv state to zero for sequences without initial
    # state, so brand-new prefills see zeros instead of whatever a reused
    # slot still held. Mirrors GPU's `has_initial_state` plumbing.
    gathered_state = jnp.where(
        has_initial_state[:, None, None],
        gathered_state,
        jnp.zeros_like(gathered_state),
    )
    gathered_state_f32 = gathered_state.astype(jnp.float32)

    out = jnp.zeros((num_tokens, x.shape[-1]), dtype=jnp.float32)
    for k in range(kernel_size):
        mask = local_indices >= k

        idx_x = jnp.clip(token_idx - k, 0, num_tokens - 1)
        idx_state_t = (kernel_size - 1) + (local_indices - k)

        x_tokens = x_f32[idx_x]
        state_tokens = gathered_state_f32[req_indices, idx_state_t]

        token_k = jnp.where(mask[:, None], x_tokens, state_tokens)
        out = out + token_k * w[kernel_size - 1 - k]

    if conv_bias is not None:
        out = out + conv_bias.astype(jnp.float32)[jnp.newaxis, :]
    out = out.astype(orig_dtype)

    # 2. Update State
    padded_lengths = jnp.zeros(max_reqs, dtype=jnp.int32)
    padded_lengths = padded_lengths.at[:lengths.shape[0]].set(lengths)

    padded_q_loc = jnp.zeros(max_reqs + 1, dtype=jnp.int32)
    padded_q_loc = padded_q_loc.at[:effective_query_start_loc.shape[0]].set(
        effective_query_start_loc)
    padded_q_loc = padded_q_loc.at[effective_query_start_loc.shape[0]:].set(
        effective_query_start_loc[-1])

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

    true_valid_seq_mask = jnp.arange(max_reqs) < num_valid_seqs
    updated_conv_state = conv_state.at[state_indices].set(
        jnp.where(
            true_valid_seq_mask[:, None, None],
            new_state_extracted,
            conv_state[state_indices],
        ))

    return out, updated_conv_state
