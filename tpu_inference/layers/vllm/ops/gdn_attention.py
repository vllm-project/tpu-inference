"""
This file implements a pure JAX implementation for Gated Delta Networks (GDNs)

Key Implementation Considerations:
- Dynamic Shapes & Continuous Batching: vLLM mixes ragged prefill and decode tokens
  dynamically. To prevent XLA from recompiling on every varying batch size, we use
  `jax.lax.scan` to iterate over flat token streams using conditional masking, allow us
  to have a unified and statically-shaped execution path.
"""

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
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """L2 normalize along last dimension.

    Args:
        x: input to normalize
        eps: epsilon for numerical stability
  
    Returns:
        normalized x
    """
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def _chunk_gated_delta_rule(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Chunked gated delta rule matching HF's torch_chunk_gated_delta_rule.

    By chunking here, we can effectively transform the purely sequential
    RNN recurrence into a block-parallel operation. It processes tokens in chunks
    and then only passes the recurrent state sequentially between chunks.
  
    One detail worth pointing out is that the continuous decay mask (`g`) is
    cumulative, so
    Applying the triangular mask *before* exponentiation is key here to prevent
    NaNs
    when dealing with large sequence lengths.
  
    Args:
        query: (B, H, T, d_k) — already L2-normed
        key: (B, H, T, d_k) — already L2-normed
        value: (B, H, T, d_v)
        g: (B, H, T) — continuous decay (negative values)
        beta: (B, H, T) — input gate (after sigmoid)
        chunk_size: chunk processing size
        initial_state: (B, H, d_k, d_v) or None
  
    Returns:
        output: (B, H, T, d_v)
        final_state: (B, H, d_k, d_v) or None
    """
    B, H, T, d_k = query.shape
    d_v = value.shape[-1]

    # Pad to multiple of chunk_size
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

    total_len = T + pad_size
    scale = d_k**-0.5
    query = query * scale

    v_beta = value * beta[..., jnp.newaxis]
    k_beta = key * beta[..., jnp.newaxis]

    # Reshape: (B, H, n_chunks, chunk_size, dim)
    nc = total_len // chunk_size
    query = query.reshape(B, H, nc, chunk_size, d_k)
    key = key.reshape(B, H, nc, chunk_size, d_k)
    value = value.reshape(B, H, nc, chunk_size, d_v)
    k_beta = k_beta.reshape(B, H, nc, chunk_size, d_k)
    v_beta = v_beta.reshape(B, H, nc, chunk_size, d_v)
    g = g.reshape(B, H, nc, chunk_size)

    # Cumulative g within each chunk
    g = jnp.cumsum(g, axis=-1)

    # Decay mask: tril BEFORE exp for numerical stability (matching HF)
    g_diff = g[..., :, jnp.newaxis] - g[..., jnp.newaxis, :]
    decay_mask = jnp.exp(jnp.tril(g_diff)) * jnp.tril(
        jnp.ones((chunk_size, chunk_size)))

    # Intra-chunk correction matrix
    mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn = -(k_beta @ jnp.swapaxes(key, -2, -1)) * decay_mask
    attn = jnp.where(mask, 0.0, attn)

    # Iterative correction (matching HF for loop exactly)
    for i in range(1, chunk_size):
        row = attn[..., i, :i]  # (B, H, nc, i)
        sub = attn[..., :i, :i]  # (B, H, nc, i, i)
        correction = jnp.sum(row[..., jnp.newaxis] * sub,
                             axis=-2)  # (B, H, nc, i)
        attn = attn.at[..., i, :i].set(row + correction)

    attn = attn + jnp.eye(chunk_size)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * jnp.exp(g)[..., jnp.newaxis])

    # Initialize state
    state = (jnp.zeros((B, H, d_k, d_v), dtype=jnp.float32)
             if initial_state is None else initial_state.astype(jnp.float32))

    upper2 = jnp.triu(jnp.ones((chunk_size, chunk_size)), k=1)

    all_outputs = []
    for ci in range(nc):
        q_i = query[:, :, ci]
        k_i = key[:, :, ci]
        v_i = value[:, :, ci]
        g_i = g[:, :, ci]

        intra = (q_i @ jnp.swapaxes(k_i, -2, -1)) * decay_mask[:, :, ci]
        intra = jnp.where(upper2, 0.0, intra)

        v_prime = k_cumdecay[:, :, ci] @ state
        v_new = v_i - v_prime

        attn_inter = (q_i * jnp.exp(g_i)[..., jnp.newaxis]) @ state
        chunk_out = attn_inter + intra @ v_new
        all_outputs.append(chunk_out)

        g_last = g_i[:, :, -1:]
        state = (state * jnp.exp(g_last)[..., jnp.newaxis] + jnp.swapaxes(
            k_i * jnp.exp(g_last[..., jnp.newaxis] - g_i[..., jnp.newaxis]),
            -2,
            -1,
        ) @ v_new)

    output = jnp.concatenate(all_outputs, axis=2)[:, :, :T]
    return output, state if output_final_state else None


def _recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Single-step for decode.

    Args:
        query: (B, H, T, d_k)
        key: (B, H, T, d_k)
        value: (B, H, T, d_v)
        g: (B, H, T)
        beta: (B, H, T)
        state: (B, H, d_k, d_v)
  
    Returns:
        output: (B, H, T, d_v)
        new_state: (B, H, d_k, d_v)
    """
    # Just use chunk_gated_delta_rule with T=1
    output, new_state = _chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=1,
        initial_state=state,
        output_final_state=True,
    )
    return output, new_state


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


def ragged_gated_delta_rule(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    n_kq,
    n_v,
    d_k,
    d_v,
):
    """Applies the gated delta rule over ragged sequences and updates recurrent state.

    Args:
      mixed_qkv: Combined QKV tensor of shape `(num_tokens, 2 * n_kq * d_k + n_v *
        d_v)`.
      b: B tensor of shape `(num_tokens, n_v)`.
      a: A tensor of shape `(num_tokens, n_v)`.
      recurrent_state: Recurrent state of shape `(max_reqs, n_v, d_k, d_v)`.
      A_log: Log of A parameter of shape `(n_v,)`.
      dt_bias: Delta T bias of shape `(n_v,)`.
      query_start_loc: Tensor of shape `(num_seqs + 1,)` containing the start
        indices of each sequence, with the last element being the total number of
        valid tokens.
      state_indices: Tensor of shape `(max_reqs,)` mapping request index to state
        index.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Dimension of key.
      d_v: Dimension of value.

    Returns:
      A tuple containing:
      - updated_recurrent_state: The updated recurrent state of shape `(max_reqs,
        n_v, d_k, d_v)`.
      - output: The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k
    query = mixed_qkv[..., :key_dim]
    key = mixed_qkv[..., key_dim:key_dim * 2]
    value = mixed_qkv[..., key_dim * 2:]
    max_reqs = state_indices.shape[0]
    token_idx = jnp.arange(num_tokens)

    req_indices = (
        jnp.sum(token_idx[:, None] >= query_start_loc[None, :], axis=1) - 1)
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
    valid_mask = token_idx < query_start_loc[-1]

    def scan_fn(carry, xs):
        recurrent_state_all = carry
        (
            curr_q,
            curr_k,
            curr_v,
            curr_b,
            curr_a,
            request_index,
            is_valid_token,
        ) = xs

        curr_q = curr_q[None, None, :]
        curr_k = curr_k[None, None, :]
        curr_v = curr_v[None, None, :]
        curr_b = curr_b[None, None, :]
        curr_a = curr_a[None, None, :]

        state_index = state_indices[request_index]
        recurrent_state = recurrent_state_all[state_index][None, ...]

        B, T = 1, 1
        query_reshaped = curr_q.reshape(B, T, n_kq, d_k)
        key_reshaped = curr_k.reshape(B, T, n_kq, d_k)
        value_reshaped = curr_v.reshape(B, T, n_v, d_v)

        beta = jax.nn.sigmoid(curr_b)
        g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
            curr_a.astype(jnp.float32) + dt_bias.astype(jnp.float32))

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            query_reshaped = jnp.repeat(query_reshaped, repeat_factor, axis=2)
            key_reshaped = jnp.repeat(key_reshaped, repeat_factor, axis=2)

        query_reshaped = jnp.transpose(query_reshaped,
                                       (0, 2, 1, 3)).astype(jnp.float32)
        key_reshaped = jnp.transpose(key_reshaped,
                                     (0, 2, 1, 3)).astype(jnp.float32)
        value_reshaped = jnp.transpose(value_reshaped,
                                       (0, 2, 1, 3)).astype(jnp.float32)
        beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
        g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

        query_reshaped = _l2_normalize(query_reshaped)
        key_reshaped = _l2_normalize(key_reshaped)

        output, new_recurrent_state = _recurrent_gated_delta_rule_step(
            query_reshaped,
            key_reshaped,
            value_reshaped,
            g,
            beta,
            state=recurrent_state,
        )

        output = jnp.transpose(output, (0, 2, 1, 3)).astype(query.dtype)
        output = output.reshape(B, T, -1)

        recurrent_state_all = jnp.where(
            is_valid_token,
            recurrent_state_all.at[state_index].set(new_recurrent_state[0]),
            recurrent_state_all,
        )

        return recurrent_state_all, output[0, 0]

    carry_init = recurrent_state
    xs = (query, key, value, b, a, req_indices, valid_mask)

    new_recurrent_state, output = jax.lax.scan(scan_fn, carry_init, xs)
    return new_recurrent_state, output


def run_jax_gdn_attention_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local JAX GDN attention mechanism with combined QKV tensors.

    Args:
        mixed_qkv: Combined QKV tensor of shape `(num_tokens, dim)`.
        b: B tensor of shape `(num_tokens, n_v)`.
        a: A tensor of shape `(num_tokens, n_v)`.
        conv_state: Combined convolutional state of shape `(max_reqs, kernel_size
          - 1, dim)`.
        recurrent_state: Recurrent state of shape `(max_reqs, n_v, d_k, d_v)`.
        conv_weight: Combined convolutional weight of shape `(dim, 1,
          kernel_size)`.
        conv_bias: Optional combined convolutional bias of shape `(dim,)`.
        A_log: Log of A parameter of shape `(n_v,)`.
        dt_bias: Delta T bias of shape `(n_v,)`.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
  
    Returns:
        A tuple containing the new states and the output.
        - A tuple of (new_conv_state, new_recurrent_state).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    # Ensure query_start_loc is monotonically increasing to handle padded slots
    query_start_loc = jnp.maximum.accumulate(query_start_loc)

    out_mixed_qkv, new_conv_state = ragged_conv1d(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        kernel_size,
    )

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    new_recurrent_state, output = ragged_gated_delta_rule(
        out_mixed_qkv,
        b,
        a,
        recurrent_state,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        n_kq,
        n_v,
        d_k,
        d_v,
    )

    return (new_conv_state, new_recurrent_state), output


def run_jax_gdn_attention(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the Jax GDN attention mechanism.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        conv_state: Convolutional state tensor of shape `(max_reqs, kernel_size -
          1, dim)`.
        recurrent_state: Recurrent state tensor of shape `(max_reqs, n_v, d_k,
          d_v)`.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.

    Returns:
        A tuple containing the new states and the output.
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(max_reqs, kernel_size - 1, dim)`
          - new_recurrent_state: `(max_reqs, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    in_specs = (
        P(None, ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(None, ShardingAxisName.ATTN_HEAD),  # j_b
        P(None, ShardingAxisName.ATTN_HEAD),  # j_a
        P(None, None, ShardingAxisName.ATTN_HEAD),  # conv_state
        P(None, ShardingAxisName.ATTN_HEAD, None, None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(),  # query_start_loc
        P(),  # state_indices
    )

    out_specs = (
        (
            P(None, None, ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(None, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(None, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = mesh.shape[ShardingAxisName.ATTN_HEAD]

    p_run_jax_gdn_attention_local = functools.partial(
        run_jax_gdn_attention_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
    )

    return (new_conv_state, new_recurrent_state), output


def gdn_attention_core_tpu(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
    mesh: jax.sharding.Mesh,
) -> None:
    """
    This acts as main bridge between PyTorch and JAX for the GDN core attention.
    Uses a robust, token-by-token scan to inherently handle any mix of
    ragged prefill and decode sequences without dynamic shape compilation errors.

    Some key details:
    1. Cache Mapping: We'll read vLLM's  `block_tables` and `query_start_loc`
       and translate them into static index arrays (`req_indices` and `state_indices`).
    2. JAX Scan: We use `jax.lax.scan` to perform a robust, token-by-token loop over
       the flat inputs. This allows us to handle ANY mix of prefill and decode tokens
       in a single compiled XLA graph.
    3. Conditional Updates: The `valid_mask` ensures that padded dummy tokens
       (used to keep the tensor shape static) do not corrupt the recurrent state
       in the cache.
    """
    fc = get_forward_context()
    attn_metadata = fc.attn_metadata[layer_name]

    layer_module = fc.no_compile_layers[layer_name]
    vllm_context = get_vllm_model_wrapper_context()

    n_kq = layer_module.num_k_heads
    n_v = layer_module.num_v_heads
    d_k = layer_module.head_k_dim
    d_v = layer_module.head_v_dim
    kernel_size = layer_module.conv_kernel_size

    j_mixed_qkv = jax_view(mixed_qkv)  # [num_tokens, dim]
    j_b = jax_view(b)
    j_a = jax_view(a)

    j_conv_weight = jax_view(layer_module.conv1d.weight)
    j_conv_bias = jax_view(layer_module.conv1d.bias
                           ) if layer_module.conv1d.bias is not None else None
    j_A_log = jax_view(layer_module.A_log)
    j_dt_bias = jax_view(layer_module.dt_bias)

    # The j_mixed_qkv and j_conv_weight are not in an interleaved layout.
    # E.g. they are in [Q Q | K K | V V] layout. We need [Q K | Q K | Q K] layout.
    # Use reorder_concatenated_tensor_for_sharding to reorder into correct layout
    key_dim = n_kq * d_k
    value_dim = n_v * d_k
    tp_size = mesh.shape[ShardingAxisName.ATTN_HEAD]
    j_mixed_qkv = reorder_concatenated_tensor_for_sharding(
        j_mixed_qkv, [key_dim, key_dim, value_dim], tp_size, -1)
    j_conv_weight = reorder_concatenated_tensor_for_sharding(
        j_conv_weight, [key_dim, key_dim, value_dim], tp_size, 0)

    layer_idx = vllm_context.layer_name_to_kvcache_index[layer_name]
    conv_state, recurrent_state = vllm_context.kv_caches[layer_idx]

    # Map physical cache blocks
    flat_block_tables = jax_view(attn_metadata.block_tables)
    max_reqs = attn_metadata.seq_lens.shape[0]
    max_blocks_per_req = flat_block_tables.shape[0] // max_reqs
    block_tables_2d = jnp.reshape(flat_block_tables,
                                  (max_reqs, max_blocks_per_req))
    state_indices = block_tables_2d[:, 0].astype(jnp.int32)

    # Map tokens to their respective requests
    q_loc = jax_view(attn_metadata.query_start_loc)

    (new_conv_state,
     new_recurrent_state), j_output = run_jax_gdn_attention(j_mixed_qkv,
                                                            j_b,
                                                            j_a,
                                                            conv_state,
                                                            recurrent_state,
                                                            j_conv_weight,
                                                            j_conv_bias,
                                                            j_A_log,
                                                            j_dt_bias,
                                                            state_indices,
                                                            q_loc,
                                                            n_kq,
                                                            n_v,
                                                            d_k,
                                                            d_v,
                                                            kernel_size,
                                                            mesh=mesh)

    vllm_context.kv_caches[layer_idx] = (new_conv_state, new_recurrent_state)

    j_output_flat = j_output.reshape(core_attn_out.shape)
    core_attn_out.copy_(torch_view(j_output_flat))


def gdn_in_proj_tpu(
    hidden_states: torch.Tensor,
    qkvz_size: int,
    ba_size: int,
    prefix: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch OP replacement for gdn_in_proj.
    Executes the underlying linear layers (in_proj_qkvz and in_proj_ba) directly
    to mimic the old/non-kernel execution path that was reverted in:
    https://github.com/vllm-project/vllm/pull/36795

    Args:
        hidden_states: Tensor of shape (num_tokens, hidden_size)
        qkvz_size: int, unused but needed for signature
        ba_size: int, unused but needed for signature
        prefix: str
    Returns:
        mixed_qkvz: Tensor of shape (num_tokens, qkvz_size)
        ba: Tensor of shape (num_tokens, ba_size)
    """
    fc = get_forward_context()
    # The 'prefix' argument perfectly matches the key used to register the module
    layer_module = fc.no_compile_layers[prefix]

    # Run the original projections instead of the fused C++ kernel
    mixed_qkvz, _ = layer_module.in_proj_qkvz(hidden_states)
    ba, _ = layer_module.in_proj_ba(hidden_states)

    return mixed_qkvz, ba


def apply_gated_delta_net_torch_ops_patch(mesh: jax.sharding.Mesh) -> None:
    """
    This is a patch to inject the `gdn_attention_core` op so the
    Torch/GPU  kernel is bypassed in favor of the TPU kernel
    here:
    https://github.com/vllm-project/vllm/blob/697e4ff3528c72806a4d00ed9b7581332b9efd43/vllm/model_executor/models/qwen3_next.py#L671

    """
    try:
        import vllm.model_executor.models.qwen3_next  # noqa: F401
    except ImportError:
        pass

    # Ensure the op exists in the namespace, which initializes the OpOverloadPacket
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm,
                                              "gdn_attention_core"):
        # dummy call to ensure the op is registered
        torch.ops.vllm.gdn_attention_core = functools.partial(
            gdn_attention_core_tpu, mesh=mesh)

    if hasattr(torch.ops.vllm, "gdn_in_proj"):
        # dummy call to ensure the op is registered
        torch.ops.vllm.gdn_in_proj = gdn_in_proj_tpu
