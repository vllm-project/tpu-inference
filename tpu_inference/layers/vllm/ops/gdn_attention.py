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


def _causal_conv1d_step(
    x_new: jnp.ndarray,
    conv_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step causal conv1d for decode mode.

    During decode, instead of padding and convoluting over the whole sequence, we
    maintain a rolling KV-cache equivalent (`conv_state`) of the last `kernel_size - 1` tokens
    and compute a single step of the depthwise convolution

    Args:
        x_new: (B, 1, C) input
        conv_state: (B, T, C) state
        conv_weight: (C, 1, kernel_size) depthwise kernel (PyTorch Conv1d format)
        conv_bias: optional (C,) bias

    Returns:
        (B, 1, C) output
        (B, T, C) new state
    """
    new_state = jnp.concatenate([conv_state[:, 1:, :], x_new], axis=1)
    window = jnp.concatenate([conv_state, x_new], axis=1)

    if conv_weight.ndim == 2:
        w = conv_weight
    else:
        w = conv_weight[:, 0, :].T  # (C,1,K) → (K,C)

    out = jnp.sum(window * w[jnp.newaxis, :, :], axis=1, keepdims=True)

    if conv_bias is not None:
        out = out + conv_bias[jnp.newaxis, jnp.newaxis, :]

    return out, new_state


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

    One detail worth pointing out is that the continuous decay mask (`g`) is cumulative, so
    Applying the triangular mask *before* exponentiation is key here to prevent NaNs
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


def run_jax_gdn_attention_local(
    l_query: jnp.ndarray,
    l_key: jnp.ndarray,
    l_value: jnp.ndarray,
    l_b: jnp.ndarray,
    l_a: jnp.ndarray,
    l_conv_state_q: jnp.ndarray,
    l_conv_state_k: jnp.ndarray,
    l_conv_state_v: jnp.ndarray,
    l_recurrent_state: jnp.ndarray,
    l_conv_weight_q: jnp.ndarray,
    l_conv_weight_k: jnp.ndarray,
    l_conv_weight_v: jnp.ndarray,
    l_conv_bias_q: jnp.ndarray,
    l_conv_bias_k: jnp.ndarray,
    l_conv_bias_v: jnp.ndarray,
    l_A_log: jnp.ndarray,
    l_dt_bias: jnp.ndarray,
    l_q_loc: jnp.ndarray,
    l_state_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
           jnp.ndarray]:
    """Runs the local JAX GDN attention mechanism.

    Args:
        l_query: Query tensor of shape `(num_tokens, local_n_kq * d_k)`.
        l_key: Key tensor of shape `(num_tokens, local_n_kq * d_k)`.
        l_value: Value tensor of shape `(num_tokens, local_n_v * d_v)`.
        l_b: B tensor of shape `(num_tokens, local_n_v)`.
        l_a: A tensor of shape `(num_tokens, local_n_v)`.
        l_conv_state_q: Convolutional state for query of shape `(max_reqs, kernel_size - 1, local_n_kq * d_k)`.
        l_conv_state_k: Convolutional state for key of shape `(max_reqs, kernel_size - 1, local_n_kq * d_k)`.
        l_conv_state_v: Convolutional state for value of shape `(max_reqs, kernel_size - 1, local_n_v * d_v)`.
        l_recurrent_state: Recurrent state of shape `(max_reqs, local_n_v, d_k, d_v)`.
        l_conv_weight_q: Convolutional weight for query of shape `(local_n_kq * d_k, 1, kernel_size)`.
        l_conv_weight_k: Convolutional weight for key of shape `(local_n_kq * d_k, 1, kernel_size)`.
        l_conv_weight_v: Convolutional weight for value of shape `(local_n_v * d_v, 1, kernel_size)`.
        l_conv_bias_q: Convolutional bias for query of shape `(local_n_kq * d_k,)`.
        l_conv_bias_k: Convolutional bias for key of shape `(local_n_kq * d_k,)`.
        l_conv_bias_v: Convolutional bias for value of shape `(local_n_v * d_v,)`.
        l_A_log: Log of A parameter of shape `(local_n_v,)`.
        l_dt_bias: Delta T bias of shape `(local_n_v,)`.
        l_q_loc: Tensor of shape `(num_seqs,)` with start locations of each sequence.
        l_state_indices: Tensor of shape `(max_reqs,)` mapping request index to state index.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.

    Returns:
        A tuple containing the new states and the output.
        - A tuple of (new_conv_state_q, new_conv_state_k, new_conv_state_v, new_recurrent_state).
        - The output tensor of shape `(num_tokens, local_n_v * d_v)`.
    """
    # Ensure q_loc is monotonically increasing to handle padded slots
    l_q_loc = jnp.maximum.accumulate(l_q_loc)

    num_tokens = l_query.shape[0]
    token_idx = jnp.arange(num_tokens)
    max_reqs = l_state_indices.shape[0]

    req_indices = jnp.sum(token_idx[:, None] >= l_q_loc[None, :], axis=1) - 1
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)

    # Exclude trailing padding indices from mutating cache
    valid_mask = token_idx < l_q_loc[-1]
    attn_head_axis_size = jax.lax.axis_size("model")
    local_n_kq = n_kq // attn_head_axis_size
    local_n_v = n_v // attn_head_axis_size

    def scan_fn(carry, xs):
        conv_state_q, conv_state_k, conv_state_v, recurrent_state_all = carry
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

        state_index = l_state_indices[request_index]
        cs_q = conv_state_q[state_index][None, ...]
        cs_k = conv_state_k[state_index][None, ...]
        cs_v = conv_state_v[state_index][None, ...]
        recurrent_state = recurrent_state_all[state_index][None, ...]

        # 1. Causal Conv1D
        query_conv, new_conv_state_q = _causal_conv1d_step(
            curr_q, cs_q, l_conv_weight_q, l_conv_bias_q)
        key_conv, new_conv_state_k = _causal_conv1d_step(
            curr_k, cs_k, l_conv_weight_k, l_conv_bias_k)
        value_conv, new_conv_state_v = _causal_conv1d_step(
            curr_v, cs_v, l_conv_weight_v, l_conv_bias_v)

        mixed_qkv = jnp.concatenate([query_conv, key_conv, value_conv],
                                    axis=-1)
        mixed_qkv = jax.nn.silu(mixed_qkv)

        # 2. Split Q, K, V
        B, T, _ = mixed_qkv.shape
        key_dim = local_n_kq * d_k
        query = mixed_qkv[..., :key_dim].reshape(B, T, local_n_kq, d_k)
        key = mixed_qkv[...,
                        key_dim:key_dim * 2].reshape(B, T, local_n_kq, d_k)
        value = mixed_qkv[..., key_dim * 2:].reshape(B, T, local_n_v, d_v)

        # 3. Compute continuous decay (g) and input gate (beta)
        beta = jax.nn.sigmoid(curr_b)
        g = -jnp.exp(l_A_log.astype(jnp.float32)) * jax.nn.softplus(
            curr_a.astype(jnp.float32) + l_dt_bias.astype(jnp.float32))

        # 4. Head Expansion (GQA -> MHA logic)
        repeat_factor = local_n_v // local_n_kq
        if repeat_factor > 1:
            query = jnp.repeat(query, repeat_factor, axis=2)
            key = jnp.repeat(key, repeat_factor, axis=2)

        # Transpose to (B, H, T, dim)
        query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
        key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
        value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
        beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
        g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

        # L2 normalize Q, K
        query = _l2_normalize(query)
        key = _l2_normalize(key)

        # 5. Delta Rule Recurrence
        output, new_recurrent_state = _recurrent_gated_delta_rule_step(
            query, key, value, g, beta, state=recurrent_state)

        # Output back to (B, T, H, d_v) -> (B, T, H * d_v)
        output = jnp.transpose(output, (0, 2, 1, 3)).astype(mixed_qkv.dtype)
        output = output.reshape(B, T, -1)

        conv_state_q = jnp.where(
            is_valid_token,
            conv_state_q.at[state_index].set(new_conv_state_q[0]),
            conv_state_q)
        conv_state_k = jnp.where(
            is_valid_token,
            conv_state_k.at[state_index].set(new_conv_state_k[0]),
            conv_state_k)
        conv_state_v = jnp.where(
            is_valid_token,
            conv_state_v.at[state_index].set(new_conv_state_v[0]),
            conv_state_v)
        recurrent_state_all = jnp.where(
            is_valid_token,
            recurrent_state_all.at[state_index].set(new_recurrent_state[0]),
            recurrent_state_all)

        return (conv_state_q, conv_state_k, conv_state_v,
                recurrent_state_all), output[0, 0]

    carry_init = (l_conv_state_q, l_conv_state_k, l_conv_state_v,
                  l_recurrent_state)
    xs = (l_query, l_key, l_value, l_b, l_a, req_indices, valid_mask)

    (
        (new_conv_state_q, new_conv_state_k, new_conv_state_v,
         new_recurrent_state),
        output,
    ) = jax.lax.scan(scan_fn, carry_init, xs)

    return (new_conv_state_q, new_conv_state_k, new_conv_state_v,
            new_recurrent_state), output


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
    q_loc: jnp.ndarray,
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
        conv_state: Convolutional state tensor of shape
          `(max_reqs, kernel_size - 1, dim)`.
        recurrent_state: Recurrent state tensor of shape
          `(max_reqs, n_v, d_k, d_v)`.
        j_conv_weight: Convolutional weight tensor of shape
          `(dim, 1, kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        q_loc: Tensor of shape `(num_seqs,)` with start locations of each
          sequence.
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
    key_dim = n_kq * d_k

    # NOTE: we pre-split Q, K, and V BEFORE passing to shard map,
    # otherwise, jax.shard_map splits across C dimension
    j_query = j_mixed_qkv[..., :key_dim]
    j_key = j_mixed_qkv[..., key_dim:key_dim * 2]
    j_value = j_mixed_qkv[..., key_dim * 2:]

    conv_state_query = conv_state[..., :key_dim]
    conv_state_key = conv_state[..., key_dim:key_dim * 2]
    conv_state_value = conv_state[..., key_dim * 2:]

    conv_weight_query = j_conv_weight[:key_dim, ...]
    conv_weight_key = j_conv_weight[key_dim:key_dim * 2, ...]
    conv_weight_value = j_conv_weight[key_dim * 2:, ...]

    if j_conv_bias is not None:
        conv_bias_query = j_conv_bias[:key_dim]
        conv_bias_key = j_conv_bias[key_dim:key_dim * 2]
        conv_bias_value = j_conv_bias[key_dim * 2:]
        bias_spec = P(ShardingAxisName.ATTN_HEAD)
    else:
        conv_bias_query = conv_bias_key = conv_bias_value = None
        bias_spec = None

    in_specs = (
        P(None, ShardingAxisName.ATTN_HEAD),  # j_query
        P(None, ShardingAxisName.ATTN_HEAD),  # j_key
        P(None, ShardingAxisName.ATTN_HEAD),  # j_value
        P(None, ShardingAxisName.ATTN_HEAD),  # j_b
        P(None, ShardingAxisName.ATTN_HEAD),  # j_a
        P(None, None, ShardingAxisName.ATTN_HEAD),  # conv_state_query
        P(None, None, ShardingAxisName.ATTN_HEAD),  # conv_state_key
        P(None, None, ShardingAxisName.ATTN_HEAD),  # conv_state_value
        P(None, ShardingAxisName.ATTN_HEAD, None, None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # conv_weight_query
        P(ShardingAxisName.ATTN_HEAD, None, None),  # conv_weight_key
        P(ShardingAxisName.ATTN_HEAD, None, None),  # conv_weight_value
        bias_spec,
        bias_spec,
        bias_spec,  # conv_bias_query, conv_bias_key, conv_bias_value
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(),  # q_loc
        P(),  # state_indices
    )

    out_specs = (
        (
            P(None, None, ShardingAxisName.ATTN_HEAD),  # new_conv_state_q
            P(None, None, ShardingAxisName.ATTN_HEAD),  # new_conv_state_k
            P(None, None, ShardingAxisName.ATTN_HEAD),  # new_conv_state_v
            P(None, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(None, ShardingAxisName.ATTN_HEAD),  # output
    )

    p_run_jax_gdn_attention_local = functools.partial(
        run_jax_gdn_attention_local,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size)

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    (
        (new_conv_state_q, new_conv_state_k, new_conv_state_v,
         new_recurrent_state),
        output,
    ) = mapped_fn(j_query, j_key, j_value, j_b, j_a, conv_state_query,
                  conv_state_key, conv_state_value, recurrent_state,
                  conv_weight_query, conv_weight_key, conv_weight_value,
                  conv_bias_query, conv_bias_key, conv_bias_value, j_A_log,
                  j_dt_bias, q_loc, state_indices)

    new_conv_state = jnp.concatenate(
        [new_conv_state_q, new_conv_state_k, new_conv_state_v], axis=-1)

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
