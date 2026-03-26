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
from jax.sharding import PartitionSpec as P
import torch
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


def _causal_conv1d(
    x: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray] = None,
    kernel_size: int = 4,
) -> jnp.ndarray:
    """Apply causal depthwise 1D convolution.

    Needed because Delta Networks usually prepend a
    sliding-window Conv1D before the recurrent mechanism to mix local token
    context efficiently. This function handles the parallelized computation
    over the entire sequence during the `prefill` phase.


    Args:
        x: (B, T, C) input
        conv_weight: (C, 1, kernel_size) depthwise kernel (PyTorch Conv1d format)
        conv_bias: optional (C,) bias
        kernel_size: convolution kernel size

    Returns:
        (B, T, C) output (same length due to causal padding)
    """
    _, _, C = x.shape

    # Left-pad by (kernel_size - 1) for causal
    x_padded = jnp.pad(x, ((0, 0), (kernel_size - 1, 0), (0, 0)))

    # Transpose to (B, C, T_padded) for conv
    x_t = jnp.transpose(x_padded, (0, 2, 1))

    # Ensure weight shape: (C, 1, kernel_size) for depthwise
    if conv_weight.ndim == 2:
        w = jnp.transpose(conv_weight, (1, 0))[:, jnp.newaxis, :]
    else:
        w = conv_weight  # already (C, 1, kernel_size)

    # Depthwise conv: feature_group_count = C
    out = jax.lax.conv_general_dilated(
        x_t,
        w,
        window_strides=(1, ),
        padding="VALID",
        feature_group_count=C,
    )

    # Transpose back: (B, C, T) → (B, T, C)
    out = jnp.transpose(out, (0, 2, 1))

    if conv_bias is not None:
        out = out + conv_bias[jnp.newaxis, jnp.newaxis, :]

    return out


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


def _jax_gdn_attention_core(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    is_prefill: bool,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure JAX implementation of the GDN sequence and recurrence logic.

    Args:
        mixed_qkv: (B, T, C)
        b: (B, T, n_v)
        a: (B, T, n_v)
        conv_state: (B, T, C)
        recurrent_state: (B, H, d_k, d_v)
        conv_weight: (C, 1, kernel_size)
        conv_bias: (C,)
        A_log: (n_v,)
        dt_bias: (n_v,)
        is_prefill: bool
        n_kq: int
        n_v: int
        d_k: int
        d_v: int
        kernel_size: int

    Returns:
        output: (B, T, n_v * d_v)
        new_conv_state: (B, T, C)
        new_recurrent_state: (B, H, d_k, d_v)

    """
    B, T, _ = mixed_qkv.shape
    key_dim = n_kq * d_k

    # 1. Causal Conv1D
    if is_prefill:
        # new_conv_state = mixed_qkv[:, -(kernel_size - 1):, :]
        T_conv = kernel_size - 1
        new_conv_state = mixed_qkv[:, -T_conv:, :]
        if new_conv_state.shape[1] < T_conv:
            pad_len = T_conv - new_conv_state.shape[1]
            new_conv_state = jnp.pad(new_conv_state,
                                     ((0, 0), (pad_len, 0), (0, 0)))
        mixed_qkv = _causal_conv1d(mixed_qkv, conv_weight, conv_bias,
                                   kernel_size)
    else:
        mixed_qkv, new_conv_state = _causal_conv1d_step(
            mixed_qkv, conv_state, conv_weight, conv_bias)
    mixed_qkv = jax.nn.silu(mixed_qkv)

    # 2. Split Q, K, V
    query = mixed_qkv[..., :key_dim].reshape(B, T, n_kq, d_k)
    key = mixed_qkv[..., key_dim:key_dim * 2].reshape(B, T, n_kq, d_k)
    value = mixed_qkv[..., key_dim * 2:].reshape(B, T, n_v, d_v)

    # 3. Compute continuous decay (g) and input gate (beta)
    beta = jax.nn.sigmoid(b)
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a.astype(jnp.float32) + dt_bias.astype(jnp.float32))

    # 4. Head Expansion (GQA -> MHA logic)
    repeat_factor = n_v // n_kq
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
    if is_prefill:
        output, new_recurrent_state = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            initial_state=recurrent_state,
            output_final_state=True)
    else:
        output, new_recurrent_state = _recurrent_gated_delta_rule_step(
            query, key, value, g, beta, state=recurrent_state)

    # Output back to (B, T, H, d_v) -> (B, T, H * d_v)
    output = jnp.transpose(output, (0, 2, 1, 3)).astype(mixed_qkv.dtype)
    output = output.reshape(B, T, -1)

    return output, new_conv_state, new_recurrent_state


def run_jax_gdn_attention_local(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    req_indices: jnp.ndarray,
    valid_mask: jnp.ndarray,
    state_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    attn_head_axis_size = jax.lax.axis_size("model")
    n_kq = n_kq // attn_head_axis_size
    n_v = n_v // attn_head_axis_size
    
    def scan_fn(carry, xs):
        c_state_all, r_state_all = carry
        curr_qkv, curr_b, curr_a, req_idx, is_valid = xs

        curr_qkv = curr_qkv[None, None, :]
        curr_b = curr_b[None, None, :]
        curr_a = curr_a[None, None, :]

        state_idx = state_indices[req_idx]
        c_state = c_state_all[state_idx][None, ...]
        r_state = r_state_all[state_idx][None, ...]

        out, new_c, new_r = _jax_gdn_attention_core(curr_qkv,
                                                    curr_b,
                                                    curr_a,
                                                    c_state,
                                                    r_state,
                                                    j_conv_weight,
                                                    j_conv_bias,
                                                    j_A_log,
                                                    j_dt_bias,
                                                    is_prefill=False,
                                                    n_kq=n_kq,
                                                    n_v=n_v,
                                                    d_k=d_k,
                                                    d_v=d_v,
                                                    kernel_size=kernel_size)

        c_state_all = jnp.where(is_valid,
                                c_state_all.at[state_idx].set(new_c[0]),
                                c_state_all)
        r_state_all = jnp.where(is_valid,
                                r_state_all.at[state_idx].set(new_r[0]),
                                r_state_all)

        return (c_state_all, r_state_all), out[0, 0]

    carry_init = (conv_state, recurrent_state)
    xs = (j_mixed_qkv, j_b, j_a, req_indices, valid_mask)

    (new_conv_state,
     new_recurrent_state), j_output = jax.lax.scan(scan_fn, carry_init, xs)

    return (new_conv_state, new_recurrent_state), j_output


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
    req_indices: jnp.ndarray,
    valid_mask: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    bias_spec = P(ShardingAxisName.ATTN_HEAD) if j_conv_bias is not None else None
    
    in_specs = (
        P(None, ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv: (num_tokens, C)
        P(None, ShardingAxisName.ATTN_HEAD),  # j_b: (num_tokens, n_v)
        P(None, ShardingAxisName.ATTN_HEAD),  # j_a: (num_tokens, n_v)
        P(None, None, ShardingAxisName.ATTN_HEAD),  # conv_state: (num_blocks, kernel_size-1, C)
        P(None, ShardingAxisName.ATTN_HEAD, None, None),  # recurrent_state: (num_blocks, n_v, d_k, d_v)
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight: (C, 1, kernel_size)
        bias_spec,  # j_conv_bias: (C,)
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log: (n_v,)
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias: (n_v,)
        P(),  # req_indices: (num_tokens,)
        P(),  # valid_mask: (num_tokens,)
        P(),  # state_indices: (num_reqs,)
        None,  # n_kq
        None,  # n_v
        None,  # d_k
        None,  # d_v
        None,  # kernel_size
    )
    
    out_specs = (
        (
            P(None, None, ShardingAxisName.ATTN_HEAD),
            P(None, ShardingAxisName.ATTN_HEAD, None, None)),  # (new_conv_state, new_recurrent_state)
        P(None, ShardingAxisName.ATTN_HEAD),  # j_output: (num_tokens, n_v * d_v)
    )
    
    mapped_fn = jax.shard_map(
        run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    
    return mapped_fn(
        j_mixed_qkv, j_b, j_a, conv_state, recurrent_state,
        j_conv_weight, j_conv_bias, j_A_log, j_dt_bias,
        req_indices, valid_mask, state_indices,
        n_kq, n_v, d_k, d_v, kernel_size,
    )


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

    tp_size = layer_module.tp_size
    n_kq = layer_module.num_k_heads // tp_size
    n_v = layer_module.num_v_heads // tp_size
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
    # Ensure q_loc is monotonically increasing to handle padded slots
    q_loc = jnp.maximum.accumulate(q_loc)
    num_tokens = j_mixed_qkv.shape[0]

    token_idx = jnp.arange(num_tokens)
    req_indices = jnp.sum(token_idx[:, None] >= q_loc[None, :], axis=1) - 1
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)

    # Exclude trailing padding indices from mutating cache
    valid_mask = token_idx < q_loc[-1]

    (new_conv_state,
     new_recurrent_state), j_output = run_jax_gdn_attention(
         j_mixed_qkv, j_b, j_a, conv_state, recurrent_state,
         j_conv_weight, j_conv_bias, j_A_log, j_dt_bias,
         state_indices, req_indices, valid_mask,
         n_kq, n_v, d_k, d_v, kernel_size, mesh=mesh
     )

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
