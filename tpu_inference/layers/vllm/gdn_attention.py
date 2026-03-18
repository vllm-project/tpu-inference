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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context

from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """L2 normalize along last dimension."""
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def _causal_conv1d(
    x: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray] = None,
    kernel_size: int = 4,
) -> jnp.ndarray:
    """Apply causal depthwise 1D convolution.

    Args:
        x: (B, T, C) input
        conv_weight: (C, 1, kernel_size) depthwise kernel (PyTorch Conv1d format)
        conv_bias: optional (C,) bias
        kernel_size: convolution kernel size

    Returns:
        (B, T, C) output (same length due to causal padding)
    """
    B, T, C = x.shape

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
    """Single-step causal conv1d for decode mode."""
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
    query,
    key,
    value,
    g,
    beta,
    state,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Single-step for decode. Not yet optimized."""
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
):
    """Pure JAX implementation of the GDN sequence and recurrence logic."""
    B, T, _ = mixed_qkv.shape
    key_dim = n_kq * d_k

    # 1. Causal Conv1D
    if is_prefill:
        new_conv_state = mixed_qkv[:, -(kernel_size - 1):, :]
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


def gdn_attention_core_tpu(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """
    JAX Bridge for the GDN core attention.
    Intercepts the torch.ops.vllm call, fetches JAX state, and executes.
    """
    fc = get_forward_context()
    attn_metadata = fc.attn_metadata

    # Get the PyTorch module instance for this specific layer
    layer_module = fc.no_compile_layers[layer_name]
    vllm_context = get_vllm_model_wrapper_context()

    # 1. Resolve dimensions (Accounting for Tensor Parallelism)
    tp_size = layer_module.tp_size
    n_kq = layer_module.num_k_heads // tp_size
    n_v = layer_module.num_v_heads // tp_size
    d_k = layer_module.head_k_dim
    d_v = layer_module.head_v_dim
    kernel_size = layer_module.conv_kernel_size

    # In vLLM, inputs are flattened to [num_tokens, dim].
    # We must unflatten to[B, T, dim] for JAX convolutions.
    num_tokens = mixed_qkv.shape[0]
    is_prefill = getattr(attn_metadata, "num_prefills", 0) > 0

    if is_prefill:
        B = attn_metadata.num_prefills
        T = num_tokens // B
    else:
        B = num_tokens
        T = 1

    j_mixed_qkv = jax_view(mixed_qkv).reshape(B, T, -1)
    j_b = jax_view(b).reshape(B, T, -1)
    j_a = jax_view(a).reshape(B, T, -1)

    j_conv_weight = jax_view(layer_module.conv1d.weight)
    j_conv_bias = jax_view(layer_module.conv1d.bias
                           ) if layer_module.conv1d.bias is not None else None
    j_A_log = jax_view(layer_module.A_log)
    j_dt_bias = jax_view(layer_module.dt_bias)

    # 3. Retrieve KV Caches
    layer_idx = vllm_context.layer_name_to_kvcache_index[layer_name]
    conv_state, recurrent_state = vllm_context.kv_caches[layer_idx]

    # PyTorch vLLM keeps conv_state as (B, C, kernel_size-1)
    # The JAX op expects (B, kernel_size-1, C), so we transpose it
    j_conv_state = jnp.transpose(conv_state, (0, 2, 1))

    # 4. Run Pure JAX Logic
    j_output, j_new_conv_state, j_new_recurrent_state = _jax_gdn_attention_core(
        j_mixed_qkv, j_b, j_a, j_conv_state, recurrent_state, j_conv_weight,
        j_conv_bias, j_A_log, j_dt_bias, is_prefill, n_kq, n_v, d_k, d_v,
        kernel_size)

    # Transpose the conv state back to PyTorch's expected alignment
    j_new_conv_state = jnp.transpose(j_new_conv_state, (0, 2, 1))

    # 5. Update KV Caches in the TPU context
    vllm_context.kv_caches[layer_idx] = (j_new_conv_state,
                                         j_new_recurrent_state)

    # 6. Write output back to PyTorch out-tensor
    # Reshape JAX output back to [num_tokens, n_v, d_v] and copy into `core_attn_out`
    j_output_flat = j_output.reshape(core_attn_out.shape)
    core_attn_out.copy_(torch_view(j_output_flat))


def apply_gated_delta_net_torch_ops_patch():
    try:
        import vllm.model_executor.models.qwen3_next  # noqa: F401
    except ImportError:
        pass

    # Ensure the op exists in the namespace, which initializes the OpOverloadPacket
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm,
                                              "gdn_attention_core"):
        torch.ops.vllm.gdn_attention_core = gdn_attention_core_tpu
