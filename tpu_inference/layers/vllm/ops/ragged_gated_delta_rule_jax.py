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
"""Ragged gated delta rule JAX implementation."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


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


def _recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step recurrent update for decode.

    Args:
      query: Tensor of shape `(B, H, T, d_k)`. Current implementation assumes
        `T=1`.
      key: Tensor of shape `(B, H, T, d_k)`. Current implementation assumes `T=1`.
      value: Tensor of shape `(B, H, T, d_v)`. Current implementation assumes
        `T=1`.
      g: Tensor of shape `(B, H, T)`. Current implementation assumes `T=1`.
      beta: Tensor of shape `(B, H, T)`. Current implementation assumes `T=1`.
      state: Optional initial recurrent state of shape `(B, H, d_k, d_v)`.

    Returns:
      A tuple containing:
      - output: The output tensor of shape `(B, H, T, d_v)`.
      - new_state: The updated recurrent state of shape `(B, H, d_k, d_v)`.
    """
    B, H, T, d_k = query.shape
    d_v = value.shape[-1]

    if state is None:
        state = jnp.zeros((B, H, d_k, d_v), dtype=query.dtype)

    q = query[:, :, 0]
    k = key[:, :, 0]
    v = value[:, :, 0]
    beta_val = beta[:, :, 0]
    g_val = g[:, :, 0]

    scale = d_k**-0.5
    q = q * scale

    # v_diff = v - e^g * (k @ state)
    k_state = jnp.einsum("bhd, bhdm -> bhm", k, state)
    v_diff = v - jnp.exp(g_val)[..., None] * k_state

    # v_new = beta * v_diff
    v_new = beta_val[..., None] * v_diff

    # out = e^g * (q @ state) + (q . k) * v_new
    q_state = jnp.einsum("bhd, bhdm -> bhm", q, state)
    q_k = jnp.sum(q * k, axis=-1, keepdims=True)

    out = jnp.exp(g_val)[..., None] * q_state + q_k * v_new

    # s_new = state * exp(g) + k outer v_new
    k_v_new = jnp.einsum("bhd, bhm -> bhdm", k, v_new)
    new_state = state * jnp.exp(g_val)[..., None, None] + k_v_new

    return out[:, :, None, :], new_state


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
