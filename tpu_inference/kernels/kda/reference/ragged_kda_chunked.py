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
"""Ragged KDA (Kimi Delta Attention) chunked JAX reference implementation.

KDA (arXiv:2510.26692, used by moonshotai/Kimi-Linear-48B and Kimi-K3) is a
gated delta rule with PER-CHANNEL (diagonal) decay:

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

vs. GDN's scalar per-head decay (see
``kernels/gdn/reference/ragged_gated_delta_rule_chunked.py``, which this file
generalizes). Chunkwise form follows the WY representation + UT transform of
the Kimi Linear paper (eqs 2-9), with the per-channel cumulative log-decay
``G`` entering every decay ratio as a vector over the key dimension.

Numerics notes:
  * Both log-decay ("gate") forms are supported, selected by
    ``gate_lower_bound`` exactly as in the reference implementations
    (HF modeling / flash-linear-attention `ops/kda`):
      - lower_bound is None:  g = -exp(A_log) * softplus(g_raw + dt_bias)
      - lower_bound = L:      g = L * sigmoid(exp(A_log) * (g_raw + dt_bias))
    Both are <= 0, so every intra-chunk decay ratio exp(G_r - G_c) with
    r >= c is <= 1 and safe to exponentiate directly; only these
    "backward-looking" ratios are ever formed.
  * q/k are always L2-normalized (KDA has this baked in), and the attention
    scale is 1/sqrt(K).
  * All decay/gate math runs in fp32; matmuls use HIGHEST precision with
    fp32 accumulation, mirroring the GDN reference.
"""

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_chunked import (
    l2norm, pack_inputs_single_stream)


def kda_gate(
    g_raw: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    gate_lower_bound: float | None = None,
) -> jnp.ndarray:
    """Computes the per-channel log-decay from raw gate inputs, in fp32.

    Args:
      g_raw: Raw gate input of shape ``[..., H, K]`` (output of the low-rank
        f_b(f_a(x)) projection, before activation).
      A_log: Per-head log-A parameter, shape ``[H]`` -- one scalar per head,
        broadcast over that head's K channels. Kimi-K3 checkpoints store it
        zero-padded out to ``[head_dim]`` (leading ``H`` entries are the
        parameter, tail is zeros), so a caller reading a checkpoint directly
        must narrow it to the leading ``[H]`` slice first.
      dt_bias: Per-channel bias, shape ``[H * K]``.
      gate_lower_bound: ``None`` selects the softplus form (Kimi-Linear-48B);
        a float L selects the sigmoid lower-bound form (Kimi-K3, L = -5.0).

    Returns:
      Log-decay of shape ``[..., H, K]``, fp32, always <= 0.
    """
    h = A_log.shape[0]
    if A_log.ndim != 1 or g_raw.shape[-2] != h:
        raise ValueError(
            f"[kda] A_log has shape {A_log.shape} but the gate input carries "
            f"{g_raw.shape[-2]} heads x {g_raw.shape[-1]} channels. A_log is "
            f"one log-decay scalar per head, shape [H]; a checkpoint that "
            f"stores it zero-padded to [head_dim] (Kimi-K3: narrow to the "
            f"leading [H] slice, dropping the zero tail) or as [1, 1, H, 1] "
            f"(Kimi-Linear-48B: reshape to [H]) must be converted before it "
            f"gets here.")
    x = g_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32).reshape(h, -1)
    a = jnp.exp(A_log.astype(jnp.float32))[:, None]
    if gate_lower_bound is None:
        return -a * jax.nn.softplus(x)
    return gate_lower_bound * jax.nn.sigmoid(a * x)


def ragged_kda_mixed_prefill(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    b_reshaped: jnp.ndarray,
    a_reshaped: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray | None = None,
    chunk_size: int = 64,
    gate_lower_bound: float | None = None,
    compute_dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Chunked KDA for the mixed / prefill case (ragged sequences).

    Interface mirrors ``ragged_gated_delta_rule_mixed_prefill``; differences:
      * ``a_reshaped`` (raw gate) has shape ``[num_tokens, H, K]`` — per
        channel, not per head.
      * ``gate_lower_bound`` selects the gate form (see :func:`kda_gate`).
      * q/k L2 norm is always applied.

    Returns ``(updated_recurrent_state, output)`` with output shape
    ``[num_tokens, H * V]``.
    """
    initial_dtype = query.dtype

    beta = jax.nn.sigmoid(b_reshaped.astype(jnp.float32))
    g = kda_gate(a_reshaped, A_log, dt_bias, gate_lower_bound)

    (packed_query, packed_key, packed_value, packed_g, packed_beta, reset_mask,
     new_query_start_loc, padded_indices_valid) = pack_inputs_single_stream(
         query,
         key,
         value,
         g,
         beta,
         query_start_loc,
         distribution,
         chunk_size,
         compute_dtype=compute_dtype,
     )

    packed_query = l2norm(packed_query, dim=-1, eps=1e-6)
    packed_key = l2norm(packed_key, dim=-1, eps=1e-6)

    scale = jax.lax.rsqrt(jnp.array(packed_query.shape[-1],
                                    dtype=jnp.float32)).astype(compute_dtype)
    packed_query = packed_query * scale

    total_tokens = packed_query.shape[0]
    num_chunks = total_tokens // chunk_size
    H = packed_query.shape[1]
    K_dim = packed_query.shape[2]
    V_dim = packed_value.shape[2]

    def to_chunk(x):
        return x.reshape(num_chunks, chunk_size, H, -1).transpose(0, 2, 1, 3)

    q_c = to_chunk(packed_query)
    k_c = to_chunk(packed_key)
    v_c = to_chunk(packed_value)
    g_c = to_chunk(packed_g)  # [N, H, BT, K] fp32 log-decay
    beta_c = packed_beta.reshape(num_chunks, chunk_size,
                                 H).transpose(0, 2, 1)  # [N, H, BT]

    # Per-chunk cumulative log-decay (fp32, <= 0, decreasing in time).
    g_cumsum = jnp.cumsum(g_c, axis=-2)  # [N, H, BT, K]

    # Initial states per chunk (same slot machinery as the GDN reference).
    init_h_per_chunk = jnp.zeros((num_chunks, H, K_dim, V_dim),
                                 dtype=recurrent_state.dtype)
    start_chunk_indices = new_query_start_loc[:-1] // chunk_size
    init_states_for_seqs = recurrent_state[state_indices]
    if has_initial_state is not None:
        init_states_for_seqs = jnp.where(
            has_initial_state[:, None, None, None],
            init_states_for_seqs,
            jnp.zeros_like(init_states_for_seqs),
        )
    init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(
        init_states_for_seqs)

    h_init = jnp.zeros((H, K_dim, V_dim), dtype=jnp.float32)
    mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool),
                           k=-1)
    mask_incl = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    identity = jnp.eye(chunk_size, dtype=jnp.float32)

    xs = (q_c, k_c, v_c, g_cumsum, beta_c, reset_mask, init_h_per_chunk)

    def scan_body(h, args):
        q, k, v, G, beta, reset, init_h = args
        # q,k,v: [H, BT, D]; G: [H, BT, K]; beta: [H, BT]; h: [H, K, V].
        h = jnp.where(reset, init_h.astype(jnp.float32), h)

        kf = k.astype(jnp.float32)
        qf = q.astype(jnp.float32)
        vf = v.astype(jnp.float32)

        # Per-channel decay ratios exp(G_r - G_c) for r >= c (safe: <= 1).
        # [H, BT, BT, K]
        g_diff = G[:, :, None, :] - G[:, None, :, :]
        decay = jnp.exp(jnp.where(mask_incl[None, :, :, None], g_diff, 0.0))

        # Intra-chunk k->k similarity with decay, strictly causal, row-scaled
        # by beta: S[r,c] = beta_r * sum_d k_r exp(G_r - G_c) k_c.
        S = jnp.einsum('hrd,hrcd,hcd->hrc', kf, decay, kf, precision=precision)
        S = jnp.where(mask_strict[None], S, 0.0) * beta[..., None]

        # UT transform: Tmat = (I + S)^{-1}. (I + S) is unit lower-triangular;
        # solve against I instead of using the Pallas triangle solver so this
        # reference stays runnable on any backend (the 1c kernel will use the
        # in-repo Pallas solver).
        Tmat = jax.scipy.linalg.solve_triangular(identity[None] + S,
                                                 jnp.broadcast_to(
                                                     identity[None], S.shape),
                                                 lower=True,
                                                 unit_diagonal=True)

        u = jnp.matmul(Tmat,
                       vf * beta[..., None],
                       precision=precision,
                       preferred_element_type=jnp.float32)
        # w[c] = sum_r Tmat[c,r] * beta_r * k_r * exp(G_r)  (exp(G) <= 1).
        k_beta_g = kf * beta[..., None] * jnp.exp(G)
        w = jnp.matmul(Tmat,
                       k_beta_g,
                       precision=precision,
                       preferred_element_type=jnp.float32)

        # Intra-chunk attention (inclusive diagonal) with per-channel decay.
        attn_i = jnp.einsum('hrd,hrcd,hcd->hrc',
                            qf,
                            decay,
                            kf,
                            precision=precision)
        attn_i = jnp.where(mask_incl[None], attn_i, 0.0)

        v_prime = jnp.matmul(w,
                             h,
                             precision=precision,
                             preferred_element_type=jnp.float32)
        v_new = u - v_prime

        # Inter-chunk contribution: (q * exp(G)) @ h.
        q_g = qf * jnp.exp(G)
        o_c = jnp.matmul(q_g,
                         h,
                         precision=precision,
                         preferred_element_type=jnp.float32)
        o_c = o_c + jnp.matmul(attn_i,
                               v_new,
                               precision=precision,
                               preferred_element_type=jnp.float32)

        # State update, per-channel:
        #   h_new[k, :] = exp(G_last[k]) * h[k, :]
        #                 + sum_c exp(G_last[k] - G_c[k]) k_c[k] * v_new[c, :]
        g_last = G[:, -1, :]  # [H, K]
        decay_state = jnp.exp(g_last[:, None, :] - G)  # [H, BT, K], <= 1
        k_g = kf * decay_state
        h_new = h * jnp.exp(g_last)[..., None]
        h_new = h_new + jnp.matmul(k_g.swapaxes(-1, -2),
                                   v_new,
                                   precision=precision,
                                   preferred_element_type=jnp.float32)

        return h_new, (o_c, h_new)

    _, (o_chunks, h_chunks) = lax.scan(scan_body, h_init, xs)

    o = o_chunks.transpose(0, 2, 1, 3).reshape(-1, H, V_dim)
    o = o.astype(initial_dtype)
    output = o.reshape(-1, H * V_dim)[padded_indices_valid]

    last_chunk_indices = (new_query_start_loc[1:] // chunk_size) - 1
    final_states = h_chunks[last_chunk_indices]

    num_seqs = last_chunk_indices.shape[0]
    valid_seq_mask = jnp.arange(num_seqs) < distribution[2]
    current_states = recurrent_state[state_indices]
    states_to_set = jnp.where(
        valid_seq_mask[:, None, None, None],
        final_states.astype(recurrent_state.dtype),
        current_states,
    )
    updated_recurrent_state = recurrent_state.at[state_indices].set(
        states_to_set)

    return updated_recurrent_state, output


def recurrent_kda_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-token recurrent KDA update (decode).

    Args:
      query/key/value: ``[B, H, D]`` (q/k already L2-normalized; q scaled).
      g: Per-channel log-decay ``[B, H, K]`` (fp32, post-gate).
      beta: ``[B, H]`` (post-sigmoid).
      state: ``[B, H, K, V]`` or None.

    Returns ``(out [B, H, V], new_state [B, H, K, V])``, fp32 state math.
    """
    B, H, d_k = query.shape
    d_v = value.shape[-1]
    orig_dtype = query.dtype
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    g = g.astype(jnp.float32)
    if state is None:
        state = jnp.zeros((B, H, d_k, d_v), dtype=jnp.float32)
    else:
        state = state.astype(jnp.float32)

    # S <- Diag(exp(g)) S ; S <- S + beta k (v - k.S)^T ; o = S^T q.
    decayed = state * jnp.exp(g)[..., None]
    k_state = jnp.einsum('bhk,bhkv->bhv',
                         key,
                         decayed,
                         precision=jax.lax.Precision.HIGHEST)
    v_new = beta[..., None] * (value - k_state)
    new_state = decayed + key[..., :, None] * v_new[..., None, :]
    out = jnp.einsum('bhk,bhkv->bhv',
                     query,
                     new_state,
                     precision=jax.lax.Precision.HIGHEST)
    return out.astype(orig_dtype), new_state


def ragged_kda_decode_only(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    b_reshaped: jnp.ndarray,
    a_reshaped: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    gate_lower_bound: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """KDA decode-only path (all sequence lengths == 1).

    Mirrors ``ragged_gated_delta_rule_decode_only``; ``a_reshaped`` is
    ``[num_tokens, H, K]`` and the gate form follows ``gate_lower_bound``.
    """
    num_tokens = query.shape[0]
    max_reqs = recurrent_state.shape[0]

    token_idx = jnp.arange(num_tokens)
    req_indices = jnp.clip(token_idx, 0, max_reqs - 1)
    valid_mask = token_idx < distribution[2]

    beta = jax.nn.sigmoid(b_reshaped.astype(jnp.float32))
    g = kda_gate(a_reshaped, A_log, dt_bias, gate_lower_bound)

    query = l2norm(query)
    key = l2norm(key)
    scale = query.shape[-1]**-0.5
    query = query * jnp.asarray(scale, dtype=query.dtype)

    req_state_indices = state_indices[req_indices]
    current_states = recurrent_state[req_state_indices]

    outputs, new_states = recurrent_kda_step(query,
                                             key,
                                             value,
                                             g,
                                             beta,
                                             state=current_states)

    outputs = jnp.where(valid_mask[:, None, None], outputs, 0.0)
    outputs = outputs.reshape(num_tokens, -1)

    states_to_set = jnp.where(valid_mask[:, None, None, None], new_states,
                              current_states)
    updated_recurrent_state = recurrent_state.at[req_state_indices].set(
        states_to_set.astype(recurrent_state.dtype))

    return updated_recurrent_state, outputs
