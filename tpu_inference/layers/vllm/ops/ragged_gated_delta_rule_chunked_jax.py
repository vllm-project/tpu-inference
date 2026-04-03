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
"""Ragged gated delta rule chunked JAX implementation."""

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
from tpu_inference.layers.vllm.ops.gated_deltanet import jax_chunk_gated_delta_rule_pure_jax


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  inv_norm = jax.lax.rsqrt(
      (x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype)
  )
  return x * inv_norm


def pack_inputs_single_stream(
    query, key, value, g, beta, query_start_loc, chunk_size, max_seqlen
):
  num_tokens = query.shape[0]
  num_seqs = len(query_start_loc) - 1

  seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
  num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
  padded_lengths = num_chunks * chunk_size
  new_query_start_loc = jnp.cumsum(
      jnp.concatenate([jnp.array([0]), padded_lengths])
  )

  # JIT-friendly mapping from original token index to packed token index
  seq_id = (
      jnp.searchsorted(query_start_loc, jnp.arange(num_tokens), side='right')
      - 1
  )
  original_start = query_start_loc[seq_id]
  new_start = new_query_start_loc[seq_id]
  padded_indices_valid = new_start + (jnp.arange(num_tokens) - original_start)

  max_packed_tokens = num_tokens + num_seqs * chunk_size
  max_packed_tokens = (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size

  def pad_and_concatenate(unpadded_data, fill_value=0.0):
    output_shape = (max_packed_tokens,) + unpadded_data.shape[1:]
    packed_data = jnp.full(output_shape, fill_value, dtype=unpadded_data.dtype)
    packed_data = packed_data.at[padded_indices_valid].set(unpadded_data)
    return packed_data

  packed_query = pad_and_concatenate(query)
  packed_key = pad_and_concatenate(key)
  packed_value = pad_and_concatenate(value)
  packed_g = pad_and_concatenate(g, fill_value=0.0)
  packed_beta = pad_and_concatenate(beta)

  num_chunks_total = max_packed_tokens // chunk_size
  reset_mask = jnp.zeros((num_chunks_total,), dtype=bool)
  start_chunk_indices = new_query_start_loc[:-1] // chunk_size
  reset_mask = reset_mask.at[start_chunk_indices].set(True)

  return (
      packed_query,
      packed_key,
      packed_value,
      packed_g,
      packed_beta,
      reset_mask,
      new_query_start_loc,
      padded_indices_valid,
  )


def jax_chunk_gated_delta_rule_packed_pure_jax(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    reset_mask: Array,
    init_h_per_chunk: Array,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> tuple[Array, Array]:
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(
      compute_dtype
  )
  query = query * scale

  total_tokens = query.shape[0]
  num_chunks = total_tokens // chunk_size
  H = query.shape[1]
  K_dim = query.shape[2]
  V_dim = value.shape[2]

  def to_chunk(x):
    return x.reshape(num_chunks, chunk_size, H, -1).transpose(0, 2, 1, 3)

  def to_chunk_scalar(x):
    return x.reshape(num_chunks, chunk_size, H).transpose(0, 2, 1)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  S = jnp.matmul(
      k_beta,
      k_c.swapaxes(-1, -2),
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
  S = S.astype(jnp.float32)

  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  identity = jnp.eye(chunk_size, dtype=jnp.float32)
  identity_broadcasted = jnp.broadcast_to(identity, S.shape)

  A = jax.scipy.linalg.solve_triangular(
      identity + S, identity_broadcasted, lower=True, unit_diagonal=True
  )

  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(
      A,
      v_beta.astype(jnp.float32),
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.matmul(
      A,
      k_beta_g,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
  w_chunks = w_chunks.astype(compute_dtype)

  attn_chunks = jnp.matmul(
      q_c,
      k_c.swapaxes(-1, -2),
      precision=precision,
      preferred_element_type=preferred_element_type,
  ).astype(jnp.float32)
  g_diff_chunks = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
  g_diff_chunks = jnp.where(mask_intra, g_diff_chunks, -1e30)
  attn_i_chunks = jnp.where(
      mask_intra, attn_chunks * jnp.exp(g_diff_chunks), 0.0
  ).astype(compute_dtype)

  q_g_chunks = (q_c.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]).astype(
      compute_dtype
  )
  g_i_last_exp_chunks = jnp.exp(g_cumsum[..., -1, None, None])
  g_diff_exp_state_chunks = jnp.exp(g_cumsum[..., -1, None] - g_cumsum)[
      ..., None
  ]
  k_i_g_diff_chunks = (
      k_c.astype(jnp.float32) * g_diff_exp_state_chunks
  ).astype(compute_dtype)

  # STAGE 3: INTER-CHUNK RECURRENCE
  # For single stream, total_chunks is the first dimension.
  # Shapes are (total_chunks, H, chunk_size, ...)
  w_scan = w_chunks
  u_scan = u_chunks
  q_g_scan = q_g_chunks
  attn_i_scan = attn_i_chunks
  g_i_last_exp_scan = g_i_last_exp_chunks
  k_i_g_diff_scan = k_i_g_diff_chunks

  h_init = jnp.zeros((H, K_dim, V_dim), dtype=jnp.float32)
  # Cast h_init to be varying along the manual 'model' axis to match output carry sharding
  try:
    h_init = jax.lax.pcast(h_init, ('model',), to='varying')
  except (ValueError, TypeError, NameError):
    # Fallback if not in a parallel context with 'model' axis
    pass

  xs = (
      w_scan,
      u_scan,
      q_g_scan,
      attn_i_scan,
      g_i_last_exp_scan,
      k_i_g_diff_scan,
      reset_mask,
      init_h_per_chunk,
  )

  def scan_body(h, args):
    w, u, q_g, attn_i, g_i_last_exp, k_i_g_diff, reset, init_h = args

    # Reset state if needed
    h = jnp.where(reset, init_h, h)

    attn_inter = jnp.matmul(
        q_g,
        h,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    v_prime = jnp.matmul(
        w.astype(jnp.float32),
        h,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    v_new = u.astype(jnp.float32) - v_prime

    term2 = jnp.matmul(
        attn_i,
        v_new,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    o_c = attn_inter + term2

    h_new = h * g_i_last_exp
    update_term = jnp.matmul(
        k_i_g_diff.swapaxes(-1, -2),
        v_new,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    h_new = h_new + update_term

    return h_new, (o_c, h_new)

  _, (o_chunks, h_chunks) = lax.scan(scan_body, h_init, xs)

  # STAGE 4: FINALIZATION
  # o_chunks shape: (total_chunks, H, chunk_size, V_dim)
  o = o_chunks.transpose(0, 2, 1, 3)  # (total_chunks, chunk_size, H, V_dim)
  o = o.reshape(-1, H, V_dim)

  o = o.astype(initial_dtype)

  return o, h_chunks


def _recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Single-step recurrent update for decode."""
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

  k_state = jnp.einsum("bhd, bhdm -> bhm", k, state)
  v_diff = v - jnp.exp(g_val)[..., None] * k_state

  v_new = beta_val[..., None] * v_diff

  q_state = jnp.einsum("bhd, bhdm -> bhm", q, state)
  q_k = jnp.sum(q * k, axis=-1, keepdims=True)

  out = jnp.exp(g_val)[..., None] * q_state + q_k * v_new

  k_v_new = jnp.einsum("bhd, bhm -> bhdm", k, v_new)
  new_state = state * jnp.exp(g_val)[..., None, None] + k_v_new

  return out[:, :, None, :], new_state


def _ragged_gated_delta_rule_token_by_token(
    query,
    key,
    value,
    b_reshaped,
    a_reshaped,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    use_qk_norm_in_gdn,
):
  num_tokens = query.shape[0]
  max_reqs = recurrent_state.shape[0]
  n_v = value.shape[1]
  d_k = query.shape[2]
  d_v = value.shape[2]

  token_idx = jnp.arange(num_tokens)
  req_indices = (
      jnp.sum(token_idx[:, None] >= query_start_loc[None, :], axis=1) - 1
  )
  req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
  valid_mask = token_idx < query_start_loc[-1]

  beta = jax.nn.sigmoid(b_reshaped)
  g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
      a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)
  )

  if use_qk_norm_in_gdn:
    query = l2norm(query)
    key = l2norm(key)

  def scan_fn(carry, xs):
    recurrent_state_all = carry
    (curr_q, curr_k, curr_v, curr_g, curr_beta, req_idx, is_valid) = xs

    curr_q = curr_q[None, :, None, :]
    curr_k = curr_k[None, :, None, :]
    curr_v = curr_v[None, :, None, :]
    curr_g = curr_g[None, :, None]
    curr_beta = curr_beta[None, :, None]

    state_idx = state_indices[req_idx]
    state = recurrent_state_all[state_idx][None, ...]

    output, new_state = _recurrent_gated_delta_rule_step(
        curr_q, curr_k, curr_v, curr_g, curr_beta, state=state
    )

    output = output.reshape(1, -1) # (1, H * d_v)

    recurrent_state_all = jnp.where(
        is_valid,
        recurrent_state_all.at[state_idx].set(new_state[0]),
        recurrent_state_all,
    )

    return recurrent_state_all, output[0]

  carry_init = recurrent_state
  xs = (query, key, value, g, beta, req_indices, valid_mask)

  new_recurrent_state, output = jax.lax.scan(scan_fn, carry_init, xs)
  return new_recurrent_state, output


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
    chunk_size: int = 64,
    max_seqlen: int | None = None,
    use_qk_norm_in_gdn: bool = True,
):
  num_tokens = mixed_qkv.shape[0]
  key_dim = n_kq * d_k
  query = mixed_qkv[..., :key_dim]
  key = mixed_qkv[..., key_dim : key_dim * 2]
  value = mixed_qkv[..., key_dim * 2 :]
  num_seqs = len(query_start_loc) - 1

  if max_seqlen is None:
    sequence_lengths = query_start_loc[1:] - query_start_loc[:-1]
    max_seqlen = int(jnp.max(sequence_lengths))

  q_reshaped = query.reshape(num_tokens, n_kq, d_k)
  k_reshaped = key.reshape(num_tokens, n_kq, d_k)
  v_reshaped = value.reshape(num_tokens, n_v, d_v)

  repeat_factor = n_v // n_kq
  if repeat_factor > 1:
    q_reshaped = jnp.repeat(q_reshaped, repeat_factor, axis=1)
    k_reshaped = jnp.repeat(k_reshaped, repeat_factor, axis=1)
  b_reshaped = b.reshape(num_tokens, n_v)
  a_reshaped = a.reshape(num_tokens, n_v)

  beta_unpadded = jax.nn.sigmoid(b_reshaped)
  g_unpadded = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
      a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)
  )

  # Pack inputs
  def token_by_token_branch(_):
    new_state, output = _ragged_gated_delta_rule_token_by_token(
        q_reshaped,
        k_reshaped,
        v_reshaped,
        b_reshaped,
        a_reshaped,
        recurrent_state,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        use_qk_norm_in_gdn,
    )
    return new_state, output.astype(mixed_qkv.dtype)

  def chunked_branch(_):
    (
        packed_query,
        packed_key,
        packed_value,
        packed_g,
        packed_beta,
        reset_mask,
        new_query_start_loc,
        padded_indices_valid,
    ) = pack_inputs_single_stream(
        q_reshaped,
        k_reshaped,
        v_reshaped,
        g_unpadded,
        beta_unpadded,
        query_start_loc,
        chunk_size,
        max_seqlen,
    )

    num_chunks_total = packed_query.shape[0] // chunk_size
    init_h_per_chunk = jnp.zeros(
        (num_chunks_total, n_v, d_k, d_v), dtype=recurrent_state.dtype
    )
    start_chunk_indices = new_query_start_loc[:-1] // chunk_size
    init_states_for_seqs = recurrent_state[state_indices]
    init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(
        init_states_for_seqs
    )

    packed_output, h_chunks = jax_chunk_gated_delta_rule_packed_pure_jax(
        query=packed_query,
        key=packed_key,
        value=packed_value,
        g=packed_g,
        beta=packed_beta,
        reset_mask=reset_mask,
        init_h_per_chunk=init_h_per_chunk,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
    )

    packed_output_flat = packed_output.reshape(-1, n_v * d_v)
    output = packed_output_flat[padded_indices_valid]

    last_chunk_indices = (new_query_start_loc[1:] // chunk_size) - 1
    final_states = h_chunks[last_chunk_indices]

    updated_recurrent_state = recurrent_state.at[state_indices].set(final_states)

    return updated_recurrent_state, output

  sequence_lengths = query_start_loc[1:] - query_start_loc[:-1]
  is_all_seqlen_1 = jnp.all(sequence_lengths <= 1)

  return jax.lax.cond(
      is_all_seqlen_1, token_by_token_branch, chunked_branch, operand=None
  )

