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
"""Ragged gated delta rule packed JAX implementation."""

import jax
import jax.numpy as jnp
from jax import lax
import tpu_inference.layers.vllm.ops.triangle_solver as triangle_solver


def l2norm(x: jnp.ndarray, dim: int = -1, eps: float = 1e-6) -> jnp.ndarray:
  """Normalizes x along the specified dimension using L2 norm.

  Args:
    x: Input array.
    dim: Dimension along which to normalize.
    eps: Epsilon value to avoid division by zero.

  Returns:
    Normalized array.
  """
  inv_norm = jax.lax.rsqrt(
      (x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype)
  )
  return x * inv_norm


def pack_inputs_single_stream(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    chunk_size: int,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
  """Pads each sequence to multiple of chunk_size and concatenates.

  This function takes ragged sequences and pads each of them so that their
  lengths become a multiple of `chunk_size`. It then concatenates these
  padded sequences into a single continuous stream. This allows for efficient
  chunk-based processing on hardware like TPUs, where fixed-size operations
  are preferred.

  It also computes a `reset_mask` to indicate where a new sequence starts
  (aligned to chunk boundaries), which is used to reset the recurrent state
  during processing.

  Example:
    Original sequences (ragged):
    Seq 1: [A, A, A] (len 3)
    Seq 2: [B, B, B, B, B] (len 5)
    Seq 3: [C, C] (len 2)

    Packed stream (chunk_size=4):
    Chunk 1: [A, A, A, P]  <- Seq 1 padded (New sequence starts)
    Chunk 2: [B, B, B, B]  <- Seq 2 (part 1) (New sequence starts)
    Chunk 3: [B, P, P, P]  <- Seq 2 (part 2) padded
    Chunk 4: [C, C, P, P]  <- Seq 3 padded (New sequence starts)
    (where 'P' denotes padding)

    reset_mask = [True, True, False, True]
    (Indicates whether each chunk starts a new sequence)

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    g: Gate tensor.
    beta: Beta tensor.
    query_start_loc: Start locations of each sequence in original stream.
    chunk_size: Chunk size for padding.

  Returns:
    A tuple containing:
      - packed_query: Packed query tensor.
      - packed_key: Packed key tensor.
      - packed_value: Packed value tensor.
      - packed_g: Packed gate tensor.
      - packed_beta: Packed beta tensor.
      - reset_mask: Mask indicating start of sequences (per chunk).
      - new_query_start_loc: Start locations in packed stream.
      - padded_indices_valid: Indices mapping original to packed.
  """
  num_tokens = query.shape[0]
  num_seqs = len(query_start_loc) - 1

  # Calculate sequence lengths and pad them to multiples of chunk_size.
  seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
  num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
  padded_lengths = num_chunks * chunk_size

  # Compute the start locations for each sequence in the packed stream.
  new_query_start_loc = jnp.cumsum(
      jnp.concatenate([jnp.array([0]), padded_lengths])
  )

  # Map each original token index to its sequence ID in a JIT-friendly way.
  # For each token index, searchsorted finds the insertion point in query_start_loc
  # where the token would go to maintain order (using side="right").
  # Subtracting 1 gives the index of the sequence the token actually belongs to.
  seq_id = (
      jnp.searchsorted(query_start_loc, jnp.arange(num_tokens), side="right")
      - 1
  )
  original_start = query_start_loc[seq_id]
  new_start = new_query_start_loc[seq_id]
  padded_indices_valid = new_start + (jnp.arange(num_tokens) - original_start)

  max_packed_tokens = num_tokens + num_seqs * chunk_size
  max_packed_tokens = (
      (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size
  )

  def pad_and_concatenate(unpadded_data, fill_value=0.0):
    """Pads unpadded_data to max_packed_tokens and places valid data at computed indices."""
    output_shape = (max_packed_tokens,) + unpadded_data.shape[1:]
    packed_data = jnp.full(output_shape, fill_value, dtype=unpadded_data.dtype)
    packed_data = packed_data.at[padded_indices_valid].set(unpadded_data)
    return packed_data

  packed_query = pad_and_concatenate(query)
  packed_key = pad_and_concatenate(key)
  packed_value = pad_and_concatenate(value)
  packed_g = pad_and_concatenate(g)
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


def ragged_gated_delta_rule_mixed_prefill(
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
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Applies chunked gated delta rule for mixed prefill case.

  This function handles the case where sequences can have lengths greater than
  1.
  It pads sequences to multiples of `chunk_size` and processes them in parallel
  within chunks, and sequentially across chunks.

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    b_reshaped: Reshaped b tensor (for beta).
    a_reshaped: Reshaped a tensor (for g).
    A_log: A_log tensor.
    dt_bias: dt_bias tensor.
    query_start_loc: Start locations of sequences in original stream.
    recurrent_state: Recurrent state tensor.
    state_indices: Indices mapping sequences to recurrent state slots.
    chunk_size: Chunk size for padding and processing.
    use_qk_norm_in_gdn: Whether to use QK normalization.
    compute_dtype: Dtype for computation.
    precision: Precision for matrix multiplication.
    preferred_element_type: Preferred element type for matrix multiplication.

  Returns:
    A tuple containing:
      - updated_recurrent_state: Updated recurrent state tensor.
      - output: Output tensor.
  """
  initial_dtype = query.dtype

  beta = jax.nn.sigmoid(b_reshaped)
  g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
      a_reshaped.astype(jnp.float32) + dt_bias.astype(jnp.float32)
  )

  # Pack inputs
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
      query,
      key,
      value,
      g,
      beta,
      query_start_loc,
      chunk_size,
  )

  if use_qk_norm_in_gdn:
    packed_query = l2norm(packed_query, dim=-1, eps=1e-6)
    packed_key = l2norm(packed_key, dim=-1, eps=1e-6)

  packed_g = packed_g.astype(jnp.float32)

  packed_query = packed_query.astype(compute_dtype)
  packed_key = packed_key.astype(compute_dtype)
  packed_value = packed_value.astype(compute_dtype)
  packed_beta = packed_beta.astype(compute_dtype)

  scale = jax.lax.rsqrt(
      jnp.array(packed_query.shape[-1], dtype=jnp.float32)
  ).astype(compute_dtype)
  packed_query = packed_query * scale

  total_tokens = packed_query.shape[0]
  num_chunks = total_tokens // chunk_size
  H = packed_query.shape[1]
  K_dim = packed_query.shape[2]
  V_dim = packed_value.shape[2]

  def to_chunk(x):
    return x.reshape(num_chunks, chunk_size, H, -1).transpose(0, 2, 1, 3)

  def to_chunk_scalar(x):
    return x.reshape(num_chunks, chunk_size, H).transpose(0, 2, 1)

  q_c = to_chunk(packed_query)
  k_c = to_chunk(packed_key)
  v_c = to_chunk(packed_value)
  g_c = to_chunk_scalar(packed_g)
  beta_c = to_chunk_scalar(packed_beta)

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

  A = triangle_solver.decompose_triangular_matrix_inverse_pallas(identity + S)

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
  w_scan = w_chunks
  u_scan = u_chunks
  q_g_scan = q_g_chunks
  attn_i_scan = attn_i_chunks
  g_i_last_exp_scan = g_i_last_exp_chunks
  k_i_g_diff_scan = k_i_g_diff_chunks

  # Prepare init_h_per_chunk
  init_h_per_chunk = jnp.zeros(
      (num_chunks, H, K_dim, V_dim), dtype=recurrent_state.dtype
  )
  start_chunk_indices = new_query_start_loc[:-1] // chunk_size
  init_states_for_seqs = recurrent_state[state_indices]
  init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(
      init_states_for_seqs
  )

  h_init = jnp.zeros((H, K_dim, V_dim), dtype=jnp.float32)

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
  o = o_chunks.transpose(0, 2, 1, 3)
  o = o.reshape(-1, H, V_dim)

  o = o.astype(initial_dtype)

  # Unpack output
  packed_output_flat = o.reshape(-1, H * V_dim)
  output = packed_output_flat[padded_indices_valid]

  # Update recurrent state
  last_chunk_indices = (new_query_start_loc[1:] // chunk_size) - 1
  final_states = h_chunks[last_chunk_indices]
  updated_recurrent_state = recurrent_state.at[state_indices].set(
      final_states.astype(recurrent_state.dtype)
  )

  return updated_recurrent_state, output


def recurrent_gated_delta_rule_step(
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


def ragged_gated_delta_rule_decode_only(
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
    use_qk_norm_in_gdn: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Applies gated delta rule for decode-only case (sequence lengths = 1).

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    b_reshaped: Reshaped b tensor (for beta).
    a_reshaped: Reshaped a tensor (for g).
    recurrent_state: Recurrent state tensor.
    A_log: A_log tensor.
    dt_bias: dt_bias tensor.
    query_start_loc: Start locations of sequences.
    state_indices: Indices mapping sequences to recurrent state slots.
    use_qk_norm_in_gdn: Whether to use QK normalization.

  Returns:
    A tuple containing:
      - updated_recurrent_state: Updated recurrent state tensor.
      - output: Output tensor.
  """
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

  req_state_indices = state_indices[req_indices]
  curr_states = recurrent_state[req_state_indices]

  def single_token_fn(curr_q, curr_k, curr_v, curr_g, curr_beta, state):
    curr_q = curr_q[None, :, None, :]
    curr_k = curr_k[None, :, None, :]
    curr_v = curr_v[None, :, None, :]
    curr_g = curr_g[None, :, None]
    curr_beta = curr_beta[None, :, None]
    state = state[None, ...]

    output, new_state = recurrent_gated_delta_rule_step(
        curr_q, curr_k, curr_v, curr_g, curr_beta, state=state
    )
    return output[0, :, 0, :], new_state[0]

  vmapped_fn = jax.vmap(single_token_fn)
  outputs, new_states = vmapped_fn(query, key, value, g, beta, curr_states)

  outputs = outputs.reshape(num_tokens, -1)

  dummy_idx = max_reqs
  recurrent_state_padded = jnp.pad(
      recurrent_state,
      ((0, 1), (0, 0), (0, 0), (0, 0)),
      mode="constant",
      constant_values=0,
  )

  safe_indices = jnp.where(valid_mask, req_state_indices, dummy_idx)
  updated_recurrent_state_padded = recurrent_state_padded.at[
      safe_indices
  ].set(new_states.astype(recurrent_state.dtype))

  new_recurrent_state = updated_recurrent_state_padded[:max_reqs]

  return new_recurrent_state, outputs


def ragged_gated_delta_rule(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Applies the gated delta rule over ragged seq lengths

  This function separates mixed QKV, handles repeating for multi-query attention
  if needed, and routes to either the decode-only or mixed-prefill branch
  depending on sequence lengths.

  Args:
    mixed_qkv: Mixed query, key, value tensor.
    b: b tensor (for beta).
    a: a tensor (for g).
    recurrent_state: Recurrent state tensor.
    A_log: A_log tensor.
    dt_bias: dt_bias tensor.
    query_start_loc: Start locations of sequences.
    state_indices: Indices mapping sequences to recurrent state slots.
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key/query dimension.
    d_v: Value dimension.
    chunk_size: Chunk size for padding in mixed prefill.
    use_qk_norm_in_gdn: Whether to use QK normalization.

  Returns:
    A tuple containing:
      - updated_recurrent_state: Updated recurrent state tensor.
      - output: Output tensor.
  """
  num_tokens = mixed_qkv.shape[0]
  key_dim = n_kq * d_k
  query = mixed_qkv[..., :key_dim]
  key = mixed_qkv[..., key_dim : key_dim * 2]
  value = mixed_qkv[..., key_dim * 2 :]

  q_reshaped = query.reshape(num_tokens, n_kq, d_k)
  k_reshaped = key.reshape(num_tokens, n_kq, d_k)
  v_reshaped = value.reshape(num_tokens, n_v, d_v)

  repeat_factor = n_v // n_kq
  if repeat_factor > 1:
    q_reshaped = jnp.repeat(q_reshaped, repeat_factor, axis=1)
    k_reshaped = jnp.repeat(k_reshaped, repeat_factor, axis=1)
  b_reshaped = b.reshape(num_tokens, n_v)
  a_reshaped = a.reshape(num_tokens, n_v)

  def decode_only_branch(_):
    new_state, output = ragged_gated_delta_rule_decode_only(
        query=q_reshaped,
        key=k_reshaped,
        value=v_reshaped,
        b_reshaped=b_reshaped,
        a_reshaped=a_reshaped,
        recurrent_state=recurrent_state,
        A_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
    )
    return new_state, output.astype(mixed_qkv.dtype)

  def mixed_prefill_branch(_):
    return ragged_gated_delta_rule_mixed_prefill(
        query=q_reshaped,
        key=k_reshaped,
        value=v_reshaped,
        b_reshaped=b_reshaped,
        a_reshaped=a_reshaped,
        A_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
    )

  # Decode-only is that all sequence lengths are 1
  sequence_lengths = query_start_loc[1:] - query_start_loc[:-1]
  is_all_seqlen_1 = jnp.all(sequence_lengths <= 1)

  return jax.lax.cond(
      is_all_seqlen_1, decode_only_branch, mixed_prefill_branch, operand=None
  )

