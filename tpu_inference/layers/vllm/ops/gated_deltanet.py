import functools

import jax
from jax import Array
from jax import lax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import tpu_inference.layers.vllm.ops.triangle_solver as triangle_solver


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  """L2 normalization function. Normalizes a vector to have a length of 1.

  Args:
    x: Input array.
    dim: The axis or axes along which to normalize. Defaults to the last axis.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array with the same shape as x.
  """

  inv_norm = jax.lax.rsqrt(
      (x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype)
  )
  return x * inv_norm


def jax_chunk_gated_delta_rule_pure_jax(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    preferred_element_type: jnp.dtype = jnp.float32,
    cu_seqlens: Array | None = None,
    max_seqlen: int | None = None,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  # 2. Cast inputs to the requested compute_dtype (cfg.dtype) to save memory/compute
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Scale Query (keep in compute_dtype)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(
      compute_dtype
  )
  query = query * scale

  if cu_seqlens is not None:
    if max_seqlen is None:
      raise ValueError(
          "max_seqlen must be provided when cu_seqlens is provided"
      )

    N = cu_seqlens.shape[0] - 1
    indices = jnp.arange(max_seqlen)

    def unpack(x):
      padded_x = jnp.pad(x, ((0, max_seqlen),) + ((0, 0),) * (x.ndim - 1))

      def get_seq(i):
        start = cu_seqlens[i]
        length = cu_seqlens[i + 1] - start
        seq = jax.lax.dynamic_slice_in_dim(padded_x, start, max_seqlen, axis=0)
        mask = indices < length
        return seq * mask[(...,) + (None,) * (x.ndim - 1)]

      return jax.vmap(get_seq)(jnp.arange(N))

    query = unpack(query)
    key = unpack(key)
    value = unpack(value)
    g = unpack(g)
    beta = unpack(beta)

  B, seq_len, H, K_dim = key.shape
  V_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(
          x,
          ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2),
          constant_values=val,
      )

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  # Helper: (B, S, H, D) -> (B, N, H, C, D)
  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

  # Helper for scalars: (B, S, H) -> (B, N, H, C)
  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION (Parallel)
  # =========================================================================

  # Cumulative decay (Must be float32)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  # S Matrix Calculation
  S = jnp.matmul(
      k_beta,
      k_c.swapaxes(-1, -2),
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
  S = S.astype(jnp.float32)

  # Apply mask BEFORE exp to prevent 'inf' gradients
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  # Inversion (A) - Strictly float32
  identity = jnp.eye(chunk_size, dtype=jnp.float32)
  identity_broadcasted = jnp.broadcast_to(identity, S.shape)

  A = jax.scipy.linalg.solve_triangular(
      identity + S, identity_broadcasted, lower=True, unit_diagonal=True
  )

  # 5. WY Factors
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

  # Different from Qwen3 implementation to achieve better parallelism
  # 1. Intra-chunk attention matrix
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

  # 2. Query and Key decay factors
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

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Scan)
  # =========================================================================
  scan_perm_vec = (1, 0, 2, 3, 4)

  w_scan = w_chunks.transpose(scan_perm_vec)
  u_scan = u_chunks.transpose(scan_perm_vec)
  q_g_scan = q_g_chunks.transpose(scan_perm_vec)
  attn_i_scan = attn_i_chunks.transpose(scan_perm_vec)
  g_i_last_exp_scan = g_i_last_exp_chunks.transpose(scan_perm_vec)
  k_i_g_diff_scan = k_i_g_diff_chunks.transpose(scan_perm_vec)

  if initial_state is None:
    h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    h_init = initial_state.astype(jnp.float32)

  xs = (
      w_scan,
      u_scan,
      q_g_scan,
      attn_i_scan,
      g_i_last_exp_scan,
      k_i_g_diff_scan,
  )

  def scan_body(h, args):
    w, u, q_g, attn_i, g_i_last_exp, k_i_g_diff = args

    # 1. Inter-chunk output
    attn_inter = jnp.matmul(
        q_g,
        h,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # 2. Delta Rule Subtraction
    v_prime = jnp.matmul(
        w.astype(jnp.float32),
        h,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Add Intra-chunk output
    term2 = jnp.matmul(
        attn_i,
        v_new,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    o_c = attn_inter + term2

    # 4. State Update
    h_new = h * g_i_last_exp
    update_term = jnp.matmul(
        k_i_g_diff.swapaxes(-1, -2),
        v_new,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    h_new = h_new + update_term

    return h_new, o_c

  final_h, o_chunks = lax.scan(scan_body, h_init, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  o = o_chunks.transpose(1, 0, 3, 2, 4)
  o = o.reshape(B, -1, H, V_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, (final_h if initial_state is not None else None)


def inter_chunk_recurrence_kernel(
    w_ref,
    u_ref,
    q_ref,
    k_ref,
    g_ref,
    h_init_ref,
    o_ref,
    h_final_ref,
    *,
    chunk_size,
    num_chunks,
    precision,
    preferred_element_type,
):
  b = pl.program_id(0)
  h_id = pl.program_id(1)

  # Initial state
  h_vmem = h_init_ref[(0, 0, pl.dslice(None), pl.dslice(None))]

  def loop_body(c, h_vmem):
    w = w_ref[(0, c, 0, pl.dslice(None), pl.dslice(None))]
    u = u_ref[(0, c, 0, pl.dslice(None), pl.dslice(None))]
    q = q_ref[(0, c, 0, pl.dslice(None), pl.dslice(None))]
    k = k_ref[(0, c, 0, pl.dslice(None), pl.dslice(None))]
    g = g_ref[(0, c, 0, pl.dslice(None), 0)]

    # 1. Inter-chunk
    q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
    attn_inter = jnp.matmul(
        q_g,
        h_vmem,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # 2. Delta Rule Subtraction
    v_prime = jnp.matmul(
        w.astype(jnp.float32),
        h_vmem,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Intra-chunk
    attn = jnp.matmul(
        q,
        k.swapaxes(-1, -2),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    attn = attn.astype(jnp.float32)

    g_diff = g[..., :, None] - g[..., None, :]

    idx = jnp.arange(chunk_size)
    mask = idx[:, None] >= idx[None, :]

    g_diff = jnp.where(mask, g_diff, -10000.0)

    attn_i = attn * jnp.exp(g_diff)
    attn_i = jnp.where(mask, attn_i, 0.0)

    # 4. Combine Core Output
    term2 = jnp.matmul(
        attn_i,
        v_new,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    o_c = attn_inter + term2

    o_ref[(0, c, 0, pl.dslice(None), pl.dslice(None))] = o_c.astype(o_ref.dtype)

    # 5. State Update
    g_i_last_exp = jnp.exp(g[-1])
    h_vmem = h_vmem * g_i_last_exp

    g_diff_exp_state = jnp.exp(g[-1] - g)[..., None]
    k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state
    update_term = jnp.matmul(
        k_i_g_diff.swapaxes(-1, -2),
        v_new,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    h_vmem = h_vmem + update_term
    return h_vmem

  h_vmem = jax.lax.fori_loop(0, num_chunks, loop_body, h_vmem)

  h_final_ref[(0, 0, pl.dslice(None), pl.dslice(None))] = h_vmem


def jax_chunk_gated_delta_rule_with_kernel(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule using Pallas."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  # Cast inputs
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Scale Query
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(
      compute_dtype
  )
  query = query * scale

  B, seq_len, H, K_dim = key.shape
  V_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(
          x,
          ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2),
          constant_values=val,
      )

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION
  # =========================================================================
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

  # A = jax.scipy.linalg.solve_triangular(
  #     identity + S, identity_broadcasted, lower=True, unit_diagonal=True
  # )
  A = triangle_solver.lower_triangle_solver_pallas(
      identity + S, identity_broadcasted
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

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE WITH PALLAS
  # =========================================================================
  if initial_state is None:
    initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    initial_state = initial_state.astype(jnp.float32)

  # Prepare specs for Pallas
  grid = (B, H)
  o_chunks_shape = (B, num_chunks, H, chunk_size, V_dim)
  out_shapes = [
      jax.ShapeDtypeStruct(o_chunks_shape, compute_dtype),
      jax.ShapeDtypeStruct(initial_state.shape, jnp.float32),
  ]

  in_specs = [
      pl.BlockSpec(
          (1, num_chunks, 1, chunk_size, K_dim), lambda b, h: (b, 0, h, 0, 0)
      ),  # w
      pl.BlockSpec(
          (1, num_chunks, 1, chunk_size, V_dim), lambda b, h: (b, 0, h, 0, 0)
      ),  # u
      pl.BlockSpec(
          (1, num_chunks, 1, chunk_size, K_dim), lambda b, h: (b, 0, h, 0, 0)
      ),  # q
      pl.BlockSpec(
          (1, num_chunks, 1, chunk_size, K_dim), lambda b, h: (b, 0, h, 0, 0)
      ),  # k
      pl.BlockSpec(
          (1, num_chunks, 1, chunk_size, 1), lambda b, h: (b, 0, h, 0, 0)
      ),  # g
      pl.BlockSpec((1, 1, K_dim, V_dim), lambda b, h: (b, h, 0, 0)),  # h_init
  ]

  out_specs = [
      pl.BlockSpec(
          (1, num_chunks, 1, chunk_size, V_dim), lambda b, h: (b, 0, h, 0, 0)
      ),  # o_c
      pl.BlockSpec((1, 1, K_dim, V_dim), lambda b, h: (b, h, 0, 0)),  # final_h
  ]

  o_chunks, final_h = pl.pallas_call(
      functools.partial(
          inter_chunk_recurrence_kernel,
          chunk_size=chunk_size,
          num_chunks=num_chunks,
          precision=precision,
          preferred_element_type=preferred_element_type,
      ),
      out_shape=out_shapes,
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
      name="gdn_recurrence_kernel",
  )(w_chunks, u_chunks, q_c, k_c, jnp.expand_dims(g_cumsum, -1), initial_state)

  # o_chunks is (B, num_chunks, H, chunk_size, V_dim)
  o = o_chunks.transpose(0, 1, 3, 2, 4).reshape(B, -1, H, V_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, final_h

