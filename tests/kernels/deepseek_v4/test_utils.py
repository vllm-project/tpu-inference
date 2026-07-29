import jax
import jax.numpy as jnp

def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(x, a):
  return cdiv(x, a) * a


def get_dtype_packing(dtype):
  bits = jax.dtypes.itemsize_bits(dtype)
  return 32 // bits

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


@jax.jit(donate_argnames="cache_kv")
def update_kv_cache(
    new_kv: jax.Array,  # [num_tokens, actual_lkv_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
) -> tuple[jax.Array, jax.Array]:
  """Update KV cache with new tokens."""
  actual_lkv_dim = new_kv.shape[-1]
  lkv_dim = align_to(actual_lkv_dim, 128)
  if actual_lkv_dim != lkv_dim:
    new_kv = jnp.pad(
        new_kv, ((0, 0), (0, lkv_dim - actual_lkv_dim)), constant_values=0
    )
  kv_dim = lkv_dim
  _, page_size_per_kv_packing, kv_packing, cache_kv_dim = cache_kv.shape
  assert kv_dim == cache_kv_dim
  page_size = page_size_per_kv_packing * kv_packing

  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  pages_per_seq = num_page_indices // max_num_seqs

  def seq_loop_body(i, cache_kv):
    q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]

    def token_loop_body(j, cache_kv_):
      token_idx_in_seq = kv_len - q_len + j
      page_num_in_seq = token_idx_in_seq // page_size
      page_indices_start = i * pages_per_seq
      page_idx = page_indices[page_indices_start + page_num_in_seq]
      row = (token_idx_in_seq % page_size) // kv_packing
      col = (token_idx_in_seq % page_size) % kv_packing

      cache_kv_ = cache_kv_.at[page_idx, row, col, ..., :lkv_dim].set(
          new_kv[q_start + j]
      )
      return cache_kv_

    return jax.lax.fori_loop(0, q_len, token_loop_body, cache_kv)

  cache_kv = jax.lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

  return cache_kv


def ref_mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    new_kv: jax.Array,  # [num_tokens, actual_lkv_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):

  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  updated_cache_kv = update_kv_cache(
      new_kv,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )
  # Pad ql_nope and q_pe to make the last dimension 128-byte aligned.
  actual_lkv_dim = ql_nope.shape[-1]
  lkv_dim = align_to(actual_lkv_dim, 128)
  if lkv_dim != actual_lkv_dim:
    ql_nope = jnp.pad(
        ql_nope,
        ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
        constant_values=0,
    )

  q = ql_nope
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs

  total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
      updated_cache_kv.shape
  )
  page_size = page_size_per_kv_packing * kv_packing
  assert lkv_dim == ql_nope.shape[-1]

  kv_c_cache = updated_cache_kv[..., :lkv_dim].reshape(
      total_num_pages, page_size, lkv_dim
  )

  outputs = []
  ls = []
  ms = []

  for i in range(distribution[-1]):
    q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]

    q_i = q[q_start:q_end]  # [q_len, actual_num_q_heads, lkv_dim+r_dim]

    indices_start = i * pages_per_seq
    num_pages_i = cdiv(kv_len, page_size)
    indices_end = indices_start + num_pages_i
    indices = page_indices[indices_start:indices_end]

    # Gather paged kv_c and k_pe
    gathered_kv_c = kv_c_cache[indices]  # [num_pages_i, page_size, lkv_dim]

    # Flatten pages to sequence
    flat_kv_c = gathered_kv_c.reshape(
        -1, lkv_dim
    )  # [num_pages_i * page_size, lkv_dim]

    # Prepare k and v for attention
    k_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]
    v_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]

    # MQA attention:
    # q:[q_len, actual_num_q_heads, lkv_dim+r_dim]
    # k:[kv_len, lkv_dim+r_dim]
    # v:[kv_len, lkv_dim]
    # attn: [actual_num_q_heads, q_len, kv_len]
    attn = jnp.einsum(
        "qnh,kh->nqk", q_i, k_i, preferred_element_type=jnp.float32
    )
    attn *= sm_scale

    # Causal mask
    q_span = kv_len - q_len + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    attn = jnp.where(mask, mask_value, attn)
    m = jnp.max(attn, axis=-1, keepdims=True)
    l = jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
    attn = jax.nn.softmax(attn, axis=-1).astype(v_i.dtype)

    # out_i: [q_len, actual_num_q_heads, lkv_dim]
    out_i = jnp.einsum("nqk,kl->qnl", attn, v_i).astype(q_i.dtype)
    outputs.append(out_i)
    ls.append(jnp.transpose(l[..., 0]))
    ms.append(jnp.transpose(m[..., 0]))

  return (
      jnp.concatenate(outputs, axis=0),
      updated_cache_kv,
      jnp.concatenate(ls, axis=0),
      jnp.concatenate(ms, axis=0),
  )