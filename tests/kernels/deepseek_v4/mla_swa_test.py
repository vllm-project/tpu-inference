"""Correctness test for MLA kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

# Import the kernels
from tpu_inference.kernels.experimental.deepseek_v4 import mla_swa


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = jax.dtypes.itemsize_bits(dtype)
    return 32 // bits


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


@jax.jit(donate_argnames="cache_kv")
def update_kv_cache(
        new_kv: jax.Array,  # [num_tokens, actual_lkv_dim]
        cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        distribution: jax.Array,  # i32[3]
) -> tuple[jax.Array, jax.Array]:
    """Update KV cache with new tokens."""
    actual_lkv_dim = new_kv.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if actual_lkv_dim != lkv_dim:
        new_kv = jnp.pad(new_kv, ((0, 0), (0, lkv_dim - actual_lkv_dim)),
                         constant_values=0)
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

            cache_kv_ = cache_kv_.at[page_idx, row, col,
                                     ..., :lkv_dim].set(new_kv[q_start + j])
            return cache_kv_

        return jax.lax.fori_loop(0, q_len, token_loop_body, cache_kv)

    cache_kv = jax.lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

    return cache_kv


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def quantize_and_dequantize_ref_cache(kv_c_cache):
  total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim = (
      kv_c_cache.shape
  )
  page_size = page_size_per_kv_packing * kv_packing
  kv_c_cache = kv_c_cache.reshape(total_num_pages, page_size, lkv_dim)

  fp8_part = kv_c_cache[..., :448]
  bf16_part = kv_c_cache[..., 448:512]

  fp8_blocked = fp8_part.reshape(total_num_pages, page_size, 7, 64)
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  x_amax = jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True)
  x_amax = jnp.clip(x_amax, 1e-4, None)
  sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))

  fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
  scales_quant = sf.reshape(total_num_pages, page_size, 7).astype(
      jnp.float8_e8m0fnu
  )
  print("gxd XXXXXX")

  fp8_dequant = (
      fp8_quant.astype(jnp.bfloat16)
      * scales_quant[..., None].astype(jnp.bfloat16)
  ).reshape(total_num_pages, page_size, 448)
  kv_c_cache = jnp.concatenate([fp8_dequant, bf16_part], axis=-1)

  return kv_c_cache.reshape(
      total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim
  )


def dequantize_swa_cache(swc_cache):
  total_num_pages = swc_cache.shape[0]
  # quantized has shape [total_num_pages, page_size_per_kv_packing, 4, 640]
  page_size_per_kv_packing = swc_cache.shape[1]
  quantized_flat = swc_cache.reshape(
      total_num_pages * page_size_per_kv_packing * 4, 640
  )

  fp8_uint8 = quantized_flat[..., :448]
  bf16_uint8 = quantized_flat[..., 448:576]
  scales_uint8 = quantized_flat[..., 576:583]

  fp8_val = jax.lax.bitcast_convert_type(fp8_uint8, jnp.float8_e4m3fn).astype(
      jnp.bfloat16
  )
  bf16_uint8_reshaped = bf16_uint8.reshape(quantized_flat.shape[:-1] + (64, 2))
  rope_val = jax.lax.bitcast_convert_type(bf16_uint8_reshaped, jnp.bfloat16)
  sf_val = jax.lax.bitcast_convert_type(
      scales_uint8, jnp.float8_e8m0fnu
  ).astype(jnp.bfloat16)

  fp8_val_blocked = fp8_val.reshape(
      total_num_pages * page_size_per_kv_packing * 4, 7, 64
  )
  sf_val_expanded = sf_val[..., None]
  dequant_nope = (fp8_val_blocked * sf_val_expanded).reshape(
      total_num_pages * page_size_per_kv_packing * 4, 448
  )
  dequantized_flat = jnp.concatenate([dequant_nope, rope_val], axis=-1)
  return dequantized_flat.reshape(
      total_num_pages, page_size_per_kv_packing * 4, 512
  )


def ref_implementation(
    q: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    new_kv: jax.Array,  # [num_tokens, actual_lkv_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    *,
    sliding_window: int,
    sm_scale: float = 1.0,
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
  # Pad q and q_pe to make the last dimension 128-byte aligned.
  actual_lkv_dim = q.shape[-1]
  lkv_dim = align_to(actual_lkv_dim, 128)
  if lkv_dim != actual_lkv_dim:
    q = jnp.pad(
        q,
        ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
        constant_values=0,
    )

  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs

  total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
      updated_cache_kv.shape
  )
  page_size = page_size_per_kv_packing * kv_packing
  assert lkv_dim == q.shape[-1]

  kv_c_cache = updated_cache_kv[..., :lkv_dim]

  # Quantize and dequantize kv_c_cache to simulate the loss of quantization
  kv_c_cache = quantize_and_dequantize_ref_cache(kv_c_cache).reshape(
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
    l_sinks = jnp.exp(attention_sinks[..., None, None] - m)
    l_final = l + l_sinks
    attn = jnp.exp(attn - m) / l_final

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


class CorrectnessTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(1234)
    self.rng_key = jax.random.PRNGKey(1234)

    self.kv_dtype = jnp.bfloat16
    self.q_dtype = jnp.bfloat16
    self.kv_packing = get_dtype_packing(self.kv_dtype)

    # Configuration (Smaller for correctness test)
    self.batch_size = 12
    self.num_heads = 128
    self.head_dim = 512
    self.lkv_dim = 640
    self.sliding_window = 32
    self.page_size = 12
    self.attention_sinks = jnp.array(
        self.rng.random(size=(self.num_heads,), dtype=np.float32) * 300.0
        + 200.0
    )

    self.ref_pages_per_seq = cdiv(self.sliding_window * 10, self.page_size)
    self.ref_page_indices = jnp.arange(
        self.batch_size * self.ref_pages_per_seq, dtype=jnp.int32
    )
    head_dim = self.head_dim
    self.ref_cache = jnp.zeros(
        (
            self.batch_size * self.ref_pages_per_seq,
            self.page_size // self.kv_packing,
            self.kv_packing,
            head_dim,
        ),
        dtype=self.kv_dtype,
    )

    # SW cache is in DDS V4 FP8 format.
    self.sw_physical_page_size = self.page_size + 4
    self.swc_cache = jnp.zeros(
        (
            self.batch_size * self.ref_pages_per_seq,
            self.sw_physical_page_size // self.kv_packing,
            get_dtype_packing(jnp.uint8),
            640,
        ),
        dtype=jnp.uint8,
    )
    self.swc_page_indices = self.ref_page_indices
    self.kv_lens = jnp.zeros((self.batch_size,), dtype=jnp.int32)

  def compare_cache(self, kv_lens):
    print("Comparing output cache...")
    # Construct valid token mask for cache comparison
    batch_size = self.batch_size
    pages_per_seq = self.ref_pages_per_seq
    page_size = self.page_size
    kv_packing = self.kv_packing

    # Create a grid of token indices for each position in the cache
    pages = np.arange(pages_per_seq)[:, None, None]
    rows = np.arange(page_size // kv_packing)[None, :, None]
    cols = np.arange(kv_packing)[None, None, :]
    token_indices = pages * page_size + rows * kv_packing + cols

    # kv_lens shape is (batch_size,). Reshape for broadcasting
    kv_lens_np = np.array(kv_lens)[:, None, None, None]
    valid_mask = token_indices[None, ...] < kv_lens_np
    valid_mask = valid_mask.reshape(
        batch_size * pages_per_seq, page_size // kv_packing, kv_packing, 1
    )

    ref_cache_masked = np.where(
        valid_mask, quantize_and_dequantize_ref_cache(self.ref_cache), 0
    )

    kv_packing_uint8 = get_dtype_packing(jnp.uint8)
    swc_dequant = dequantize_swa_cache(
        self.swc_cache[:, : page_size // kv_packing_uint8, :, :]
    )
    swc_dequant = swc_dequant.reshape(
        batch_size * pages_per_seq, page_size // kv_packing, kv_packing, 512
    )
    swc_cache_masked = np.where(valid_mask, swc_dequant, 0)

    diff_cache = np.abs(ref_cache_masked - swc_cache_masked)
    print(f"Max Diff Cache: {np.max(diff_cache)}")
    np.testing.assert_allclose(
        ref_cache_masked, swc_cache_masked, rtol=0.1, atol=0.1
    )

  def run_and_compare_outputs(
      self, q, new_kv, kv_lens, cu_q_lens, distribution
  ):
    total_tokens = q.shape[0]
    out_base, self.ref_cache, l_base, m_base = ref_implementation(
        q,
        new_kv,
        self.ref_cache,
        kv_lens,
        self.ref_page_indices,
        cu_q_lens,
        distribution,
        self.attention_sinks,
        sm_scale=1.0,
        sliding_window=self.sliding_window,
    )

    out, self.swc_cache, l, m = (
        mla_swa.mla_sliding_window_ragged_paged_attention(
            q,
            new_kv,
            self.swc_cache,
            kv_lens,
            self.swc_page_indices,
            cu_q_lens,
            distribution,
            self.attention_sinks,
            sm_scale=1.0,
            sliding_window=self.sliding_window,
            num_queries_per_block=8,
            num_kv_pages_per_block=2,
            logical_page_size=self.page_size,
        )
    )

    # Compare output
    print("Comparing output attention...")
    out_base.block_until_ready()
    out.block_until_ready()
    diff_out = np.abs(out_base - out)
    print(f"Max Diff Out: {np.max(diff_out)}")
    print(f"kv_lens: {kv_lens}")
    print(f"cu_q_lens: {cu_q_lens}")
    np.testing.assert_allclose(out_base, out, rtol=0.1, atol=0.1)

    l.block_until_ready()
    m.block_until_ready()
    l_base.block_until_ready()
    m_base.block_until_ready()
    assert l.shape == (total_tokens, self.num_heads)
    assert m.shape == (total_tokens, self.num_heads)
    np.testing.assert_allclose(l_base, l, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(m_base, m, rtol=0.1, atol=0.1)

    # Cache comparison
    self.compare_cache(kv_lens)

  def gen_random(self, shape, dtype):
    return jnp.array(self.rng.random(size=shape, dtype=np.float32)).astype(
        dtype
    )

  def gen_random_int(self, shape, low, high):
    self.rng_key, subkey = jax.random.split(self.rng_key)
    return jax.random.randint(
        subkey, shape=shape, minval=low, maxval=high, dtype=jnp.int32
    )

  def test_correctness_rng(self):
    print(f"JAX Backend: {jax.default_backend()}")

    # First step, contains variable length prefill
    new_kv_lens = self.gen_random_int(
        (self.batch_size,), self.sliding_window // 2, self.sliding_window * 2
    )
    cu_q_lens = jnp.concatenate(
        [jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)]
    )
    self.kv_lens += new_kv_lens
    total_tokens = jnp.sum(new_kv_lens)
    q = self.gen_random(
        (total_tokens, self.num_heads, self.head_dim), self.q_dtype
    )
    new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
    distribution = jnp.array([0, 0, self.batch_size], dtype=jnp.int32)

    self.run_and_compare_outputs(
        q, new_kv, self.kv_lens, cu_q_lens, distribution
    )

    # Second step, contains half decode and half prefill
    num_decode_seqs = self.batch_size // 2
    new_kv_lens = self.gen_random_int(
        (self.batch_size - num_decode_seqs,),
        self.sliding_window // 2,
        self.sliding_window * 2,
    )
    new_kv_lens = jnp.concatenate([
        jnp.ones((num_decode_seqs,), dtype=jnp.int32),
        new_kv_lens,
    ])
    cu_q_lens = jnp.concatenate(
        [jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)]
    )
    self.kv_lens += new_kv_lens
    total_tokens = jnp.sum(new_kv_lens)
    q = self.gen_random(
        (total_tokens, self.num_heads, self.head_dim), self.q_dtype
    )
    new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
    distribution = jnp.array(
        [num_decode_seqs, num_decode_seqs, self.batch_size], dtype=jnp.int32
    )

    self.run_and_compare_outputs(
        q, new_kv, self.kv_lens, cu_q_lens, distribution
    )

    # Third step, contains full decode
    new_kv_lens = jnp.ones((self.batch_size,), dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)]
    )
    self.kv_lens += new_kv_lens
    total_tokens = jnp.sum(new_kv_lens)
    q = self.gen_random(
        (total_tokens, self.num_heads, self.head_dim), self.q_dtype
    )
    new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
    distribution = jnp.array(
        [self.batch_size, self.batch_size, self.batch_size], dtype=jnp.int32
    )
    self.run_and_compare_outputs(
        q, new_kv, self.kv_lens, cu_q_lens, distribution
    )


if __name__ == "__main__":
  absltest.main()