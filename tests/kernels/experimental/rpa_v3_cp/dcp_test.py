from absl.testing import absltest, parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

import os
USE_BATCHED_RPA_KERNEL = (
    os.environ.get("USE_BATCHED_RPA_KERNEL", "0").lower() == "1"
)

if USE_BATCHED_RPA_KERNEL:
    print('Use batched RPA kernel')
    from tpu_inference.kernels.experimental.batched_rpa import configs
    from tpu_inference.kernels.experimental.batched_rpa.utils import align_to, get_dtype_packing
    from tpu_inference.kernels.experimental.batched_rpa.wrapper import ragged_paged_attention
else:
    from tpu_inference.kernels.experimental.batched_rpa import configs
    from tpu_inference.kernels.experimental.batched_rpa.utils import align_to, get_dtype_packing
    from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import ragged_paged_attention


def cdiv(a, b):
  return (a + b - 1) // b


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
  assert k.shape == v.shape
  assert k.dtype == v.dtype
  max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
  kv_packing = get_dtype_packing(k.dtype)
  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

  head_dim = align_to(actual_head_dim, 128)
  kv = jnp.pad(
      jnp.concat([k, v], axis=-1).reshape(
          max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
      ),
      (
          (0, 0),
          (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  ).reshape(
      max_num_tokens,
      num_kv_heads_x2 // kv_packing,
      kv_packing,
      head_dim,
  )
  return kv


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if out_dtype is None:
    out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

  if mask_value is None:
    # We do not set to -inf directly because (-inf) - (-inf) is nan.
    mask_value = -float(jnp.finfo(out_dtype).max)

  actual_head_dim = queries.shape[2]
  actual_num_q_heads = queries.shape[1]
  actual_num_kv_heads = keys.shape[1]
  merged_kv = merge_kv(keys, values)
  assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

  _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
      kv_cache.shape
  )
  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
  assert num_kv_heads_x2 % 2 == 0
  assert actual_num_q_heads % actual_num_kv_heads == 0
  assert head_dim % 128 == 0
  assert get_dtype_packing(kv_cache.dtype) == kv_packing
  assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  outputs = []

  for i in range(distribution[-1]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start

    kv_len = kv_lens[i]
    indices_start = i * pages_per_seq
    indices_end = indices_start + cdiv(kv_len, page_size)
    indices = page_indices[indices_start:indices_end]
    q = queries[q_start:q_end, :, :actual_head_dim]

    # Update the kv cache.
    assert kv_len - q_len >= 0
    gathered_kv = kv_cache[indices]
    gathered_shape = gathered_kv.shape
    gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
    gathered_kv = gathered_kv.at[kv_len - q_len : kv_len].set(
        merged_kv[q_start:q_end]
    )
    kv_cache = kv_cache.at[indices].set(gathered_kv.reshape(gathered_shape))

    kv = gathered_kv.reshape(-1, num_kv_heads_x2, head_dim)[
        :, : actual_num_kv_heads * 2, :
    ].reshape(-1, actual_num_kv_heads, head_dim * 2)
    k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
    v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
    k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
    v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

    if q_scale is not None:
      q = q / q_scale
      if jnp.issubdtype(k.dtype, jnp.floating):
        dtype_info = jnp.finfo(k.dtype)
        minval = float(dtype_info.min)
        maxval = float(dtype_info.max)
        q = jnp.clip(q, min=minval, max=maxval)
      q = q.astype(k.dtype)

    attn = jnp.einsum(
        "qhd,khd->hqk", q, k, preferred_element_type=jnp.float32
    ).astype(out_dtype)
    attn *= sm_scale
    if k_scale is not None:
      attn *= k_scale
    if q_scale is not None:
      attn *= q_scale
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)

    if use_causal_mask:
      q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
          jnp.int32, attn.shape, 1
      )
      kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
      mask = q_span >= kv_span
      if sliding_window is not None:
        mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
      attn = jnp.where(mask, attn, mask_value)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
    if v_scale is not None:
      out *= v_scale

    outputs.append(out)

  result = jnp.concatenate(outputs, axis=0)
  return result, kv_cache


jax.config.parse_flags_with_absl()
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionDecodeContextParallelismTest(jtu.JaxTestCase):
  def _test_two_phase_attention_mixed_engine(self, seq_lens, cp_group_size):
    print(
        "-------------------- Mixed engine attention"
        " --------------------"
    )
    # Init data
    max_num_batched_tokens = 512
    max_num_seq = 8
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.bfloat16
    # Lower head dimension.
    num_heads = (16, 2)
    head_dim = 128
    page_size = 32 
    rng = np.random.default_rng(1234)
    def gen_random(shape, dtype):
      return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Expect TPUv4+")
    cu_q_lens = [0]
    kv_lens_list = []
    for q_len, kv_len in seq_lens:
      assert q_len <= kv_len
      cu_q_lens.append(cu_q_lens[-1] + q_len)
      kv_lens_list.append(kv_len)
    max_num_batched_tokens = max(
        align_to(cu_q_lens[-1], 128), max_num_batched_tokens
    )
    max_num_seq = max(align_to(len(seq_lens), 8), max_num_seq)
    max_kv_len = max(kv_lens_list)
    pages_per_seq = cdiv(max_kv_len, page_size)
    num_q_heads, num_kv_heads = num_heads
    q = gen_random((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
    k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)
    v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)
    page_cnt = 0
    page_indices_list = []
    kv_packing = get_dtype_packing(kv_dtype)
    padded_head_dim = align_to(head_dim, 128)
    num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
    for kv_len in kv_lens_list:
      num_pages_for_seq = cdiv(kv_len, page_size)
      indices = page_cnt + jnp.arange(num_pages_for_seq, dtype=jnp.int32)
      indices = jnp.pad(
          indices,
          ((0, pages_per_seq - indices.shape[0]),),
          constant_values=0,
      )
      page_indices_list.append(indices)
      # print(f"page indices for seq: {indices}")
      page_cnt += num_pages_for_seq
    num_pages = max(1000, page_cnt)
    kv_cache_shape = (
        num_pages,
        page_size,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        padded_head_dim,
    )
    kv_cache = jnp.full(kv_cache_shape, 0.0, dtype=kv_dtype)
    page_indices = jnp.stack(page_indices_list, axis=0)
    page_indices = jnp.pad(
        page_indices,
        ((0, max_num_seq - page_indices.shape[0]), (0, 0)),
        constant_values=0,
    )
    page_indices = page_indices.reshape(-1)
    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
    cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seq + 1 - cu_q_lens.shape[0]))
    kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
    kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
    distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)
    q_lens = np.array([q_len for q_len, _ in seq_lens], dtype=np.int32)
    q_lens = np.pad(q_lens, (0, max_num_seq - q_lens.shape[0]))
    # Pre-populate the prefix (context) in the kv_cache.
    prefixes_kv = []
    for i, (q_len, kv_len) in enumerate(seq_lens):
      prefix_len = kv_len - q_len
      if prefix_len <= 0:
        prefixes_kv.append(None)
        continue
      prefix_k = gen_random((prefix_len, num_kv_heads, head_dim), kv_dtype)
      prefix_v = gen_random((prefix_len, num_kv_heads, head_dim), kv_dtype)
      prefix_kv = merge_kv(prefix_k, prefix_v)
      prefixes_kv.append(prefix_kv)

    # Pre-populate the full prefix in the reference kv_cache
    for i, prefix_kv in enumerate(prefixes_kv):
      if prefix_kv is None:
        continue
      prefix_len = prefix_kv.shape[0]
      indices_start = i * pages_per_seq
      num_prefix_pages = cdiv(prefix_len, page_size)
      indices = page_indices[indices_start : indices_start + num_prefix_pages]
      padded_prefix_len = num_prefix_pages * page_size
      prefix_kv_padded = jnp.pad(
          prefix_kv,
          ((0, padded_prefix_len - prefix_len), (0, 0), (0, 0), (0, 0)),
          constant_values=0.0,
      ).reshape(num_prefix_pages, page_size, *prefix_kv.shape[1:])
      kv_cache = kv_cache.at[indices].set(prefix_kv_padded)

    def get_kv_cache_for_rank(rank):
      kv_cache_rank = jnp.full(kv_cache_shape, 0.0, dtype=kv_dtype)
      for i, prefix_kv in enumerate(prefixes_kv):
        if prefix_kv is None:
          continue
        prefix_kv_rank = prefix_kv[rank::cp_group_size]
        prefix_len_rank = prefix_kv_rank.shape[0]
        if prefix_len_rank == 0:
          continue
        indices_start = i * pages_per_seq
        num_prefix_pages = cdiv(prefix_len_rank, page_size)
        indices = page_indices[indices_start : indices_start + num_prefix_pages]
        padded_prefix_len = num_prefix_pages * page_size
        prefix_kv_padded = jnp.pad(
            prefix_kv_rank,
            ((0, padded_prefix_len - prefix_len_rank), (0, 0), (0, 0), (0, 0)),
            constant_values=0.0,
        ).reshape(num_prefix_pages, page_size, *prefix_kv_rank.shape[1:])
        kv_cache_rank = kv_cache_rank.at[indices].set(prefix_kv_padded)
      return kv_cache_rank

    kwargs = {
        "use_causal_mask": True,
    }
    # Reference baseline: full KV write back
    expected_out, expected_kv_cache = ref_ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        **kwargs,
    )
    print(f"expected_out: {expected_out.shape}")
    query_outs = []
    query_lses = []
    context_outs = []
    context_lses = []
    for rank in range(cp_group_size):
      print(f"Compute attention for current token only on rank {rank}")
      query_out, updated_kv_cache, query_lse = ragged_paged_attention(
          q,
          k,
          v,
          get_kv_cache_for_rank(rank),
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          #
          update_kv_cache=True,
          cp_rank=jnp.array([rank], dtype=jnp.int32),
          cp_group_size=cp_group_size,
          skip_cache_attn=True,
          return_lse=True,
          # debug_mode=True,
          **kwargs,
      )
      query_outs.append(query_out)
      query_lses.append(query_lse)
      print("Verifying KV cache for rank after FIRST call...")
      for i, (q_len, kv_len) in enumerate(seq_lens):
        num_pages_seq = cdiv(kv_len, page_size)
        indices_start = i * pages_per_seq
        page_indices_seq = page_indices[indices_start : indices_start + num_pages_seq]
        expected_kv_seq = expected_kv_cache[page_indices_seq].reshape(-1, *kv_cache.shape[-3:])[:kv_len]
        rank_kv_seq = updated_kv_cache[page_indices_seq].reshape(-1, *kv_cache.shape[-3:])[:kv_len]
        expected_kv_for_rank = expected_kv_seq[rank::cp_group_size]
        num_tokens_for_rank = expected_kv_for_rank.shape[0]
        self.assertAllClose(
            rank_kv_seq[:num_tokens_for_rank],
            expected_kv_for_rank,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"KV cache after FIRST call for rank {rank}, seq {i} does not match expected.",
        )
        print(f"KV cache for rank {rank}, seq {i} passed!")
      print(f"Compute attention for context only on rank {rank}")
      context_out, final_kv_cache, context_lse = ragged_paged_attention(
          q,
          k,
          v,
          updated_kv_cache,  # use updated kv_cache
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          #
          update_kv_cache=False,
          cp_rank=jnp.array([rank], dtype=jnp.int32),
          cp_group_size=cp_group_size,
          return_lse=True,
          use_causal_mask=False,
          skip_current_attn=True,
          # debug_mode=True,
      )
      context_outs.append(context_out)
      context_lses.append(context_lse)
      print(f"LSE: current={query_lse[:seq_lens[0][0]]}")
      print(f"LSE: context={context_lse[:seq_lens[0][0]]}")
      print(f"Verifying KV cache for rank {rank}...")
      for i, (q_len, kv_len) in enumerate(seq_lens):
        num_pages_seq = cdiv(kv_len, page_size)
        indices_start = i * pages_per_seq
        page_indices_seq = page_indices[indices_start : indices_start + num_pages_seq]
        expected_kv_seq = expected_kv_cache[page_indices_seq].reshape(
            -1, *kv_cache.shape[-3:]
        )[:kv_len]
        rank_kv_seq = final_kv_cache[page_indices_seq].reshape(
            -1, *kv_cache.shape[-3:]
        )[:kv_len]
        expected_kv_for_rank = expected_kv_seq[rank::cp_group_size]
        num_tokens_for_rank = expected_kv_for_rank.shape[0]
        self.assertAllClose(
            rank_kv_seq[:num_tokens_for_rank],
            expected_kv_for_rank,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"KV cache for rank {rank}, seq {i} does not match expected KV cache.",
        )

    # Merge all attention results from all ranks and phases
    print("Merging attention results across ranks and phases...")
    outs_to_merge = []
    lses_to_merge = []
    for rank in range(cp_group_size):
      outs_to_merge.extend([query_outs[rank], context_outs[rank]])
      lses_to_merge.extend([query_lses[rank], context_lses[rank]])

    stacked_lses = jnp.stack(lses_to_merge, axis=0)
    max_lse = jnp.max(stacked_lses, axis=0)
    exp_sums = jnp.zeros_like(max_lse)
    weighted_outs = jnp.zeros_like(outs_to_merge[0])

    for out, lse in zip(outs_to_merge, lses_to_merge):
      exp_val = jnp.exp(lse - max_lse)
      exp_sums += exp_val
      weighted_outs += out * exp_val[..., None]

    merged_out = weighted_outs / exp_sums[..., None]
    print(f"merged_out: {merged_out.shape}")
    print("Verifying Output...")
    self.assertAllClose(
        merged_out[:cu_q_lens[distribution[-1]]],
        expected_out,
        rtol=1e-6,
        atol=0.2,
        err_msg="Attention output does not match the expected baseline",
    )
    print("Output test passed!")

  @parameterized.parameters(1, 2)
  def test_two_phase_attention_mixed_engine(self, cp_group_size):
    seq_lens = []
    q_lens = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1011
    ]
    kv_lens = [
        1026, 1026, 1026, 1025, 1025, 1025, 1025, 1025, 1025, 1025,
        1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1011
    ]
    for q_len, kv_len in zip(q_lens, kv_lens):
      seq_lens.append((q_len, kv_len))
    self._test_two_phase_attention_mixed_engine(seq_lens=seq_lens, cp_group_size=cp_group_size)
if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
