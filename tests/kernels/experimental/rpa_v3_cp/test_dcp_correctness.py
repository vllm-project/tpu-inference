from absl.testing import absltest
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    ref_ragged_paged_attention,
    ragged_paged_attention,
    merge_kv,
    align_to,
    cdiv,
    get_dtype_packing,
)


jax.config.parse_flags_with_absl()

rtol, atol = 1e-5, 1e-2

@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionDecodeContextParallelismTest(jtu.JaxTestCase):
  def _test_two_phase_attention_mixed_engine(self, seq_lens, profile: bool = False):
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
    for i, (q_len, kv_len) in enumerate(seq_lens):
      prefix_len = kv_len - q_len
      if prefix_len <= 0:
        continue
      prefix_k = gen_random((prefix_len, num_kv_heads, head_dim), kv_dtype)
      prefix_v = gen_random((prefix_len, num_kv_heads, head_dim), kv_dtype)
      prefix_kv = merge_kv(prefix_k, prefix_v)

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

    cp_group_size = 1
    session = None
    session_id = None
    for rank in range(cp_group_size):
      if profile and rank == 0:
        session = xprof_session.XprofSession()
        session.start_session(
            device_name="", enable_python_tracer=True, host_trace_level=2
        )
      print("Compute attention for current token only")
      query_out, updated_kv_cache, query_lse = ragged_paged_attention(
          q,
          k,
          v,
          kv_cache,
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

      print("Compute attention for context only")
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
      print(f"LSE: current={query_lse[:seq_lens[0][0]]}")
      print(f"LSE: context={context_lse[:seq_lens[0][0]]}")

      # Merge two attention results
      max_lse = jnp.maximum(query_lse, context_lse)
      exp_query = jnp.exp(query_lse - max_lse)
      exp_context = jnp.exp(context_lse - max_lse)
      sum_exp = exp_query + exp_context

      merged_out = (
          context_out * exp_context[..., None]
          + query_out * exp_query[..., None]
      ) / sum_exp[..., None]

      if profile and rank == 0:
        jax.block_until_ready([merged_out, final_kv_cache])

      print(f"merged_out: {merged_out.shape}")
      print("Verifying Output...")
      print(f"Merged out: {merged_out[:cu_q_lens[distribution[-1]]]}")
      print(f"Expected out: {merged_out[:cu_q_lens[distribution[-1]]]}")
      self.assertAllClose(
          merged_out[:cu_q_lens[distribution[-1]]],
          expected_out,
          rtol=rtol,
          atol=atol,
          err_msg="Attention output does not match the expected baseline",
      )
      print("Output test passed!")

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
            rtol=rtol,
            atol=atol,
            err_msg=f"KV cache for rank {rank}, seq {i} does not match expected KV cache.",
        )

  def test_two_phase_attention_mixed_engine(self):
    print("-------------------- Mixed")
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
    self._test_two_phase_attention_mixed_engine(seq_lens=seq_lens)



if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
