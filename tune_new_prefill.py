import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 import (
    get_kv_cache_shape, ragged_paged_attention_hd64)
from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes_hd64 import \
    get_simplified_raw_key
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

# Temporarily set a large vmem limit for autotuning.
VMEM_LIMIT_BYTES = 60 * 1024 * 1024
SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
jax.config.parse_flags_with_absl()



# Temporarily set a large vmem limit for autotuning.
VMEM_LIMIT_BYTES = 120 * 1024 * 1024
SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
jax.config.parse_flags_with_absl()


# This is a typical distrubution of kv_lens and cu_q_lens.
def get_qkv_lens_example(max_num_tokens, max_model_len, actual_num_seqs):
  assert max_num_tokens >= actual_num_seqs
  # 0 = pure prefill
  decode_end = 0 # actual_num_seqs - 1
  cu_q_lens = list(range(actual_num_seqs + 1))
  cu_q_lens[-1] = min(max_num_tokens, max_model_len)
  kv_lens = [max_model_len for _ in range(actual_num_seqs)]
  return cu_q_lens, kv_lens, decode_end


def autotune(
    example,
    key,
    max_num_tokens,
    max_num_seqs,
    bkv_p_lst,
    bq_sz_lst,
    total_num_pages=1000,
    num_iterations=10,
    *,
    use_xprof=False,
):
  """Find the best (num_kv_pages_per_block, num_q_per_block)."""
  (
      page_size,
      q_dtype_name,
      kv_dtype_name,
      num_q_heads,
      num_kv_heads,
      head_dim,
      max_model_len,
  ) = key
  q_dtype = jnp.dtype(q_dtype_name)
  kv_dtype = jnp.dtype(kv_dtype_name)
  pages_per_seq = cdiv(max_model_len, page_size)
  cu_q_lens, kv_lens, decode_end = example
  actual_num_seqs = len(kv_lens)
  cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
  kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
  cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
  kv_lens = jnp.pad(kv_lens, (0, max_num_seqs - kv_lens.shape[0]))

  q_shape = (max_num_tokens, num_q_heads, head_dim)
  kv_shape = (max_num_tokens, num_kv_heads, head_dim)
  kv_cache_shape = get_kv_cache_shape(
      total_num_pages,
      page_size,
      num_kv_heads,
      head_dim,
      kv_dtype,
  )

  q = jnp.array(
      np.random.rand(*q_shape),
      dtype=q_dtype,
  )
  k = jnp.array(
      np.random.rand(*kv_shape),
      dtype=kv_dtype,
  )
  v = jnp.array(
      np.random.rand(*kv_shape),
      dtype=kv_dtype,
  )
  kv_cache = jnp.array(
      np.random.rand(*kv_cache_shape),
      dtype=kv_dtype,
  )
  page_indices = np.random.randint(
      0, total_num_pages, size=(max_num_seqs * pages_per_seq,), dtype=jnp.int32
  )

  distribution = jnp.array(
      [decode_end, decode_end, actual_num_seqs], dtype=jnp.int32
  )

  args = [
      q,
      k,
      v,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  ]

  best_block_size = None
  best_t = None
  for num_kv_pages_per_block in bkv_p_lst:
    if num_kv_pages_per_block > pages_per_seq:
      print(
          f"[Debug] Skip ({page_size=}, {num_kv_pages_per_block=}) because"
          f" {num_kv_pages_per_block=} > {pages_per_seq=}"
      )
      continue
    if page_size * num_kv_pages_per_block > 4096:
      print(
          f"[Debug] Skip because ({page_size=}) * ({num_kv_pages_per_block=}) ="
          f" {page_size * num_kv_pages_per_block} > 4096"
      )
      continue
    for num_q_per_block in bq_sz_lst:
      expected_cnt = 1

      kwargs = {
          "num_kv_pages_per_block": num_kv_pages_per_block,
          "num_queries_per_block": num_q_per_block,
          # Temporarily set a large vmem limit for autotuning.
          "vmem_limit_bytes": VMEM_LIMIT_BYTES,
      }
      if use_xprof:
        pass
      else:
        try:
          # Warm up.
          _, args[3] = jax.block_until_ready(
              ragged_paged_attention_hd64(*args, **kwargs)
          )
          start_time = time.perf_counter_ns()
          for _ in range(num_iterations):
            _, args[3] = jax.block_until_ready(
                ragged_paged_attention_hd64(*args, **kwargs)
            )
          end_time = time.perf_counter_ns()
          t = (end_time - start_time) / num_iterations
          print(t, distribution, kv_lens, cu_q_lens)
        except Exception as err:
          print(
              f"[Debug] Failed with ({page_size=}, {num_kv_pages_per_block=},"
              f" {num_q_per_block=}), got error: {err=}"
          )
          continue
      print(
          f"[Debug] {page_size=}, {num_kv_pages_per_block=},"
          f" {num_q_per_block=}, {t=}"
      )
      if best_t is None or t < best_t:
        best_block_size = (num_kv_pages_per_block, num_q_per_block)
        best_t = t
  return best_block_size


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class Autotune(jtu.JaxTestCase):

  @parameterized.product(
      page_size=[256],
      q_dtype=[jnp.bfloat16],
      kv_dtype=[jnp.bfloat16, jnp.float8_e4m3fn],
      num_q_kv_heads=[(2,1), (4,1), (8,1), (2,2), (4,2), (8,2), (4,4), (8,4)],
      head_dim=[64],
      max_model_len=[131072], # 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
      max_num_tokens=[4096],
      max_num_seqs=[1],
      bkv_p_lst=[(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)],
      bq_sz_lst=[(8, 16, 32, 64, 128, 256, 512)],
  )
  def test_autotune(
      self,
      page_size,
      q_dtype,
      kv_dtype,
      num_q_kv_heads,
      head_dim,
      max_model_len,
      max_num_tokens,
      max_num_seqs,
      bkv_p_lst,
      bq_sz_lst,
  ):
    # Currently we only use one example to autotune. If necessary, we can
    # construct decode-heavy or prefill-heavy examples.
    max_num_tokens = max_model_len
    example = get_qkv_lens_example(
        max_num_tokens,
        max_model_len,
        actual_num_seqs=1,
    )
    num_q_heads, num_kv_heads = num_q_kv_heads
    if max_model_len < page_size:
      return
    print(f"[Debug] {num_q_heads=}, {num_kv_heads=}")
    print(f"[Debug] {example=}")

    rows = []
    key = get_simplified_raw_key(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
    )
    print(f"[Debug] Simplified key: {key}")
    # best_block_size: (num_kv_pages_per_block, num_q_per_block).
    # try:
    best_block_size = autotune(
        example,
        key,
        max_num_tokens,
        max_num_seqs,
        bkv_p_lst,
        bq_sz_lst,
        num_iterations=50,
    )
    # except Exception as err:
    #   print(f"[Debug] Failed with {key=} {err=}")
    #   return
    if best_block_size is not None:
      rows.append(f"{key}: {best_block_size},")

    print(rows)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
