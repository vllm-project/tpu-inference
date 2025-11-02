"""Autotune for ragged paged attention kernel."""

# Run
# - v5e:
#   blaze test --test_output=errors //experimental/users/jevinjiang/ullm:google/ragged_paged_attention/v3/autotune/autotune_vl
# - v6e:
#   blaze test --test_output=errors //experimental/users/jevinjiang/ullm:google/ragged_paged_attention/v3/autotune/autotune_gl
# - v7:
#   blaze test --test_output=errors //experimental/users/jevinjiang/ullm:google/ragged_paged_attention/v3/autotune/autotune_gf

import time
import uuid
import os
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    dynamic_validate_inputs,
    get_kv_cache_shape,
    get_smem_estimate_bytes,
    get_vmem_estimate_bytes,
    ragged_paged_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes import get_simplified_raw_key
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

from tpu_inference.kernels.ragged_paged_attention.v3.benchmark import extract_op_times_from_xplane

# Temporarily set a large vmem limit for autotuning.
VMEM_LIMIT_BYTES = 60 * 1024 * 1024
SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
jax.config.parse_flags_with_absl()

# This is a typical distrubution of kv_lens and cu_q_lens.
def get_qkv_lens_example(max_num_tokens, max_model_len, actual_num_seqs):
  assert max_num_tokens >= actual_num_seqs
  decode_end = actual_num_seqs - 1
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
    num_iterations=100,
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
    # if page_size * num_kv_pages_per_block > 4096:
    #   print(
    #       f"[Debug] Skip because ({page_size=}) * ({num_kv_pages_per_block=}) ="
    #       f" {page_size * num_kv_pages_per_block} > 4096"
    #   )
    #   continue
    for num_q_per_block in bq_sz_lst:
      expected_cnt = 1

      kwargs = {
          "num_kv_pages_per_block": num_kv_pages_per_block,
          "num_queries_per_block": num_q_per_block,
          # Temporarily set a large vmem limit for autotuning.
          "vmem_limit_bytes": VMEM_LIMIT_BYTES,
      }

      try:
        dynamic_validate_inputs(*args, **kwargs)
      except Exception as err:
        print(
            f"[Debug] Failed with ({page_size=}, {num_kv_pages_per_block=},"
            f" {num_q_per_block=}), got error: {err=}"
        )
        continue

      vmem_estimate = get_vmem_estimate_bytes(
          num_q_heads,
          num_kv_heads,
          head_dim,
          num_q_per_block,
          num_kv_pages_per_block,
          q_dtype,
          kv_dtype,
      )
      if vmem_estimate > VMEM_LIMIT_BYTES:
        print(
            f"[Debug] Skip ({page_size=}, {num_kv_pages_per_block=},"
            f" {num_q_per_block=}) because {vmem_estimate=} >"
            f" {VMEM_LIMIT_BYTES=}"
        )
        continue
      smem_estimate = get_smem_estimate_bytes(
          max_num_seqs,
          pages_per_seq,
      )
      if smem_estimate > SMEM_LIMIT_BYTES:
        print(
            f"[Debug] Skip ({page_size=}, {num_kv_pages_per_block=},"
            f" {num_q_per_block=}) because {smem_estimate=} >"
            f" {SMEM_LIMIT_BYTES=}"
        )
        continue

      uid = str(uuid.uuid4())
      uid_profile_dir = f"/tmp/jax-trace/{uid}"
      options = jax.profiler.ProfileOptions()
      options.python_tracer_level = os.getenv("PYTHON_TRACER_LEVEL", 0)

      with jax.profiler.trace(uid_profile_dir, profiler_options=options):
        try:
          _, args[3] = jax.block_until_ready(
              ragged_paged_attention(*args, **kwargs)
          )
        except Exception as err:
          print(
              f"[Debug] Failed with ({page_size=}, {num_kv_pages_per_block=},"
              f" {num_q_per_block=}), got error: {err=}"
          )
          continue

      t, cnt, path = extract_op_times_from_xplane(uid_profile_dir, "jit_ragged_paged_attention")
      assert cnt == expected_cnt, f"{cnt=}, {expected_cnt=}"
      t /= cnt
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
      num_q_heads=[2, 4, 8],
      num_kv_heads=[2, 4],
      head_dim=[128, 256],
      max_model_len=[16384, 65536, 131072],
      max_num_tokens=[1024],
      max_num_seqs=[256],
      bkv_p_lst=[(1, 2, 4, 8, 16, 32, 64, 128, 256)],
      bq_sz_lst=[(8, 16, 32, 64, 128, 256)],
  )
  def test_autotune(
      self,
      page_size,
      q_dtype,
      kv_dtype,
      num_q_heads,
      num_kv_heads,
      head_dim,
      max_model_len,
      max_num_tokens,
      max_num_seqs,
      bkv_p_lst,
      bq_sz_lst,
  ):
    # Currently we only use one example to autotune. If necessary, we can
    # construct decode-heavy or prefill-heavy examples.
    example = get_qkv_lens_example(
        max_num_tokens,
        max_model_len,
        actual_num_seqs=256,
    )
    if num_q_heads % num_kv_heads != 0:
      print("Skip because num_q_heads % num_kv_heads != 0")
      return
    if max_model_len < page_size:
      print("Skip because max_model_len < page_size")
      return
    # print(f"[Debug] {example=}")

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
    # best_block_size: (num_kv_pages_per_block, num_q_per_block).
    print("STARTING AUTOTUNE", key)
    best_block_size = autotune(
        example,
        key,
        max_num_tokens,
        max_num_seqs,
        bkv_p_lst,
        bq_sz_lst,
        num_iterations=100,
    )
    print("Best block size:", best_block_size)
    if best_block_size is not None:
      rows.append(f"{key}: {best_block_size},")

    if rows:
      with open("autotune_table.txt", "a") as f:
        data = "\n".join(rows)
        f.write(str(data) + '\n')


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
