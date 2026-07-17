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
"""Speed benchmark: MRIS fused attention vs production RPA kernel.

Measures wall-clock time for processing N_FUSED=4 decode requests using:
  1. **Sequential RPA**: 4 independent calls to the production
     ``ragged_paged_attention`` kernel from tpu_inference (v3).
  2. **MRIS Fused**: A single call to ``mris_fused_attention`` that fuses all 4
     requests into one kernel launch.

Reports average latency, throughput (requests/sec), and speedup ratio.
Requires TPU hardware for meaningful results.
"""

import timeit

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tpu_inference  # noqa: F401
from tpu_inference.kernels.mris_attention.kernel import (
    N_FUSED,
    mris_fused_attention_v3c,
    mris_fused_paged_attention_v3,
    ref_mris_fused_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape,
    ragged_paged_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to,
    cdiv,
    get_dtype_packing,
)


def log_trace(msg: str):
  from absl import logging
  import os

  logging.error(msg)
  outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
  if outputs_dir:
    with open(os.path.join(outputs_dir, "test_trace.txt"), "a") as f:
      f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _setup_rpa_inputs(
    *,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype,
    kv_dtype,
    num_pages: int,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 512,
    rng: np.random.RandomState,
):
  """Sets up inputs for a single-request RPA call (decode: q_len=1).

  Returns:
    Tuple of (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens,
              distribution) ready for ``ragged_paged_attention()``.
    Also returns (k_gathered, v_gathered) for the MRIS comparison.
  """

  def gen(shape, dtype):
    return jnp.array(rng.random(size=shape).astype(np.float32)).astype(dtype)

  padded_head_dim = align_to(head_dim, 128)
  kv_packing = get_dtype_packing(kv_dtype)
  num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
  pages_per_seq = cdiv(kv_len, page_size)
  max_num_batched_tokens = max(align_to(q_len, 128), max_num_batched_tokens)
  max_num_seqs = max(align_to(1, 8), max_num_seqs)

  q = gen((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
  k = gen((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)
  v = gen((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)

  # Build paged KV cache for this request
  kv_data = gen(
      (kv_len, num_kv_heads_x2 // kv_packing, kv_packing, padded_head_dim),
      kv_dtype,
  )
  kv_data = jnp.pad(
      kv_data,
      ((0, pages_per_seq * page_size - kv_len), (0, 0), (0, 0), (0, 0)),
      constant_values=0,
  ).reshape(
      -1, page_size, num_kv_heads_x2 // kv_packing, kv_packing, padded_head_dim
  )

  indices = jnp.arange(kv_data.shape[0], dtype=jnp.int32)
  indices = jnp.pad(
      indices, ((0, pages_per_seq - indices.shape[0]),), constant_values=0
  )

  kv_cache = jnp.pad(
      kv_data,
      ((0, num_pages - kv_data.shape[0]), (0, 0), (0, 0), (0, 0), (0, 0)),
      constant_values=0,
  )
  page_indices = jnp.pad(
      indices.reshape(1, -1),
      ((0, max_num_seqs - 1), (0, 0)),
      constant_values=0,
  ).reshape(-1)

  cu_q_lens = jnp.zeros(max_num_seqs + 1, dtype=jnp.int32).at[1].set(q_len)
  kv_lens_arr = jnp.zeros(max_num_seqs, dtype=jnp.int32).at[0].set(kv_len)

  # For decode: all requests are in the "mixed" bucket
  distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

  # Also extract gathered K/V for MRIS comparison
  # Gather from cache: [pages, page_size, ...] -> [kv_len, head_dim]
  gathered_kv = kv_data.reshape(
      -1, num_kv_heads_x2 // kv_packing, kv_packing, padded_head_dim
  )
  gathered_kv = gathered_kv[:kv_len]
  # Reshape to [kv_len, num_kv_heads*2, padded_head_dim] then extract K and V
  gathered_flat = gathered_kv.reshape(kv_len, num_kv_heads_x2, padded_head_dim)
  k_gathered = gathered_flat[
      :, :num_kv_heads, :head_dim
  ]  # [kv_len, num_kv_heads, head_dim]
  v_gathered = gathered_flat[:, num_kv_heads : num_kv_heads * 2, :head_dim]

  return (
      q,
      k,
      v,
      kv_cache,
      kv_lens_arr,
      page_indices,
      cu_q_lens,
      distribution,
      k_gathered,
      v_gathered,
  )


def _benchmark_fn(fn, warmup: int = 5, iterations: int = 20):
  """Benchmark a JAX function with warmup and measurement."""
  from absl import logging

  for i in range(warmup):
    logging.error(f"  Warmup step {i} start")
    result = fn()
    logging.error(f"  Warmup step {i} fn run done, waiting")
    jax.tree.map(lambda x: x.block_until_ready(), result)
    logging.error(f"  Warmup step {i} done")

  def timed_fn():
    result = fn()
    jax.tree.map(lambda x: x.block_until_ready(), result)

  elapsed = timeit.timeit(timed_fn, number=iterations)
  avg_ms = (elapsed / iterations) * 1000.0
  return avg_ms


def _format_report(
    label: str,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    kv_lens: jax.Array,
    rpa_ms: float,
    mris_ms: float,
) -> str:
  """Format a human-readable benchmark report."""
  speedup = rpa_ms / mris_ms if mris_ms > 0 else float("inf")
  reduction_pct = (1 - mris_ms / rpa_ms) * 100 if rpa_ms > 0 else 0

  lines = [
      f"\n{'='*70}",
      f"  MRIS Benchmark: {label}",
      f"{'='*70}",
      (
          f"  Config: head_dim={head_dim}, q_heads={num_q_heads}, "
          f"kv_heads={num_kv_heads}, N_FUSED={N_FUSED}"
      ),
      f"  KV lengths: {kv_lens}",
      f"  Sequential RPA (4x):  {rpa_ms:8.3f} ms",
      f"  MRIS Fused:           {mris_ms:8.3f} ms",
      f"  Speedup:              {speedup:8.2f}x",
      f"  Latency reduction:    {reduction_pct:8.1f}%",
      f"{'='*70}",
  ]
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class MrisBenchmarkTest(parameterized.TestCase):
  """Speed comparison: production RPA kernel (sequential) vs MRIS fused."""

  def setUp(self):
    super().setUp()
    import jax

    jax.clear_caches()
    import os

    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not outputs_dir:
      raise ValueError("TEST_UNDECLARED_OUTPUTS_DIR is not set!")
    log_trace(f"[STARTING TEST] {self.id()}")
    if not any("TPU" in d.device_kind for d in jax.devices()):
      self.skipTest("Speed benchmarks require TPU hardware")

  def tearDown(self):
    import jax

    jax.clear_caches()
    super().tearDown()

  @parameterized.parameters(
      dict(
          label="Batch 1 (1k ctx)",
          batch_size=1,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
      dict(
          label="Batch 3 (1k ctx)",
          batch_size=3,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
      dict(
          label="Batch 4 (1k ctx)",
          batch_size=4,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
      dict(
          label="Batch 5 (1k ctx)",
          batch_size=5,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
      dict(
          label="Batch 7 (1k ctx)",
          batch_size=7,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
      dict(
          label="Batch 15 (1k ctx)",
          batch_size=15,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
      dict(
          label="Batch 16 (1k ctx)",
          batch_size=16,
          kv_len=1024,
          num_q_heads=8,
          num_kv_heads=1,
          head_dim=128,
          page_size=16,
      ),
  )
  def test_benchmark_mris_vs_rpa(
      self,
      label,
      batch_size,
      kv_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      page_size,
  ):
    log_trace(
        f"[STARTING METHOD] {label} (batch_size={batch_size}, kv_len={kv_len})"
    )
    sm_scale = head_dim**-0.5
    kv_lens = jnp.array([kv_len] * batch_size, dtype=jnp.int32)
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.bfloat16
    num_pages = cdiv(kv_len, page_size) * batch_size + 64  # extra padding
    rng = np.random.RandomState(42)
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads

    # Set up per-request RPA inputs
    rpa_inputs = []
    mris_queries = []
    mris_k_pages = []
    mris_v_pages = []

    for i in range(batch_size):
      (
          q,
          k,
          v,
          kv_cache,
          kv_lens_arr,
          page_indices,
          cu_q_lens,
          distribution,
          k_gathered,
          v_gathered,
      ) = _setup_rpa_inputs(
          q_len=1,  # decode
          kv_len=kv_len,
          num_q_heads=num_q_heads,
          num_kv_heads=num_kv_heads,
          head_dim=head_dim,
          page_size=page_size,
          q_dtype=q_dtype,
          kv_dtype=kv_dtype,
          num_pages=num_pages,
          rng=rng,
      )
      rpa_inputs.append((
          q,
          k,
          v,
          kv_cache,
          kv_lens_arr,
          page_indices,
          cu_q_lens,
          distribution,
      ))

      # Extract data for MRIS
      mris_queries.append(q[:1, :num_q_heads_per_kv_head, :head_dim])
      # Use first KV head for MRIS comparison
      mris_k_pages.append(k_gathered[:, 0, :])  # [kv_len, head_dim]
      mris_v_pages.append(v_gathered[:, 0, :])

    # --- JITted RPA kernel-only (pure execution of 1 call) ---
    (
        q0,
        k0,
        v0,
        kv_cache0,
        kv_lens_arr0,
        page_indices0,
        cu_q_lens0,
        dist0,
    ) = rpa_inputs[0]

    @jax.jit
    def single_rpa_jit_fn(
        q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_lens, distribution
    ):
      out, _ = ragged_paged_attention(
          q,
          k,
          v,
          kv_cache,
          kv_lens_arr,
          page_indices,
          cu_q_lens,
          distribution,
          sm_scale=sm_scale,
      )
      return out

    log_trace("Starting sub-benchmark: RPA single")
    rpa_single_ms = _benchmark_fn(
        lambda: single_rpa_jit_fn(
            q0,
            k0,
            v0,
            kv_cache0,
            kv_lens_arr0,
            page_indices0,
            cu_q_lens0,
            dist0,
        )
    )

    # Prepare flat inputs for JITted sequential RPA
    rpa_flat_args = []
    for (
        q,
        k,
        v,
        kv_cache,
        kv_lens_arr,
        page_indices,
        cu_q_lens,
        distribution,
    ) in rpa_inputs:
      rpa_flat_args.extend([
          q,
          k,
          v,
          kv_cache,
          kv_lens_arr,
          page_indices,
          cu_q_lens,
          distribution,
      ])

    @jax.jit
    def sequential_rpa_jit_fn(*args):
      results = []
      for i in range(batch_size):
        offset = i * 8
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens_arr,
            page_indices,
            cu_q_lens,
            distribution,
        ) = args[offset : offset + 8]
        out = single_rpa_jit_fn(
            q,
            k,
            v,
            kv_cache,
            kv_lens_arr,
            page_indices,
            cu_q_lens,
            distribution,
        )
        results.append(out)
      return results

    # --- JITted Sequential RPA baseline ---
    log_trace("Starting sub-benchmark: RPA sequential")
    rpa_ms = _benchmark_fn(lambda: sequential_rpa_jit_fn(*rpa_flat_args))

    # --- MRIS V3c (2D Pallas Grid Fused, JITted) ---
    @jax.jit
    def mris_v3c_jit_fn(*args):
      queries_arg = list(args[0:batch_size])
      k_pages_arg = list(args[batch_size : 2 * batch_size])
      v_pages_arg = list(args[2 * batch_size : 3 * batch_size])
      return mris_fused_attention_v3c(
          queries_arg,
          k_pages_arg,
          v_pages_arg,
          kv_lens,
          sm_scale=sm_scale,
          bkv_sz=128,
      )

    mris_flat_args = []
    mris_flat_args.extend(mris_queries)
    mris_flat_args.extend(mris_k_pages)
    mris_flat_args.extend(mris_v_pages)

    log_trace("Starting sub-benchmark: MRIS V3c")
    mris_v3c_ms = _benchmark_fn(lambda: mris_v3c_jit_fn(*mris_flat_args))

    # --- In-Kernel Paged MRIS V3 (Pallas DMA Paged Prefetching) ---
    paged_kv_cache0 = rpa_inputs[0][3]
    pages_per_seq = cdiv(kv_len, page_size)
    paged_page_indices = jnp.stack(
        [inp[5][:pages_per_seq] for inp in rpa_inputs], axis=0
    )

    @jax.jit
    def in_kernel_paged_mris_jit_fn(q_args, kv_cache_arg, page_indices_arg):
      return mris_fused_paged_attention_v3(
          q_args,
          kv_cache_arg,
          kv_lens,
          page_indices_arg,
          sm_scale=sm_scale,
          bkv_sz=128,
          page_size=page_size,
      )

    log_trace("Starting sub-benchmark: In-Kernel Paged MRIS V3")
    try:
      in_kernel_paged_ms = _benchmark_fn(
          lambda: in_kernel_paged_mris_jit_fn(
              mris_queries, paged_kv_cache0, paged_page_indices
          )
      )
    except Exception as e:
      log_trace(f"In-Kernel Paged MRIS error: {e}")
      in_kernel_paged_ms = -1.0

    # --- Hybrid Paged MRIS Dispatcher (MRIS Bulk 4-way + RPA Tail) ---
    num_mris_reqs = (batch_size // 4) * 4
    num_tail_reqs = batch_size % 4

    @jax.jit
    def hybrid_paged_mris_jit_fn(
        q_args, kv_cache_arg, page_indices_arg, *tail_args
    ):
      outputs = []
      if num_mris_reqs > 0:
        mris_outs = mris_fused_paged_attention_v3(
            q_args[:num_mris_reqs],
            kv_cache_arg,
            kv_lens[:num_mris_reqs],
            page_indices_arg[:num_mris_reqs],
            sm_scale=sm_scale,
            bkv_sz=128,
            page_size=page_size,
        )
        outputs.extend(mris_outs)

      for i in range(num_tail_reqs):
        offset = i * 8
        (
            q,
            k,
            v,
            cache,
            kv_lens_arr,
            page_indices_single,
            cu_q_lens,
            distribution,
        ) = tail_args[offset : offset + 8]
        out = single_rpa_jit_fn(
            q,
            k,
            v,
            cache,
            kv_lens_arr,
            page_indices_single,
            cu_q_lens,
            distribution,
        )
        outputs.append(out)

      return outputs

    tail_flat_args = []
    for i in range(num_mris_reqs, batch_size):
      tail_flat_args.extend(rpa_inputs[i])

    log_trace("Starting sub-benchmark: Hybrid Paged MRIS Dispatcher")
    try:
      hybrid_paged_ms = _benchmark_fn(
          lambda: hybrid_paged_mris_jit_fn(
              mris_queries,
              paged_kv_cache0,
              paged_page_indices,
              *tail_flat_args,
          )
      )
    except Exception as e:
      log_trace(f"Hybrid Paged MRIS error: {e}")
      hybrid_paged_ms = -1.0

    mris_ms = mris_v3c_ms

    # Report — detailed breakdown
    report = _format_report(
        label,
        head_dim,
        num_q_heads,
        num_kv_heads,
        kv_lens,
        rpa_ms,
        mris_ms,
    )
    v3c_speedup = rpa_ms / mris_v3c_ms if mris_v3c_ms > 0 else float("inf")
    paged_speedup = (
        rpa_ms / in_kernel_paged_ms if in_kernel_paged_ms > 0 else float("inf")
    )
    hybrid_speedup = (
        rpa_ms / hybrid_paged_ms if hybrid_paged_ms > 0 else float("inf")
    )
    report += f"\n  --- Detailed JITted Breakdown ---"
    report += f"\n  Production Paged RPA V3 (JIT):     {rpa_ms:10.3f} ms"
    report += (
        f"\n  MRIS Kernel Contiguous (JIT):       {mris_v3c_ms:10.3f} ms  (vs"
        f" Prod RPA: {v3c_speedup:.2f}x)"
    )
    report += (
        f"\n  In-Kernel Paged MRIS V3 (JIT):      {in_kernel_paged_ms:10.3f} ms"
        f"  (vs Prod RPA: {paged_speedup:.2f}x)"
    )
    report += (
        f"\n  Hybrid Paged MRIS Dispatcher (JIT): {hybrid_paged_ms:10.3f} ms"
        f"  (vs Prod RPA: {hybrid_speedup:.2f}x)"
    )
    print(report)
    log_trace(f"[REPORT OUTPUT]\n{report}")


if __name__ == "__main__":
  absltest.main()
