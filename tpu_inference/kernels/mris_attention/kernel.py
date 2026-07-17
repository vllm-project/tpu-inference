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
"""Multi-Request Interleaved Streaming (MRIS) fused attention kernel.

Implements fused multi-request decode attention for TPU, processing N_FUSED=4
decode requests in a single MXU pass. Includes:

  - V3c: VPU-interleaved kernel with async DMA double-buffering and native
    Pallas batch grid tiling for arbitrary batch sizes.
  - V3 Paged: In-kernel paged KV cache prefetching via scattered HBM page
    lookups, directly consuming the production paged KV cache table.
  - Reference implementation (pure JAX) for correctness testing.
"""

from __future__ import annotations

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    cdiv,
    get_tpu_version,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FUSED = 4  # Number of decode requests fused into one MXU pass.


# ---------------------------------------------------------------------------
# Reference implementation (pure JAX, no Pallas)
# ---------------------------------------------------------------------------
def ref_mris_fused_attention(
    queries: list[jax.Array],
    k_pages: list[jax.Array],
    v_pages: list[jax.Array],
    kv_lens: list[int],
    *,
    sm_scale: float = 1.0,
    mask_value: float | None = None,
) -> list[jax.Array]:
  """Reference implementation: processes each request independently."""
  num_reqs = len(queries)
  assert len(k_pages) == num_reqs
  assert len(v_pages) == num_reqs

  if mask_value is None:
    mask_value = jnp.finfo(jnp.float32).min

  outputs = []
  for i in range(num_reqs):
    q = queries[i]
    if kv_lens[i] == 0:
      outputs.append(jnp.zeros_like(q))
      continue
    k = k_pages[i][: kv_lens[i]]
    v = v_pages[i][: kv_lens[i]]
    num_q_heads = q.shape[1]
    head_dim = q.shape[2]
    q = jnp.transpose(q, (1, 0, 2))
    s = (
        jnp.matmul(
            q,
            k.T[jnp.newaxis, :, :],
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.matmul(
        p.astype(v.dtype),
        v[jnp.newaxis, :, :],
        preferred_element_type=jnp.float32,
    )
    outputs.append(jnp.transpose(o, (1, 0, 2)))
  return outputs


# ---------------------------------------------------------------------------
# V3c: VPU-interleaved Kernel (separate HBM inputs, stacked in VMEM via VPU)
# ---------------------------------------------------------------------------


def _mris_kernel_body_v3c(
    # Scalar prefetch (SMEM)
    kv_lens_ref,  # [N_FUSED]
    # HBM inputs (4 separate K, 4 separate V per chunk)
    q_hbm_ref,  # [1, total_q_rows, head_dim]
    k0_hbm,
    k1_hbm,
    k2_hbm,
    k3_hbm,
    v0_hbm,
    v1_hbm,
    v2_hbm,
    v3_hbm,
    # HBM output
    o_hbm_ref,
    # VMEM scratch
    q_vmem,
    acc_vmem,
    l_vmem,
    m_vmem,
    # 4 separate double buffers for K and V
    k0_buf_x2,
    k1_buf_x2,
    k2_buf_x2,
    k3_buf_x2,  # [2, bkv_sz, head_dim] each
    v0_buf_x2,
    v1_buf_x2,
    v2_buf_x2,
    v3_buf_x2,  # [2, bkv_sz, head_dim] each
    bo_vmem,
    sems,  # [1, 2] — only need 1 sem row for K/V double buffering
    q_sem_ref,  # [1] for Q
    o_sem_ref,  # [1] for O
    *,
    sm_scale: float,
    bkv_sz: int,
    num_q_heads_per_kv_head: int,
    mask_value: float = -1e30,
):
  total_q_rows = q_hbm_ref.shape[1]
  head_dim = q_hbm_ref.shape[2]
  max_kv_len = k0_hbm.shape[1]
  num_bkv = cdiv(max_kv_len, bkv_sz)

  def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def fetch_kv_v3c(bkv_idx, buf_idx, wait):
    """Contiguous DMAs into VMEM double buffers (start or wait)."""
    kv_start = bkv_idx * bkv_sz
    sem = sems.at[0, buf_idx]

    _async_copy(
        k0_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        k0_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        k1_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        k1_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        k2_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        k2_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        k3_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        k3_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        v0_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        v0_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        v1_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        v1_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        v2_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        v2_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )
    _async_copy(
        v3_hbm.at[0, pl.ds(kv_start, bkv_sz)],
        v3_buf_x2.at[buf_idx],
        sem,
        wait=wait,
    )

  # ---- Stage 1: DMA Q ----
  _async_copy(q_hbm_ref.at[0], q_vmem, q_sem_ref.at[0], wait=False)
  _async_copy(q_hbm_ref.at[0], q_vmem, q_sem_ref.at[0], wait=True)

  # ---- Stage 2: Init acc ----
  acc_vmem[...] = jnp.zeros_like(acc_vmem)
  l_vmem[...] = jnp.zeros_like(l_vmem)
  m_vmem[...] = jnp.full_like(m_vmem, -jnp.inf)

  # ---- Stage 3: Prologue ----
  fetch_kv_v3c(bkv_idx=0, buf_idx=0, wait=False)

  # ---- Stage 4: KV loop ----
  @pl.loop(0, num_bkv, unroll=False)
  def kv_loop(bkv_idx):
    buf_idx = bkv_idx % 2
    next_buf = (bkv_idx + 1) % 2
    kv_start = bkv_idx * bkv_sz

    @pl.when(bkv_idx + 1 < num_bkv)
    def prefetch_next():
      fetch_kv_v3c(bkv_idx + 1, next_buf, wait=False)

    fetch_kv_v3c(bkv_idx, buf_idx, wait=True)

    # VPU stack: interleave the 4 buffers in registers
    # Each buffer is [bkv_sz, head_dim] -> stack to [bkv_sz, N_FUSED, head_dim]
    k_block_3d = jnp.stack(
        [
            k0_buf_x2[buf_idx],
            k1_buf_x2[buf_idx],
            k2_buf_x2[buf_idx],
            k3_buf_x2[buf_idx],
        ],
        axis=1,
    )
    v_block_3d = jnp.stack(
        [
            v0_buf_x2[buf_idx],
            v1_buf_x2[buf_idx],
            v2_buf_x2[buf_idx],
            v3_buf_x2[buf_idx],
        ],
        axis=1,
    )

    chunk_idx = pl.program_id(0)
    for req_idx in range(N_FUSED):
      req_kv_len = kv_lens_ref[chunk_idx, req_idx]
      row_start = req_idx * num_q_heads_per_kv_head
      row_end = row_start + num_q_heads_per_kv_head

      q_req = q_vmem[row_start:row_end]
      k_req = k_block_3d[:, req_idx, :]

      # QK^T
      s = (
          jnp.matmul(q_req, k_req.T, preferred_element_type=jnp.float32)
          * sm_scale
      )

      # Masking
      kv_positions = kv_start + jnp.arange(bkv_sz)
      mask = kv_positions[jnp.newaxis, :] < req_kv_len
      s = jnp.where(mask, s, mask_value)

      # Softmax
      s_rowmax = jnp.max(s, axis=1, keepdims=True)
      m_prev = m_vmem[row_start:row_end]
      m_curr = jnp.maximum(m_prev, s_rowmax)
      m_vmem[row_start:row_end] = m_curr

      p = jnp.exp(s - jnp.broadcast_to(m_curr[:, :1], s.shape))
      p_rowsum = jnp.sum(p, axis=1, keepdims=True)

      exp_m_diff = jnp.exp(m_prev - m_curr)
      l_prev = l_vmem[row_start:row_end]
      l_vmem[row_start:row_end] = exp_m_diff * l_prev + p_rowsum

      # PV
      v_req = v_block_3d[:, req_idx, :]
      v_req = jnp.where(kv_positions[:, jnp.newaxis] < req_kv_len, v_req, 0.0)
      pv = jnp.matmul(p, v_req, preferred_element_type=jnp.float32)

      acc_prev = acc_vmem[row_start:row_end]
      exp_m_diff_broad = jnp.broadcast_to(exp_m_diff[:, :1], acc_prev.shape)
      acc_vmem[row_start:row_end] = exp_m_diff_broad * acc_prev + pv

  # ---- Stage 5: Normalization + send output ----
  acc = acc_vmem[...]
  l_val = l_vmem[...]
  l_broad = jnp.broadcast_to(l_val[:, :1], acc.shape)
  out = (acc / l_broad).astype(o_hbm_ref.dtype)

  bo_vmem[...] = out
  _async_copy(bo_vmem, o_hbm_ref.at[0], o_sem_ref.at[0], wait=False)
  _async_copy(bo_vmem, o_hbm_ref.at[0], o_sem_ref.at[0], wait=True)


def mris_fused_attention_v3c(
    queries: list[jax.Array],
    k_pages: list[jax.Array],
    v_pages: list[jax.Array],
    kv_lens: jax.Array,
    *,
    sm_scale: float = 1.0,
    bkv_sz: int = 128,
) -> list[jax.Array]:
  """MRIS fused attention with native Pallas batch grid tiling (grid=(num_chunks,))."""
  num_reqs = len(queries)
  num_q_heads_per_kv_head = queries[0].shape[1]
  head_dim = queries[0].shape[2]
  max_kv_len = int(kv_lens.max()) if num_reqs > 0 else 0
  max_kv_len_aligned = max(cdiv(max_kv_len, bkv_sz) * bkv_sz, bkv_sz)
  total_q_rows = N_FUSED * num_q_heads_per_kv_head
  num_bkv = max_kv_len_aligned // bkv_sz
  num_chunks = cdiv(num_reqs, N_FUSED)

  # Pad to integer multiple of N_FUSED (4)
  q_list = list(queries)
  k_list = list(k_pages)
  v_list = list(v_pages)
  pad_count = num_chunks * N_FUSED - num_reqs

  while len(q_list) < num_chunks * N_FUSED:
    q_list.append(jnp.zeros_like(q_list[0]))
    k_list.append(k_list[0])
    v_list.append(v_list[0])

  if pad_count > 0:
    kv_lens = jnp.pad(kv_lens, (0, pad_count), constant_values=0)

  # Pack 3D batched inputs: shape [num_chunks, ...]
  q_chunks = []
  k0_chunks, k1_chunks, k2_chunks, k3_chunks = [], [], [], []
  v0_chunks, v1_chunks, v2_chunks, v3_chunks = [], [], [], []
  lens_chunks = []

  for c in range(num_chunks):
    offset = c * N_FUSED
    c_queries = q_list[offset : offset + N_FUSED]
    c_k = k_list[offset : offset + N_FUSED]
    c_v = v_list[offset : offset + N_FUSED]

    q_chunks.append(
        jnp.concatenate(
            [q.reshape(num_q_heads_per_kv_head, head_dim) for q in c_queries],
            axis=0,
        )
    )
    k0_chunks.append(c_k[0][:max_kv_len_aligned])
    k1_chunks.append(c_k[1][:max_kv_len_aligned])
    k2_chunks.append(c_k[2][:max_kv_len_aligned])
    k3_chunks.append(c_k[3][:max_kv_len_aligned])

    v0_chunks.append(c_v[0][:max_kv_len_aligned])
    v1_chunks.append(c_v[1][:max_kv_len_aligned])
    v2_chunks.append(c_v[2][:max_kv_len_aligned])
    v3_chunks.append(c_v[3][:max_kv_len_aligned])

    lens_chunks.append(kv_lens[offset : offset + N_FUSED])

  q_batched = jnp.stack(
      q_chunks, axis=0
  )  # [num_chunks, total_q_rows, head_dim]
  k0_batched = jnp.stack(
      k0_chunks, axis=0
  )  # [num_chunks, max_kv_aligned, head_dim]
  k1_batched = jnp.stack(k1_chunks, axis=0)
  k2_batched = jnp.stack(k2_chunks, axis=0)
  k3_batched = jnp.stack(k3_chunks, axis=0)

  v0_batched = jnp.stack(v0_chunks, axis=0)
  v1_batched = jnp.stack(v1_chunks, axis=0)
  v2_batched = jnp.stack(v2_chunks, axis=0)
  v3_batched = jnp.stack(v3_chunks, axis=0)

  kv_lens_batched = jnp.stack(lens_chunks, axis=0)  # [num_chunks, 4]

  kernel = _get_mris_v3c_kernel(
      num_q_heads_per_kv_head,
      head_dim,
      num_bkv,
      bkv_sz,
      sm_scale,
      q_batched.dtype,
      k0_batched.dtype,
      num_chunks=num_chunks,
  )

  # Single Pallas call executing grid=(num_chunks,) natively on TPU
  result_batched = kernel(
      kv_lens_batched,
      q_batched,
      k0_batched,
      k1_batched,
      k2_batched,
      k3_batched,
      v0_batched,
      v1_batched,
      v2_batched,
      v3_batched,
  )

  # Unpack valid requests
  outputs = []
  for i in range(num_reqs):
    c = i // N_FUSED
    req_idx = i % N_FUSED
    row_start = req_idx * num_q_heads_per_kv_head
    row_end = row_start + num_q_heads_per_kv_head
    outputs.append(result_batched[c, row_start:row_end][jnp.newaxis, :, :])

  return outputs


@functools.lru_cache(maxsize=32)
def _get_mris_v3c_kernel(
    num_q_heads_per_kv_head: int,
    head_dim: int,
    num_bkv: int,
    bkv_sz: int,
    sm_scale: float,
    q_dtype,
    kv_dtype,
    num_chunks: int = 1,
):
  total_q_rows = N_FUSED * num_q_heads_per_kv_head

  tpu_version = get_tpu_version()
  out_shape = (
      pltpu.HBM(shape=(num_chunks, total_q_rows, head_dim), dtype=q_dtype)
      if tpu_version >= 7
      else jax.ShapeDtypeStruct(
          shape=(num_chunks, total_q_rows, head_dim), dtype=q_dtype
      )
  )

  # Scratch
  q_scratch = pltpu.VMEM((total_q_rows, head_dim), q_dtype)
  acc_scratch = pltpu.VMEM((total_q_rows, head_dim), jnp.float32)
  l_scratch = pltpu.VMEM((total_q_rows, 128), jnp.float32)
  m_scratch = pltpu.VMEM((total_q_rows, 128), jnp.float32)

  # 4 separate double buffers for K and V
  k0_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)
  k1_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)
  k2_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)
  k3_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)

  v0_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)
  v1_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)
  v2_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)
  v3_db = pltpu.VMEM((2, bkv_sz, head_dim), kv_dtype)

  bo_scratch = pltpu.VMEM((total_q_rows, head_dim), q_dtype)

  # Semaphores
  dma_sems = pltpu.SemaphoreType.DMA((1, 2))  # Row 0: K/V
  q_sem = pltpu.SemaphoreType.DMA((1,))  # One-shot Q
  o_sem = pltpu.SemaphoreType.DMA((1,))  # One-shot O

  scratch_shapes = [
      q_scratch,
      acc_scratch,
      l_scratch,
      m_scratch,
      k0_db,
      k1_db,
      k2_db,
      k3_db,
      v0_db,
      v1_db,
      v2_db,
      v3_db,
      bo_scratch,
      dma_sems,
      q_sem,
      o_sem,
  ]

  kernel_fn = functools.partial(
      _mris_kernel_body_v3c,
      sm_scale=sm_scale,
      bkv_sz=bkv_sz,
      num_q_heads_per_kv_head=num_q_heads_per_kv_head,
  )

  max_kv_len_aligned = num_bkv * bkv_sz
  block_spec_hbm = pl.BlockSpec(
      index_map=lambda c, *_: (c, 0, 0), block_shape=(1, total_q_rows, head_dim)
  )
  block_spec_kv = pl.BlockSpec(
      index_map=lambda c, *_: (c, 0, 0),
      block_shape=(1, max_kv_len_aligned, head_dim),
  )

  return pl.pallas_call(
      kernel_fn,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,  # kv_lens_batched
          in_specs=[
              block_spec_hbm,  # q_fused
              block_spec_kv,  # k0
              block_spec_kv,  # k1
              block_spec_kv,  # k2
              block_spec_kv,  # k3
              block_spec_kv,  # v0
              block_spec_kv,  # v1
              block_spec_kv,  # v2
              block_spec_kv,  # v3
          ],
          out_specs=block_spec_hbm,
          grid=(num_chunks,),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=100 * 1024 * 1024,
          disable_bounds_checks=True,
          disable_semaphore_checks=True,
      ),
      out_shape=out_shape,
      name="mris_fused_attention_v3c",
  )


# ---------------------------------------------------------------------------
# V3 Paged: In-Kernel Paged Cache Prefetching (scattered physical HBM pages)
# ---------------------------------------------------------------------------


def _mris_paged_kernel_body_v3(
    # Prefetches (SMEM)
    kv_lens_ref,  # [num_chunks, N_FUSED]
    page_indices_ref,  # [num_chunks, N_FUSED, max_pages_per_seq]
    # HBM inputs
    q_hbm_ref,  # [num_chunks, total_q_rows, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // packing, packing, head_dim]
    # HBM output
    o_hbm_ref,
    # VMEM scratch
    q_vmem,
    acc_vmem,
    l_vmem,
    m_vmem,
    kv0_buf_x2,
    kv1_buf_x2,
    kv2_buf_x2,
    kv3_buf_x2,  # [2, bkv_sz, 2, 1, head_dim] each
    bo_vmem,
    sems,
    q_sem_ref,
    o_sem_ref,
    *,
    sm_scale: float,
    bkv_sz: int,
    page_size: int,
    num_q_heads_per_kv_head: int,
    mask_value: float = -1e30,
):
  chunk_idx = pl.program_id(0)
  total_q_rows = q_hbm_ref.shape[1]
  head_dim = q_hbm_ref.shape[2]
  pages_per_seq = page_indices_ref.shape[2]
  max_kv_len = pages_per_seq * page_size
  num_bkv = cdiv(max_kv_len, bkv_sz)
  pages_per_tile = bkv_sz // page_size
  num_pages_total = kv_cache_hbm_ref.shape[0]
  cache_shape = kv_cache_hbm_ref.shape
  flat_cache_hbm_ref = kv_cache_hbm_ref.reshape(
      cache_shape[0] * cache_shape[1], *cache_shape[2:]
  )

  kv_bufs = [kv0_buf_x2, kv1_buf_x2, kv2_buf_x2, kv3_buf_x2]

  def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def fetch_paged_kv(bkv_idx, buf_idx, wait):
    sem = sems.at[0, buf_idx]
    kv_start = bkv_idx * bkv_sz

    for req_idx in range(N_FUSED):
      req_kv_len = kv_lens_ref[chunk_idx, req_idx]
      kv_buf = kv_bufs[req_idx]

      for p in range(pages_per_tile):
        page_kv_start = kv_start + p * page_size
        sz = jnp.clip(req_kv_len - page_kv_start, 0, page_size)
        page_slot = bkv_idx * pages_per_tile + p
        page_idx_offset = jnp.minimum(page_slot, pages_per_seq - 1)
        phys_page_idx = jnp.minimum(
            page_indices_ref[chunk_idx, req_idx, page_idx_offset],
            num_pages_total - 1,
        )

        dst_kv = kv_buf.at[buf_idx, pl.ds(p * page_size, page_size)]
        token_offset = phys_page_idx * page_size
        src_kv = flat_cache_hbm_ref.at[pl.ds(token_offset, page_size)]

        _async_copy(src_kv, dst_kv, sem, wait)

  # Stage 1: Q DMA
  _async_copy(q_hbm_ref.at[0], q_vmem, q_sem_ref.at[0], wait=False)
  _async_copy(q_hbm_ref.at[0], q_vmem, q_sem_ref.at[0], wait=True)

  # Stage 2: Init accumulators
  acc_vmem[...] = jnp.zeros_like(acc_vmem)
  l_vmem[...] = jnp.zeros_like(l_vmem)
  m_vmem[...] = jnp.full_like(m_vmem, -jnp.inf)

  # Stage 3: Prologue
  fetch_paged_kv(bkv_idx=0, buf_idx=0, wait=False)

  # Stage 4: Loop
  @pl.loop(0, num_bkv, unroll=False)
  def kv_loop(bkv_idx):
    buf_idx = bkv_idx % 2
    next_buf = (bkv_idx + 1) % 2
    kv_start = bkv_idx * bkv_sz

    @pl.when(bkv_idx + 1 < num_bkv)
    def prefetch_next():
      fetch_paged_kv(bkv_idx + 1, next_buf, wait=False)

    fetch_paged_kv(bkv_idx, buf_idx, wait=True)

    k_block_3d = jnp.stack(
        [
            kv0_buf_x2[buf_idx, :, 0, 0, :],
            kv1_buf_x2[buf_idx, :, 0, 0, :],
            kv2_buf_x2[buf_idx, :, 0, 0, :],
            kv3_buf_x2[buf_idx, :, 0, 0, :],
        ],
        axis=1,
    )
    v_block_3d = jnp.stack(
        [
            kv0_buf_x2[buf_idx, :, 0, 1, :],
            kv1_buf_x2[buf_idx, :, 0, 1, :],
            kv2_buf_x2[buf_idx, :, 0, 1, :],
            kv3_buf_x2[buf_idx, :, 0, 1, :],
        ],
        axis=1,
    )

    for req_idx in range(N_FUSED):
      req_kv_len = kv_lens_ref[chunk_idx, req_idx]
      row_start = req_idx * num_q_heads_per_kv_head
      row_end = row_start + num_q_heads_per_kv_head

      q_req = q_vmem[row_start:row_end]
      k_req = k_block_3d[:, req_idx, :]

      s = (
          jnp.matmul(q_req, k_req.T, preferred_element_type=jnp.float32)
          * sm_scale
      )
      kv_positions = kv_start + jnp.arange(bkv_sz)
      mask = kv_positions[jnp.newaxis, :] < req_kv_len
      s = jnp.where(mask, s, mask_value)

      s_rowmax = jnp.max(s, axis=1, keepdims=True)
      m_prev = m_vmem[row_start:row_end]
      m_curr = jnp.maximum(m_prev, s_rowmax)
      m_vmem[row_start:row_end] = m_curr

      p = jnp.exp(s - jnp.broadcast_to(m_curr[:, :1], s.shape))
      p_rowsum = jnp.sum(p, axis=1, keepdims=True)

      exp_m_diff = jnp.exp(m_prev - m_curr)
      l_prev = l_vmem[row_start:row_end]
      l_vmem[row_start:row_end] = exp_m_diff * l_prev + p_rowsum

      v_req = v_block_3d[:, req_idx, :]
      v_req = jnp.where(kv_positions[:, jnp.newaxis] < req_kv_len, v_req, 0.0)
      pv = jnp.matmul(p, v_req, preferred_element_type=jnp.float32)

      acc_prev = acc_vmem[row_start:row_end]
      exp_m_diff_broad = jnp.broadcast_to(exp_m_diff[:, :1], acc_prev.shape)
      acc_vmem[row_start:row_end] = exp_m_diff_broad * acc_prev + pv

  # Stage 5: Output
  acc = acc_vmem[...]
  l_val = l_vmem[...]
  l_broad = jnp.broadcast_to(l_val[:, :1], acc.shape)
  out = (acc / l_broad).astype(o_hbm_ref.dtype)

  bo_vmem[...] = out
  _async_copy(bo_vmem, o_hbm_ref.at[0], o_sem_ref.at[0], wait=False)
  _async_copy(bo_vmem, o_hbm_ref.at[0], o_sem_ref.at[0], wait=True)


def mris_fused_paged_attention_v3(
    queries: list[jax.Array],
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    *,
    sm_scale: float = 1.0,
    bkv_sz: int = 128,
    page_size: int = 16,
) -> list[jax.Array]:
  """MRIS fused attention directly fetching from Paged KV Cache table in HBM.

  Args:
    queries: List of per-request query arrays, each [1, num_q_heads, head_dim].
    kv_cache: Paged KV cache in HBM, shape [num_pages, page_size, ...].
    kv_lens: JAX array of KV lengths per request, shape [num_reqs] int32.
    page_indices: Page table indices, shape [num_reqs, pages_per_seq] int32.
    sm_scale: Softmax scale factor.
    bkv_sz: KV block size for tiling.
    page_size: Size of each page in the paged KV cache.

  Returns:
    List of per-request output arrays, each [1, num_q_heads, head_dim].
  """
  num_reqs = len(queries)
  num_q_heads_per_kv_head = queries[0].shape[1]
  head_dim = queries[0].shape[2]
  num_chunks = cdiv(num_reqs, N_FUSED)
  pages_per_seq = page_indices.shape[1]
  total_q_rows = N_FUSED * num_q_heads_per_kv_head

  # Pad queries and kv_lens to multiple of N_FUSED using JAX ops (shard_map safe)
  pad_count = num_chunks * N_FUSED - num_reqs
  q_list = list(queries)
  while len(q_list) < num_chunks * N_FUSED:
    q_list.append(jnp.zeros_like(q_list[0]))

  # Pad kv_lens array with zeros (masked out by online softmax)
  if pad_count > 0:
    kv_lens = jnp.pad(kv_lens, (0, pad_count), constant_values=0)
    page_indices = jnp.pad(
        page_indices,
        ((0, pad_count), (0, 0)),
        constant_values=0,
    )

  q_chunks = []
  lens_chunks = []
  page_indices_chunks = []

  for c in range(num_chunks):
    offset = c * N_FUSED
    c_queries = q_list[offset : offset + N_FUSED]

    q_chunks.append(
        jnp.concatenate(
            [q.reshape(num_q_heads_per_kv_head, head_dim) for q in c_queries],
            axis=0,
        )
    )
    lens_chunks.append(kv_lens[offset : offset + N_FUSED])
    page_indices_chunks.append(page_indices[offset : offset + N_FUSED])

  q_batched = jnp.stack(q_chunks, axis=0)
  kv_lens_batched = jnp.stack(lens_chunks, axis=0)
  page_indices_batched = jnp.stack(page_indices_chunks, axis=0)

  kernel = _get_mris_paged_v3_kernel(
      num_q_heads_per_kv_head,
      head_dim,
      pages_per_seq,
      bkv_sz,
      page_size,
      sm_scale,
      q_batched.dtype,
      kv_cache.dtype,
      num_chunks=num_chunks,
  )

  result_batched = kernel(
      kv_lens_batched,
      page_indices_batched,
      q_batched,
      kv_cache,
  )

  outputs = []
  for i in range(num_reqs):
    c = i // N_FUSED
    req_idx = i % N_FUSED
    row_start = req_idx * num_q_heads_per_kv_head
    row_end = row_start + num_q_heads_per_kv_head
    outputs.append(result_batched[c, row_start:row_end][jnp.newaxis, :, :])

  return outputs


@functools.lru_cache(maxsize=32)
def _get_mris_paged_v3_kernel(
    num_q_heads_per_kv_head: int,
    head_dim: int,
    pages_per_seq: int,
    bkv_sz: int,
    page_size: int,
    sm_scale: float,
    q_dtype,
    kv_dtype,
    num_chunks: int = 1,
):
  total_q_rows = N_FUSED * num_q_heads_per_kv_head
  tpu_version = get_tpu_version()

  out_shape = (
      pltpu.HBM(shape=(num_chunks, total_q_rows, head_dim), dtype=q_dtype)
      if tpu_version >= 7
      else jax.ShapeDtypeStruct(
          shape=(num_chunks, total_q_rows, head_dim), dtype=q_dtype
      )
  )

  q_scratch = pltpu.VMEM((total_q_rows, head_dim), q_dtype)
  acc_scratch = pltpu.VMEM((total_q_rows, head_dim), jnp.float32)
  l_scratch = pltpu.VMEM((total_q_rows, 128), jnp.float32)
  m_scratch = pltpu.VMEM((total_q_rows, 128), jnp.float32)

  # 4 separate 4D double buffers for KV pages [2, bkv_sz, 1, 2, head_dim]
  kv0_db = pltpu.VMEM((2, bkv_sz, 1, 2, head_dim), kv_dtype)
  kv1_db = pltpu.VMEM((2, bkv_sz, 1, 2, head_dim), kv_dtype)
  kv2_db = pltpu.VMEM((2, bkv_sz, 1, 2, head_dim), kv_dtype)
  kv3_db = pltpu.VMEM((2, bkv_sz, 1, 2, head_dim), kv_dtype)

  bo_scratch = pltpu.VMEM((total_q_rows, head_dim), q_dtype)

  dma_sems = pltpu.SemaphoreType.DMA((1, 2))
  q_sem = pltpu.SemaphoreType.DMA((1,))
  o_sem = pltpu.SemaphoreType.DMA((1,))

  scratch_shapes = [
      q_scratch,
      acc_scratch,
      l_scratch,
      m_scratch,
      kv0_db,
      kv1_db,
      kv2_db,
      kv3_db,
      bo_scratch,
      dma_sems,
      q_sem,
      o_sem,
  ]

  kernel_fn = functools.partial(
      _mris_paged_kernel_body_v3,
      sm_scale=sm_scale,
      bkv_sz=bkv_sz,
      page_size=page_size,
      num_q_heads_per_kv_head=num_q_heads_per_kv_head,
  )

  block_spec_q = pl.BlockSpec(
      index_map=lambda c, *_: (c, 0, 0), block_shape=(1, total_q_rows, head_dim)
  )

  return pl.pallas_call(
      kernel_fn,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,  # kv_lens_batched, page_indices_batched
          in_specs=[
              block_spec_q,  # q_batched
              pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache (full HBM)
          ],
          out_specs=block_spec_q,
          grid=(num_chunks,),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=100 * 1024 * 1024,
          disable_bounds_checks=True,
          disable_semaphore_checks=True,
      ),
      out_shape=out_shape,
      name="mris_fused_paged_attention_v3",
  )
