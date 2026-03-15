# Copyright 2025 Google LLC
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
"""TPU-Friendly and Data-Movement-Friendly MLA Ragged Paged Attention kernel.

#bouache [KV-Cache Fusion] What was changed and why
==========================================
Originally the KV cache was updated by a *separate host-side JAX call*
(`update_kv_cache`) that ran **after** the attention kernel returned.
This meant the TPU had to:
  1. Finish all attention compute.
  2. Return to the host.
  3. Re-launch a second kernel to scatter new tokens into the paged cache.
  4. Only then start the next step.

The fusion eliminates step 3-4 by moving the KV cache write-back **inside**
the attention Pallas kernel itself, pipelined with attention compute:
  - A dedicated DMA channel (channel 3, semaphore index 3) is reserved for
    KV updates, separate from the bkv-fetch (0), bq-fetch (1), bo-send (2)
    channels that already existed.
  - Double-buffered VMEM staging buffers (`new_kv_c_x2_ref`,
    `new_k_pe_x2_ref`) are added so one B_q block's tokens can be staged
    while the next block's attention is running on the MXU.
  - After each B_q block's output is sent (`start_send_bo`), the kernel
    immediately fires a `start_update_kv_cache` for that block's tokens,
    overlapping the DMA scatter with the next block's MXU work.
  - The host-side `update_kv_cache()` pre-call is removed from
    `mla_ragged_paged_attention`; the updated cache is now returned directly
    as a kernel output via `input_output_aliases`.
  - `disable_bounds_checks=True` removes per-DMA software bounds checks;
    safe because the scheduler guarantees valid page indices.

"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


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


@functools.partial(
    jax.jit,
    donate_argnames=("cache_kv"),
)
def update_kv_cache(
        new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
        new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
        cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim+r_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        distribution: jax.Array,  # i32[3]
) -> tuple[jax.Array, jax.Array]:
    """ #bouache : Update KV cache with new tokens."""
    actual_r_dim = new_k_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        new_k_pe = jnp.pad(new_k_pe, ((0, 0), (0, r_dim - actual_r_dim)),
                           constant_values=0)
    actual_lkv_dim = new_kv_c.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if actual_lkv_dim != lkv_dim:
        new_kv_c = jnp.pad(new_kv_c, ((0, 0), (0, lkv_dim - actual_lkv_dim)),
                           constant_values=0)
    kv_dim = r_dim + lkv_dim
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
                                     ..., :lkv_dim].set(new_kv_c[q_start + j])
            cache_kv_ = cache_kv_.at[page_idx, row, col, ...,
                                     lkv_dim:].set(new_k_pe[q_start + j])
            return cache_kv_

        return lax.fori_loop(0, q_len, token_loop_body, cache_kv)

    cache_kv = lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

    return cache_kv


def ref_mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):

    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    dynamic_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    updated_cache_kv = update_kv_cache(
        new_kv_c,
        new_k_pe,
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
    actual_r_dim = q_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        q_pe = jnp.pad(q_pe, ((0, 0), (0, 0), (0, r_dim - actual_r_dim)),
                       constant_values=0)

    q = jnp.concatenate([ql_nope, q_pe], axis=-1)
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = updated_cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    assert lkv_dim == ql_nope.shape[-1]
    assert r_dim == q_pe.shape[-1]
    assert lkv_dim + r_dim == updated_cache_kv.shape[-1]

    kv_c_cache = updated_cache_kv[..., :lkv_dim].reshape(
        total_num_pages, page_size, lkv_dim)
    k_pe_cache = updated_cache_kv[...,
                                  lkv_dim:].reshape(total_num_pages, page_size,
                                                    r_dim)

    outputs = []

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
        gathered_kv_c = kv_c_cache[
            indices]  # [num_pages_i, page_size, lkv_dim]
        gathered_k_pe = k_pe_cache[indices]  # [num_pages_i, page_size, r_dim]

        # Flatten pages to sequence
        flat_kv_c = gathered_kv_c.reshape(
            -1, lkv_dim)  # [num_pages_i * page_size, lkv_dim]
        flat_k_pe = gathered_k_pe.reshape(
            -1, r_dim)  # [num_pages_i * page_size, r_dim]

        # Prepare k and v for attention
        k_i = jnp.concatenate([flat_kv_c[:kv_len], flat_k_pe[:kv_len]],
                              axis=-1)  # [kv_len, lkv_dim+r_dim]
        v_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]

        # MQA attention:
        # q:[q_len, actual_num_q_heads, lkv_dim+r_dim]
        # k:[kv_len, lkv_dim+r_dim]
        # v:[kv_len, lkv_dim]
        # attn: [actual_num_q_heads, q_len, kv_len]
        attn = jnp.einsum("qnh,kh->nqk",
                          q_i,
                          k_i,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        # Causal mask
        q_span = kv_len - q_len + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn = jnp.where(mask, mask_value, attn)
        attn = jax.nn.softmax(attn, axis=-1).astype(v_i.dtype)

        # out_i: [q_len, actual_num_q_heads, lkv_dim]
        out_i = jnp.einsum("nqk,kl->qnl", attn, v_i).astype(q_i.dtype)
        if v_scale is not None:
            out_i *= v_scale
        outputs.append(out_i)

    return (
        jnp.concatenate(outputs, axis=0),
        updated_cache_kv,
    )


# Expect to run this validation during runtime.
def dynamic_validate_inputs(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA kernel dynamically."""
    static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )
    max_num_tokens = ql_nope.shape[0]
    total_num_pages = cache_kv.shape[0]
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (0 <= i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")

    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")

    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(
            f"Total q tokens {cu_q_lens[k]} must be <= {max_num_tokens=}.")
    for seq_idx in range(k):
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        if not (0 < q_len <= kv_len):
            raise ValueError(
                f"Require 0 < {q_len=} <= {kv_len=} at sequence {seq_idx}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {seq_idx} where"
                f" {kv_len=} and {page_size=}.")
        for p in range(page_cnt):
            page_idx = page_indices[seq_idx * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
                    f" {seq_idx} where {kv_len=} and {page_size=}.")


# Expect to run this validation during compile time.
def static_validate_inputs(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA kernel statically."""
    if len(ql_nope.shape) != 3:
        raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
    if len(q_pe.shape) != 3:
        raise ValueError(f"Expected 3D array for {q_pe.shape=}")
    if len(new_kv_c.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
    if len(new_k_pe.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError(
            f"Expected {ql_nope.shape[:2]=} to be equal to {q_pe.shape[:2]=}")
    if ql_nope.shape[0] != new_kv_c.shape[0]:
        raise ValueError(
            f"Expected {ql_nope.shape[0]=} to be equal to {new_kv_c.shape[0]=}"
        )
    if new_kv_c.shape[0] != new_k_pe.shape[0]:
        raise ValueError(
            f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}"
        )
    if ql_nope.shape[2] != new_kv_c.shape[1]:
        raise ValueError(
            f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}"
        )
    if q_pe.shape[2] != new_k_pe.shape[1]:
        raise ValueError(
            f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}")

    actual_lkv_dim = ql_nope.shape[2]
    actual_r_dim = q_pe.shape[2]
    lkv_dim = align_to(actual_lkv_dim, 128)
    r_dim = align_to(actual_r_dim, 128)

    (
        _,
        page_size_per_kv_packing,
        kv_packing,
        kv_dim,
    ) = cache_kv.shape

    if lkv_dim + r_dim != kv_dim:
        raise ValueError(
            f"Expected {lkv_dim=} + {r_dim=} to be equal to {kv_dim=}")

    if not (cache_kv.dtype == new_kv_c.dtype):
        raise ValueError(
            f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}.")
    if not (cache_kv.dtype == new_k_pe.dtype):
        raise ValueError(
            f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}.")

    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
        raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

    if kv_packing != get_dtype_packing(cache_kv.dtype):
        raise ValueError(
            f"{kv_packing=} does not match with {cache_kv.dtype=}")

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(
            cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3, ):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    page_size = page_size_per_kv_packing * kv_packing
    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_block is not None:
        if num_kv_pages_per_block <= 0:
            raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None:
        if num_queries_per_block <= 0:
            raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del debug_mode


def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    # TODO(jevinjiang): merge these into one so we can save SMEM.
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_bq_idx, bkv_sem_1_bq_idx, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    ql_nope_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim]
    q_pe_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, r_dim]
    new_kv_c_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim]
    updated_cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
    # Scratch
    bkvc_x2_ref,  # [2, bkv_sz_per_kv_packing, kv_packing, lkv_dim].
    bkpe_x2_ref,  # [2, bkv_sz_per_kv_packing, kv_packing, r_dim]
    bq_nope_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim]
    bq_rope_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim]
    # [KV-CACHE FUSION] Two new double-buffered VMEM staging buffers for
    # in-kernel KV write-back. Shape uses packed rows (÷ kv_packing) so that
    # Mosaic can prove DMA slice sizes are divisible by the fp8 tiling (4).
    new_kv_c_x2_ref,  # [2, bq_sz_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe_x2_ref,  # [2, bq_sz_per_kv_packing, kv_packing, r_dim]
    # [KV-CACHE FUSION] Semaphore array now has 4 channels (was 3):
    #   0 = bkv fetch   1 = bq fetch   2 = bo send   3 = KV-cache update (new)
    sems,  # [4, 2]
    l_ref,  # [bq_sz * num_q_heads, 128],
    m_ref,  # [bq_sz * num_q_heads, 128],
    acc_ref,  # [bq_sz * num_q_heads, lkv_dim],
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
    debug_mode: bool = False,
):
    assert ql_nope_hbm_ref.shape == o_hbm_ref.shape
    # Validation checks on the dimensions
    nope_dim = ql_nope_hbm_ref.shape[-1]
    pe_dim = q_pe_hbm_ref.shape[-1]
    assert nope_dim + pe_dim == cache_kv_hbm_ref.shape[-1]

    _, num_q_heads_per_q_packing, q_packing, lkv_dim = ql_nope_hbm_ref.shape
    r_dim = q_pe_hbm_ref.shape[-1]
    num_q_heads = num_q_heads_per_q_packing * q_packing
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = ql_nope_hbm_ref.dtype
    # Validate against the KV dtype.
    kv_dtype = cache_kv_hbm_ref.dtype
    assert q_pe_hbm_ref.dtype == q_dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert lkv_dim % 128 == 0
    assert r_dim % 128 == 0
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    page_size = page_size_per_kv_packing * kv_packing
    seq_idx = pl.program_id(0)
    num_seqs = pl.num_programs(0)
    decode_end = distribution_ref[0]
    prefill_end = distribution_ref[1]
    mixed_end = distribution_ref[2]

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    debug_print("[RPA debug] ======= In loop seq_idx={}", seq_idx)
    debug_print("[RPA debug] num_seqs={}", num_seqs)
    debug_print("[RPA debug] decode_end={}", decode_end)
    debug_print("[RPA debug] prefill_end={}", prefill_end)
    debug_print("[RPA debug] mixed_end={}", mixed_end)
    debug_print("[RPA debug] bkv_p={}", bkv_p)
    debug_print("[RPA debug] page_size={}", page_size)
    debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
    debug_print("[RPA debug] bkv_sz_per_kv_packing={}", bkv_sz_per_kv_packing)
    debug_print("[RPA debug] bq_sz={}", bq_sz)
    debug_print("[RPA debug] q_start={}", q_start)
    debug_print("[RPA debug] q_end={}", q_end)
    debug_print("[RPA debug] q_len={}", q_len)
    debug_print("[RPA debug] kv_len={}", kv_len)

    def flash_attention(
        ql_nope,  # [actual_bq_sz * num_q_heads, lkv_dim]
        q_pe,  # [actual_bq_sz * num_q_heads, r_dim]
        kv_c,  # [bkv_sz, lkv_dim] <- Correspond to data from bkvc_x2_ref
        k_pe,  # [bkv_sz, r_dim] <- Correspond to data from bpe_x2_ref
        *,
        bq_idx,
        bkv_idx,
    ):
        assert len(ql_nope.shape) == 2
        assert len(q_pe.shape) == 2
        assert len(kv_c.shape) == 2
        assert len(k_pe.shape) == 2
        assert ql_nope.shape[0] % num_q_heads == 0
        assert ql_nope.shape[0] == q_pe.shape[0]
        assert q_pe.shape[0] % bq_sz == 0
        assert ql_nope.shape[1] == lkv_dim
        assert q_pe.shape[1] == r_dim
        assert kv_c.shape == (bkv_sz, lkv_dim)
        assert k_pe.shape == (bkv_sz, r_dim)
        head_l_ref = l_ref.at[:ql_nope.shape[0]]
        head_m_ref = m_ref.at[:ql_nope.shape[0]]
        head_acc_ref = acc_ref.at[:ql_nope.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        # Follow FlashAttention-2 forward pass.
        s_nope = jnp.einsum("nd,md->nm",
                            ql_nope,
                            kv_c,
                            preferred_element_type=jnp.float32)
        s_pe = jnp.einsum("nd,md->nm",
                          q_pe,
                          k_pe,
                          preferred_element_type=jnp.float32)
        s = s_nope + s_pe
        s *= sm_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale

        q_span = (kv_len - q_len + bq_idx * bq_sz +
                  lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads)
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
        mask = q_span < k_span
        # TODO(jevinjiang, xiowei): reduce pages_per_seq based on sliding_window.
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)
        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = load_with_init(head_m_ref, -jnp.inf)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        pv = jnp.einsum("nm,md->nd",
                        p,
                        kv_c,
                        preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 0.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            # Skip DMA if debug mode is enabled.
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    # [KV-CACHE FUSION] _fetch_bkv is extended to load two sources:
    #   (a) tokens already in the paged cache  → from cache_kv HBM pages
    #   (b) tokens being added this step       → from new_kv_c/new_k_pe HBM
    # Both are packed in kv_packing (=4) rows so every DMA size is a
    # statically-provable multiple of 4 (Mosaic tiling requirement).
    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]
        reshaped_cache_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            *updated_cache_kv_hbm_ref.shape[2:],
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        # Split KV tokens: those already in cache vs. new tokens (tail of seq).
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache
        # Packing-rounded sizes (DMA must transfer whole packed rows).
        kv_left_frm_cache_per_packing = kv_left_frm_cache // kv_packing
        kv_left_frm_new_per_packing = cdiv(kv_left_frm_new, kv_packing)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bkv-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_idx={}", bkv_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] kv_len_start={}", kv_len_start)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_left={}", kv_left)
        debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
        debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        if not wait:
            # Fetch cached KV pages using packing-aware row sizes.
            def loop_body(i, _):
                sz_per_kv_packing = jnp.minimum(
                    page_size_per_kv_packing,
                    kv_left_frm_cache_per_packing -
                    i * page_size_per_kv_packing,
                )
                page_idx = jnp.minimum(page_indices_offset + i,
                                       num_page_indices - 1)
                _async_copy(
                    reshaped_cache_hbm_ref.at[pl.ds(
                        page_indices_ref[page_idx] * page_size_per_kv_packing,
                        sz_per_kv_packing,
                    ), ..., :nope_dim],
                    bkvc_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                           sz_per_kv_packing)],
                    sem,
                    wait=False,
                )
                _async_copy(
                    reshaped_cache_hbm_ref.at[pl.ds(
                        page_indices_ref[page_idx] * page_size_per_kv_packing,
                        sz_per_kv_packing,
                    ), ..., nope_dim:],
                    bkvpe_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                            sz_per_kv_packing)],
                    sem,
                    wait=False,
                )
                debug_print(
                    "[RPA debug] loop_body i={}, sz_per_kv_packing={}",
                    i,
                    sz_per_kv_packing,
                )

            actual_bkv_p = jnp.minimum(cdiv(kv_left_frm_cache, page_size),
                                       bkv_p)
            lax.fori_loop(
                0,
                actual_bkv_p,
                loop_body,
                None,
                unroll=False,
            )

            # [KV-CACHE FUSION] Load the new (not-yet-cached) token rows
            # directly from the new_kv_c / new_k_pe HBM input buffers.
            # Row address = (absolute token index) // kv_packing so every
            # DMA offset is a multiple of kv_packing (Mosaic safety).
            new_kv_row_start = (q_end - kv_left_frm_new) // kv_packing
            new_kv_row_start_frm_cache = kv_left_frm_cache_per_packing
            debug_print("[RPA debug] new_kv_row_start={}", new_kv_row_start)
            _async_copy(
                new_kv_c_hbm_ref.at[pl.ds(new_kv_row_start,
                                          kv_left_frm_new_per_packing)],
                bkvc_vmem_ref.at[pl.ds(new_kv_row_start_frm_cache,
                                       kv_left_frm_new_per_packing)],
                sem,
                wait,
            )
            _async_copy(
                new_k_pe_hbm_ref.at[pl.ds(new_kv_row_start,
                                          kv_left_frm_new_per_packing)],
                bkvpe_vmem_ref.at[pl.ds(new_kv_row_start_frm_cache,
                                        kv_left_frm_new_per_packing)],
                sem,
                wait,
            )
        else:
            # Wait path: issue a no-op barrier copy to drain the semaphore.
            kv_total_per_packing = cdiv(kv_left, kv_packing)
            dst = bkvc_vmem_ref.at[pl.ds(0, kv_total_per_packing)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    # [KV-CACHE FUSION] NEW FUNCTION: _update_kv_cache
    # Writes one B_q block of new tokens into the paged KV cache via async DMA.
    # Two-phase pipeline:
    #   Phase 1 (stage):   new_kv_c/k_pe HBM → VMEM staging buffer (sync, wait)
    #   Phase 2 (scatter): VMEM → correct page slots in updated_cache_kv HBM
    #                      via a per-page fori_loop (async, no-wait).
    # The semaphore for Phase 2 is drained by the *next* call to
    # wait_update_kv_cache before that slot's VMEM is reused.
    def _update_kv_cache(seq_idx,
                         bq_idx,
                         bkv_update_sem_idx,
                         update_sz,
                         *,
                         wait=False):
        # update_sz is a token count (= actual_bq_sz, always <= bq_sz).
        # All DMA slices are expressed in packing-aligned rows.
        sem = sems.at[3, bkv_update_sem_idx]
        new_kv_c_vmem_ref = new_kv_c_x2_ref.at[bkv_update_sem_idx]
        new_k_pe_vmem_ref = new_k_pe_x2_ref.at[bkv_update_sem_idx]
        reshaped_cache_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            *updated_cache_kv_hbm_ref.shape[2:],
        )

        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start
        # Packing-rounded token offset within this sequence's new tokens.
        token_offset_in_seq = bq_idx * bq_sz
        # Row in the packed new_kv_c HBM buffer where this bq block starts.
        new_kv_row_start = (q_start + token_offset_in_seq) // kv_packing
        update_rows = cdiv(update_sz, kv_packing)

        # Where these tokens land in the flat KV cache token address space.
        cache_token_start = kv_lens_ref[seq_idx] - q_len + token_offset_in_seq
        cache_row_start = cache_token_start // kv_packing  # row in page
        p_start = cache_row_start // page_size_per_kv_packing
        p_row_start = cache_row_start % page_size_per_kv_packing
        p_end = cdiv(cache_row_start + update_rows, page_size_per_kv_packing)
        page_indices_offset = seq_idx * pages_per_seq + p_start

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_update_kv_cache-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bq_idx={}", bq_idx)
        debug_print("[RPA debug] bkv_update_sem_idx={}", bkv_update_sem_idx)
        debug_print("[RPA debug] new_kv_row_start={}", new_kv_row_start)
        debug_print("[RPA debug] cache_token_start={}", cache_token_start)
        debug_print("[RPA debug] update_rows={}", update_rows)

        if not wait:
            # Stage packed rows into VMEM for the subsequent scatter.
            _async_copy(
                new_kv_c_hbm_ref.at[pl.ds(new_kv_row_start, update_rows)],
                new_kv_c_vmem_ref.at[pl.ds(0, update_rows)],
                sem,
                wait=True,
            )
            _async_copy(
                new_k_pe_hbm_ref.at[pl.ds(new_kv_row_start, update_rows)],
                new_k_pe_vmem_ref.at[pl.ds(0, update_rows)],
                sem,
                wait=True,
            )

            # Scatter packed rows from VMEM into the paged HBM cache.
            def loop_body(i, state):
                rem_rows, src_row = state
                dst_row_off = jnp.where(i == 0, p_row_start, 0)
                sz = jnp.minimum(page_size_per_kv_packing - dst_row_off,
                                 rem_rows)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    new_kv_c_vmem_ref.at[pl.ds(src_row, sz)],
                    reshaped_cache_hbm_ref.at[pl.ds(
                        page_idx * page_size_per_kv_packing + dst_row_off,
                        sz,
                    ), ..., :nope_dim],
                    sem,
                    wait=False,
                )
                _async_copy(
                    new_k_pe_vmem_ref.at[pl.ds(src_row, sz)],
                    reshaped_cache_hbm_ref.at[pl.ds(
                        page_idx * page_size_per_kv_packing + dst_row_off,
                        sz,
                    ), ..., nope_dim:],
                    sem,
                    wait=False,
                )
                return rem_rows - sz, src_row + sz

            lax.fori_loop(
                0,
                p_end - p_start,
                loop_body,
                (update_rows, 0),
                unroll=False,
            )
        else:
            # Drain semaphore with a no-op barrier copy.
            dst = new_kv_c_vmem_ref.at[pl.ds(0, update_rows)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_nope_vmem_ref = bq_nope_x2_ref.at[bq_sem_idx]
        bq_rope_vmem_ref = bq_rope_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bq-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bq_idx={}", bq_idx)
        debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            ql_nope_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_nope_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

        _async_copy(
            q_pe_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_rope_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_send_bo-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bo_idx={}", bo_idx)
        debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    # [KV-CACHE FUSION] start/wait helpers follow the same double-buffer
    # contract as the existing start/wait_send_bo pair:
    #   start: record (seq_idx, bq_idx, update_sz) in bkv_update_ids_ref so
    #          the matching wait knows what semaphore state to drain.
    #   wait:  only issues the drain DMA when update_sz > 0, i.e. when a
    #          real update was actually started on this semaphore slot.
    def start_update_kv_cache(seq_idx, bq_idx, bkv_update_sem_idx, update_sz):
        bkv_update_ids_ref[bkv_update_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_update_sem_idx + 2] = bq_idx
        bkv_update_ids_ref[bkv_update_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bq_idx, bkv_update_sem_idx, update_sz)

    def wait_update_kv_cache(bkv_update_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_update_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx_ = bkv_update_ids_ref[bkv_update_sem_idx]
            bq_idx_ = bkv_update_ids_ref[bkv_update_sem_idx + 2]
            bkv_update_ids_ref[bkv_update_sem_idx + 4] = 0
            _update_kv_cache(
                seq_idx_,
                bq_idx_,
                bkv_update_sem_idx,
                update_sz,
                wait=True,
            )

    def load_bq(bq_sem_idx, *, actual_bq_sz=bq_sz):
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        )
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        )
        return q_nope_vec, q_rope_vec

    def load_bkv(bkv_sem_idx, *, bkvc_mask, bkpe_mask):
        bkvc_ref = (bkvc_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(
            bkv_sz_per_kv_packing, lkv_dim))
        bkvc_vec = pltpu.bitcast(bkvc_ref[...], kv_dtype)
        bkvc_vec = lax.select(bkvc_mask, bkvc_vec, jnp.zeros_like(bkvc_vec))

        bkpe_ref = (bkpe_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(
            bkv_sz_per_kv_packing, r_dim))
        bkpe_vec = pltpu.bitcast(bkpe_ref[...], kv_dtype)
        bkpe_vec = lax.select(bkpe_mask, bkpe_vec, jnp.zeros_like(bkpe_vec))

        return bkvc_vec, bkpe_vec

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])],
            axis=-1)[..., :shape[-1]]

    def process(static_q_len=None):
        num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                # Create bitmask for KV.
                assert bkv_sz % kv_packing == 0
                actual_bkv_sz = jnp.minimum(bkv_sz, kv_len - bkv_idx * bkv_sz)
                bkvc_shape = (bkv_sz, lkv_dim)
                bkvc_mask = (lax.broadcasted_iota(jnp.int32, bkvc_shape, 0)
                             < actual_bkv_sz)
                bkpe_shape = (bkv_sz, r_dim)
                bkpe_mask = (lax.broadcasted_iota(jnp.int32, bkpe_shape, 0)
                             < actual_bkv_sz)

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                debug_print(
                    "[RPA debug] -----------flash attention-----------")
                debug_print("[RPA debug] seq_idx={}", seq_idx)
                debug_print("[RPA debug] bq_idx={}", bq_idx)
                debug_print("[RPA debug] bkv_idx={}", bkv_idx)
                if debug_mode:
                    # Skip flash attention if debug mode is enabled.
                    return

                # Flash attention with cur bkv and bq
                bkvc, bkpe = load_bkv(bkv_sem_idx,
                                      bkvc_mask=bkvc_mask,
                                      bkpe_mask=bkpe_mask)
                bq_nope_vec, bq_pe_vec = load_bq(bq_sem_idx,
                                                 actual_bq_sz=actual_bq_sz)
                flash_attention(
                    bq_nope_vec,
                    bq_pe_vec,
                    bkvc,
                    bkpe,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (lax.div(acc, l) if q_dtype == jnp.float32 else
                   (acc * pl.reciprocal(l, approx=True)).astype(q_dtype))

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                bq_sz * num_q_heads_per_q_packing,
                lkv_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

            # [KV-CACHE FUSION] PIPELINE TRIGGER
            # After the current B_q block's output DMA is launched, immediately
            # start writing that block's new-token KVs into the paged cache.
            # The DMA engine performs the scatter while the MXU works on the
            # *next* B_q block's attention – zero extra latency on the critical
            # path.  The semaphore slot reuses bq_sem_idx (same double-buffer
            # index) so the two DMAs (bo-send and kv-update) never collide.
            update_sem_idx = bq_sem_idx
            wait_update_kv_cache(update_sem_idx)
            start_update_kv_cache(
                seq_idx,
                bq_idx,
                update_sem_idx,
                actual_bq_sz,
            )

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, 0, 0)

    @pl.when(seq_idx < decode_end)
    def process_decode():
        process(static_q_len=1)

    @pl.when(jnp.logical_and(decode_end <= seq_idx, seq_idx < prefill_end))
    def process_prefill():
        process(static_q_len=chunk_prefill_size)

    @pl.when(jnp.logical_and(prefill_end <= seq_idx, seq_idx < mixed_end))
    def process_mixed():
        process()

    @pl.when(seq_idx == num_seqs - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)

    # [KV-CACHE FUSION] Drain any in-flight KV update DMAs before the kernel
    # exits.  Without this, the last one or two B_q blocks per sequence could
    # have their cache writes still outstanding when the kernel returns.
    @pl.when(seq_idx < num_seqs)
    def epilogue_kv_update():
        for i in range(2):
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q.reshape(
            max_num_tokens,
            actual_num_q_heads,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_q_heads // q_packing,
        q_packing,
        head_dim,
    )
    return q


def prepare_kv_inputs(
        kv: jax.Array,  # [max_num_tokens, actual_head_dim],
):
    max_num_tokens, actual_head_dim = kv.shape
    kv_packing = get_dtype_packing(kv.dtype)
    assert max_num_tokens % kv_packing == 0
    head_dim = align_to(actual_head_dim, 128)

    kv = kv.reshape(max_num_tokens // kv_packing, kv_packing, actual_head_dim)
    kv = jnp.pad(kv, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)),
                 constant_values=0)

    return kv


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads // q_packing, q_packing, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    (
        max_num_tokens,
        num_q_heads_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    return out.reshape(
        max_num_tokens,
        num_q_heads_per_q_packing * q_packing,
        head_dim,
    )[:, :actual_num_q_heads, :actual_head_dim]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "debug_mode",
    ),
    donate_argnames=("cache_kv"),
)
def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    # TODO(gpolovets): Explore separating out into lkv & pe KV caches.
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128)]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
) -> tuple[
        jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
        jax.
        Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128) + align_to(r_dim, 128)]
]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

  """
    # TODO(chengjiyao): Support both autotuning table and heuristic logic to set
    # these kernel block sizes
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )

    _, actual_num_q_heads, actual_lkv_dim = ql_nope.shape

    ql_nope = prepare_q_inputs(
        ql_nope
    )  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim]
    q_pe = prepare_q_inputs(
        q_pe)  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, r_dim]
    new_kv_c = prepare_kv_inputs(
        new_kv_c)  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe = prepare_kv_inputs(
        new_k_pe)  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
    lkv_dim = new_kv_c.shape[-1]
    r_dim = new_k_pe.shape[-1]

    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    _, num_q_heads_per_q_packing, q_packing, _ = ql_nope.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    num_q_heads = num_q_heads_per_q_packing * q_packing

    bkv_p = num_kv_pages_per_block
    bq_sz = num_queries_per_block
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    grid = (distribution[2], )

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]

    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]

    bkvc_double_buf = pltpu.VMEM(
        (2, bkv_sz_per_kv_packing, kv_packing, lkv_dim),
        cache_kv.dtype,
    )

    bkpe_double_buf = pltpu.VMEM(
        (2, bkv_sz_per_kv_packing, kv_packing, r_dim),
        cache_kv.dtype,
    )

    bq_nope_double_buf = pltpu.VMEM(
        (2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim),
        ql_nope.dtype,
    )

    bq_rope_double_buf = pltpu.VMEM(
        (2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim),
        q_pe.dtype,
    )

    bo_double_buf = bq_nope_double_buf

    # bouache : [KV-CACHE FUSION] VMEM staging buffers for the in-kernel KV write-back.
    # Shape is (2, bq_sz/kv_packing, kv_packing, dim):
    #   dim-0 = double-buffer slot (0 or 1)
    #   dim-1 = packed rows (bq_sz ÷ kv_packing) – keeps DMA sizes provably
    #           divisible by kv_packing=4 for Mosaic's fp8 tiling check
    #   dim-2 = kv_packing (fp8 lane interleave)
    #   dim-3 = head dimension
    bq_sz_per_kv_packing = bq_sz // kv_packing
    new_kv_c_double_buf = pltpu.VMEM(
        (2, bq_sz_per_kv_packing, kv_packing, lkv_dim),
        cache_kv.dtype,
    )

    new_k_pe_double_buf = pltpu.VMEM(
        (2, bq_sz_per_kv_packing, kv_packing, r_dim),
        cache_kv.dtype,
    )

    l_scratch = pltpu.VMEM(
        (bq_sz * num_q_heads, 128),
        jnp.float32,
    )
    m_scratch = l_scratch

    acc_scratch = pltpu.VMEM(
        (bq_sz * num_q_heads, lkv_dim),
        jnp.float32,
    )

    scratch_shapes = [
        bkvc_double_buf,
        bkpe_double_buf,
        bq_nope_double_buf,
        bq_rope_double_buf,
        bo_double_buf,  # Double buffering for output block.
        new_kv_c_double_buf,  # Double buffering for KV-cache updates.
        new_k_pe_double_buf,  # Double buffering for KV-cache updates.
        # Semaphores for double buffering of bkv, bq, bo and bkv_update.
        pltpu.SemaphoreType.DMA((4, 2)),
        # Intermediate buffers per kv head for flash attention.
        l_scratch,
        m_scratch,
        acc_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        # TODO(jevinjiang): can we use ragged page_indices to save some smem?
        page_indices,
        cu_q_lens,
        distribution,
        # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
        jnp.zeros((3, ), jnp.int32),
        # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
        jnp.full((4, ), -1, jnp.int32),
        # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_bq_idx, bkv_sem_1_bq_idx, bkv_sem_0_sz, bkv_sem_1_sz)
        jnp.full((6, ), -1, jnp.int32),
    )

    scope_name = f"MLA-RPA-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _mla_ragged_paged_attention_kernel,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                chunk_prefill_size=chunk_prefill_size,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                debug_mode=debug_mode,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # TODO(jevinjiang): since each sequence depends on the previous
                # one, we need some extra work to support Megacore mode.
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                # [KV-CACHE FUSION] Disable per-DMA software bounds checks.
                # Safe because the scheduler guarantees all page indices are
                # within [0, total_num_pages) before calling this kernel.
                # Removing these checks reduces per-DMA overhead noticeably
                # at high page-scatter counts.
                disable_bounds_checks=True,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(shape=ql_nope.shape, dtype=ql_nope.dtype),
                jax.ShapeDtypeStruct(shape=cache_kv.shape,
                                     dtype=cache_kv.dtype),
            ],
            input_output_aliases={
                7: 0,
                11: 1,
            },
            name=scope_name,
        ))

    # [KV-CACHE FUSION] The host-side `update_kv_cache(new_kv_c, new_k_pe, …)`
    # call that previously ran here has been REMOVED.  The KV cache is now
    # updated entirely inside the Pallas kernel above, returned via
    # input_output_aliases {11: 1}.  One fewer TPU kernel launch per step.
    output, updated_kv = kernel(
        *scalar_prefetches,
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
    )
    output = prepare_outputs(
        output, actual_num_q_heads,
        actual_lkv_dim)  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]

    return output, updated_kv
