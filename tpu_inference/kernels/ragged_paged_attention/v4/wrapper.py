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
"""Wrapper for RPA kernel to match expected interface."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v4 import kernel, schedule

DEFAULT_MASK_VALUE = -float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def prepare_inputs(
    q: jax.Array,  # [max_tokens, num_q_heads, head_dim]
    k: jax.Array,  # [max_tokens, num_kv_heads, head_dim]
    v: jax.Array,  # [max_tokens, num_kv_heads, head_dim]
    q_dtype: Any,
    kv_dtype: Any,
):
    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = schedule.get_dtype_packing(q_dtype)
    kv_packing = schedule.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = align_to(num_q_heads_per_kv_head,
                                               q_packing)
    aligned_head_dim = align_to(actual_head_dim, 128)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    q_hbm = (jnp.pad(
        q.reshape(
            total_q_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        actual_num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_head_dim,
    ).swapaxes(0, 1))

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = align_to(actual_num_kv_heads_x2, kv_packing)
    new_kv_hbm = jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                 actual_num_kv_heads_x2,
                                                 actual_head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
            (0, aligned_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        num_kv_heads_x2_aligned // kv_packing,
        kv_packing,
        aligned_head_dim,
    )
    return q_hbm, new_kv_hbm


def prepare_outputs(
        out: jax.
    Array,  # [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
):
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def _get_max_steps_ub(
    max_num_seqs: int,
    pages_per_seq: int,
    bkv_sz: int,
    page_size: int,
    batch_size: int,
    case: schedule.RpaCase,
) -> int:
    """Get max_steps_ub based on SMEM limit."""
    # We use a static allocation based on SMEM limit (1 MiB) to avoid
    # data-dependent shapes which can cause host-device syncs.
    # The schedule is stored in SMEM along with kv_lens, cu_q_lens, page_indices,
    # distribution, lane_lengths, and actual_steps.
    smem_limit_bytes = (1024 - 32) * 1024
    word_size_bytes = 4
    fixed_bytes = (
        max_num_seqs * word_size_bytes  # kv_lens
        + (max_num_seqs + 1) * word_size_bytes  # cu_q_lens
        + max_num_seqs * pages_per_seq * word_size_bytes  # page_indices
        + 3 * word_size_bytes  # distribution
        + batch_size * word_size_bytes  # lane_lengths
        + 1 * word_size_bytes  # actual_steps
    )
    available_bytes = smem_limit_bytes - fixed_bytes

    bkv_p = bkv_sz // page_size
    bkv_p_cache = 0 if case == schedule.RpaCase.PREFILL else bkv_p
    bkv_p_new = 1 if case == schedule.RpaCase.DECODE else bkv_p

    # Per step per batch item:
    # s_idx, q_idx, k_idx, is_last_k, do_writeback: 5 * 4 = 20
    # dma_q: 2 * 4 = 8
    # dma_kv_cache: bkv_p_cache * 3 * 4 = 12 * bkv_p_cache
    # dma_kv_new: bkv_p_new * 4 * 4 = 16 * bkv_p_new
    bytes_per_step = batch_size * (28 + 12 * bkv_p_cache + 16 * bkv_p_new)
    max_steps_ub = available_bytes // bytes_per_step
    max_steps_ub = (max_steps_ub // 128) * 128
    if max_steps_ub <= 0:
        max_steps_ub = 128
    return max_steps_ub


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = schedule.get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


def get_vmem_estimate_bytes(
    bq_sz,
    bkv_sz,
    batch_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    q_dtype,
    kv_dtype,
    n_buffer=2,
):
    """Get VMEM estimate bytes."""
    q_packing = schedule.get_dtype_packing(q_dtype)
    kv_packing = schedule.get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    aligned_num_q_heads_per_kv_head = align_to(num_q_heads_per_kv_head,
                                               q_packing)
    aligned_head_dim = align_to(head_dim, 128)

    m_shape = (
        batch_size,
        num_kv_heads,
        bq_sz * aligned_num_q_heads_per_kv_head,
        128,
    )
    l_shape = (
        batch_size,
        num_kv_heads,
        bq_sz * aligned_num_q_heads_per_kv_head,
        128,
    )
    acc_shape = (
        batch_size,
        num_kv_heads,
        bq_sz * aligned_num_q_heads_per_kv_head,
        aligned_head_dim,
    )
    m_bytes = np.prod(m_shape) * 4
    l_bytes = np.prod(l_shape) * 4
    acc_bytes = np.prod(acc_shape) * 4

    q_vmem_shape = (
        batch_size,
        num_kv_heads,
        bq_sz,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_head_dim,
    )
    q_bytes = np.prod(q_vmem_shape) * jnp.dtype(q_dtype).itemsize

    bkv_stride = (num_kv_heads * 2) // kv_packing
    if schedule.has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    kv_vmem_shape = (
        batch_size,
        bkv_sz,
        bkv_stride,
        kv_packing,
        aligned_head_dim,
    )
    kv_bytes = np.prod(kv_vmem_shape) * jnp.dtype(kv_dtype).itemsize

    return m_bytes + l_bytes + acc_bytes + (n_buffer +
                                            2) * q_bytes + n_buffer * kv_bytes


# Expect to run this validation during compile time.
def static_validate_inputs(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
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
    bq_sz: int | None = None,
    bkv_sz: int | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the RPA kernel statically."""
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(
            f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )

    actual_head_dim = q.shape[2]
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(f"Expected {actual_num_q_heads=} to be divisible by"
                         f" {actual_num_kv_heads=}.")

    expected_kv_cache_shape = get_kv_cache_shape(
        kv_cache.shape[0],
        kv_cache.shape[1],
        actual_num_kv_heads,
        actual_head_dim,
        kv_cache.dtype,
    )

    if kv_cache.shape != expected_kv_cache_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
        )

    (
        _,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        head_dim,
    ) = kv_cache.shape

    if head_dim != align_to(actual_head_dim, 128):
        raise ValueError(
            f"Expected {head_dim=} is equal to {align_to(actual_head_dim, 128)=}"
        )
    # Note: we expect the kv quantization happens outside of the RPA kernel.
    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}."
        )
    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if kv_packing != schedule.get_dtype_packing(kv_cache.dtype):
        raise ValueError(
            f"{kv_packing=} does not match with {kv_cache.dtype=}")

    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    if num_kv_heads_x2 % 2 != 0:
        raise ValueError(
            f"Combined KV heads must be divisible by 2, but got {num_kv_heads_x2}"
        )
    if (num_kv_heads_x2 % kv_packing != 0
            or num_kv_heads_x2 // 2 < actual_num_kv_heads):
        raise ValueError(
            f"Invalid {num_kv_heads_x2=}, {actual_num_kv_heads=}, {kv_packing=}"
        )

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

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if bkv_sz is not None:
        if bkv_sz <= 0:
            raise ValueError(f"{bkv_sz=} must be positive.")
        if bkv_sz % page_size != 0:
            raise ValueError(f"{bkv_sz=} must be divisible by {page_size=}.")
    if bq_sz is not None:
        if bq_sz <= 0:
            raise ValueError(f"{bq_sz=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del debug_mode


def get_default_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    max_num_seqs,
    pages_per_seq,
):
    """Get (bq_sz, bkv_sz, batch_size) by some heuristic formulas.

  Note the default block sizes are not necessarily optimal.
  """
    del (
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        max_num_tokens,
        max_num_seqs,
        pages_per_seq,
    )
    return {
        "bq_sz": 1,
        "bkv_sz": 1024,
        "batch_size": 10
    }, {
        "bq_sz": 128,
        "bkv_sz": 512,
        "batch_size": 2,
    }


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
        "bq_sz",
        "bkv_sz",
        "bq_csz",
        "bkv_csz",
        "batch_size",
        "vmem_limit_bytes",
        "debug_mode",
        "m_block_sizes",
        "n_buffer",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    bq_sz: int | None = None,
    bkv_sz: int | None = None,
    # obsolete, for benchmarking backwards compatibility.
    bq_csz: int | None = None,
    bkv_csz: int | None = None,
    batch_size: int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    n_buffer: int = 2,
):
    static_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
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
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    num_seq_total = distribution[2]
    max_num_seqs = kv_lens.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]

    actual_num_q_heads = queries.shape[1]
    actual_head_dim = queries.shape[2]
    actual_num_kv_heads = keys.shape[1]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    max_possible_kv_len = pages_per_seq * page_size
    total_q_tokens = queries.shape[0]

    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_hbm, new_kv_hbm = prepare_inputs(queries, keys, values, queries.dtype,
                                       kv_cache.dtype)
    o_hbm = jnp.zeros_like(q_hbm)
    _, _, q_per_kv_packed, q_packing, aligned_head_dim = q_hbm.shape
    aligned_num_q_heads_per_kv_head = q_per_kv_packed * q_packing

    def run_rpa_kernel(
        case: schedule.RpaCase,
        q_hbm,
        o_hbm,
        kv_cache,
    ):
        decode_block_sizes, prefill_block_sizes = get_default_block_sizes(
            queries.dtype,
            kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            actual_head_dim,
            page_size,
            total_q_tokens,
            max_num_seqs,
            pages_per_seq,
        )
        if case == schedule.RpaCase.DECODE:
            block_sizes = decode_block_sizes
        else:
            block_sizes = prefill_block_sizes
        kernel_batch_size = (batch_size if batch_size is not None else
                             block_sizes["batch_size"])
        kernel_bq_sz = bq_sz if bq_sz is not None else block_sizes["bq_sz"]
        kernel_bkv_sz = bkv_sz if bkv_sz is not None else block_sizes["bkv_sz"]

        max_steps_ub = _get_max_steps_ub(
            max_num_seqs,
            pages_per_seq,
            kernel_bkv_sz,
            page_size,
            kernel_batch_size,
            case,
        )

        config = schedule.RPAConfig(
            num_seq=max_num_seqs,
            bq_sz=kernel_bq_sz,
            bkv_sz=kernel_bkv_sz,
            batch_size=kernel_batch_size,
            page_size=page_size,
            bkv_p=kernel_bkv_sz // page_size,
            pages_per_seq=pages_per_seq,
            max_steps_ub=max_steps_ub,
            total_q_tokens=total_q_tokens,
            head_dim=aligned_head_dim,
            num_kv_heads=actual_num_kv_heads,
            num_q_heads_per_kv_head=aligned_num_q_heads_per_kv_head,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            sliding_window=sliding_window,
            mask_value=mask_value,
            q_dtype=queries.dtype,
            kv_dtype=kv_cache.dtype,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            vmem_limit_bytes=vmem_limit_bytes
            if vmem_limit_bytes is not None else DEFAULT_VMEM_LIMIT_BYTES,
            total_num_pages=total_num_pages,
            case=case,
            n_buffer=n_buffer,
        )
        rpa_kernel_instance = kernel.make_rpa_kernel(config)
        kv_packing = schedule.get_dtype_packing(kv_cache.dtype)
        bkv_stride = (actual_num_kv_heads * 2) // kv_packing
        if schedule.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        kv_vmem_shape_single = (
            kernel_bkv_sz,
            bkv_stride,
            kv_packing,
            aligned_head_dim,
        )
        kv_cache_zero_hbm = jnp.zeros(kv_vmem_shape_single,
                                      dtype=kv_cache.dtype)
        o_hbm, kv_cache = rpa_kernel_instance(
            cu_q_lens,
            kv_lens,
            page_indices,
            distribution,
            q_hbm,
            new_kv_hbm,
            kv_cache,
            o_hbm,
            kv_cache_zero_hbm,
        )
        return o_hbm, kv_cache

    o_hbm, kv_cache = run_rpa_kernel(schedule.RpaCase.DECODE, q_hbm, o_hbm,
                                     kv_cache)
    o_hbm, kv_cache = run_rpa_kernel(schedule.RpaCase.PREFILL, q_hbm, o_hbm,
                                     kv_cache)

    # o_hbm: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm)
    # o_hbm now: [kv_heads, max_tokens, q_per_kv, d]

    # We need to slice back to original shape if padded
    o_hbm = (o_hbm[:, :, :num_q_heads_per_kv_head, :actual_head_dim].transpose(
        1, 0, 2, 3).reshape(total_q_tokens, actual_num_q_heads,
                            actual_head_dim))

    return o_hbm, kv_cache
