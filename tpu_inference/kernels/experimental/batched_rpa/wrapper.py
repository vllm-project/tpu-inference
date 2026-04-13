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
"""Wrapper for RPA kernel to match expected interface.

NOTE: all of the code in this directory is experimental and not fully tested!
To enable usage of this kernel in full run, you can pass the USE_BATCHED_RPA_KERNEL=1
environment variable.

Compared to the default RPA kernel, this kernel does the following:

1. Batches multiple sequences together to replace per-request flash_attention loops. 

2. Enables triple-buffering via Pallas emit_pipeline

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via 
scheduler.py kernel. Kernel is calculated once and ammortized across different layers in a model. 

Note: batched_rpa is build on top / derived from RPA3. 
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import (configs, kernel,
                                                            schedule, utils)


# Expect to run this validation during compile time.
def static_validate_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    use_causal_mask: bool = True,
):
    """Validate inputs to the RPA kernel statically."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    if not q.ndim == k.ndim == k.ndim == 3:
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

    if head_dim != (aligned_head_dim := utils.align_to(actual_head_dim,
                                                       num_lanes)):
        raise ValueError(
            f"Expected {head_dim=} is equal to {aligned_head_dim=}")
    # Note: we expect the kv quantization happens outside of the RPA kernel.
    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}."
        )
    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if kv_packing != utils.get_dtype_packing(kv_cache.dtype):
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

    for block_sizes in (decode_block_sizes, prefill_block_sizes):
        if block_sizes is not None:
            if block_sizes.bkv_sz <= 0:
                raise ValueError(f"{block_sizes.bkv_sz=} must be positive.")
            if block_sizes.bkv_sz % page_size != 0:
                raise ValueError(
                    f"{block_sizes.bkv_sz=} must be divisible by {page_size=}."
                )
            if block_sizes.bq_sz <= 0:
                raise ValueError(f"{block_sizes.bq_sz=} must be positive.")
            if block_sizes.n_buffer <= 0:
                raise ValueError(f"{block_sizes.n_buffer=} must be positive.")

    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale, mask_value, q_scale, k_scale, v_scale, debug_mode


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:

    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (jnp.pad(
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
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2,
                                             kv_packing)
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
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    num_lanes = pltpu.get_tpu_info().num_lanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


def get_default_block_sizes(
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    actual_num_q_heads: int,
    actual_num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_num_tokens: int,
    max_num_seqs: int,
    pages_per_seq: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Get (bq_sz, bkv_sz, batch_size) by some heuristic formulas."""
    del (
        q_dtype,
        head_dim,
        max_num_tokens,
        max_num_seqs,
        pages_per_seq,
    )
    is_8bit = utils.get_dtype_packing(kv_dtype) == 4
    # Qwen32b
    if actual_num_q_heads == 32 and actual_num_kv_heads == 4 and is_8bit:
        return configs.BlockSizes(
            bq_sz=1,
            bkv_sz=512,
            batch_size=10,
            n_buffer=2,
        ), configs.BlockSizes(
            bq_sz=256,
            bkv_sz=512,
            batch_size=2,
            n_buffer=2,
        )
    # Qwen-coder
    if actual_num_q_heads == 12 and actual_num_kv_heads == 1 and is_8bit:
        return configs.BlockSizes(
            bq_sz=1,
            bkv_sz=2304,
            batch_size=8,
            n_buffer=3,
        ), configs.BlockSizes(
            bq_sz=512,
            bkv_sz=512,
            batch_size=3,
            n_buffer=3,
        )

    default_block_sizes = configs.BlockSizes(
        bq_sz=1,
        bkv_sz=page_size,
        batch_size=1,
        n_buffer=2,
    )

    return default_block_sizes, default_block_sizes


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
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
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    Args:
        queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
        keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
        values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
        kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
            page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
            sequences. distribution[2] represents total number of sequences.
        sm_scale: Softmax scale value.
        sliding_window: Size of sliding window (also known as local attention). kvs
            outside of the window is not fetched from hbm and masked out during
            computation.
        soft_cap: Cap values of softmax inputs.
        mask_value: Value to use for causal masking. Defaults to smallest
            representable value of the activation dtype.
        q_scale: Quantization scale value of queries.
        k_scale: Quantization scale value of keys.
        v_scale: Quantization scale value of values.
        chunk_prefill_size: Not used.
        decode_block_sizes: Kernel block size to use during decode.
        prefill_block_sizes: Kernel block size to use during prefill.
        vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
            size of the hardware.
        debug_mode: Not used.
        out_dtype: Dtype of output. Defaults to dtype of queries.
        use_causal_mask: Not used.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Result of new kv cache where k & vs are
            concatenated along num kv heads dim.
    """

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
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        use_causal_mask=use_causal_mask,
    )
    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    total_q_tokens = queries.shape[0]

    q_hbm, new_kv_hbm = prepare_inputs(queries, keys, values, queries.dtype,
                                       kv_cache.dtype)

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
    ):
        default_decode, default_prefill = get_default_block_sizes(
            queries.dtype,
            kv_cache.dtype,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            total_q_tokens,
            max_num_seqs,
            pages_per_seq,
        )
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or default_decode
        else:
            effective_blocks = prefill_block_sizes or default_prefill

        cfgs = configs.RPAConfig(
            block=effective_blocks,
            model=configs.ModelConfigs(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                sliding_window=sliding_window,
                sm_scale=sm_scale,
                soft_cap=soft_cap,
                mask_value=mask_value,
            ),
            serve=configs.ServingConfigs(
                num_seqs=max_num_seqs,
                num_page_indices=num_page_indices,
                total_q_tokens=total_q_tokens,
                dtype_q=queries.dtype,
                dtype_kv=kv_cache.dtype,
                dtype_out=out_dtype,
                page_size=page_size,
                scale_q=q_scale,
                scale_k=k_scale,
                scale_v=v_scale,
            ),
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            cfgs=cfgs,
        )
        return kernel.rpa_kernel(
            cu_q_lens,
            kv_lens,
            page_indices,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            cfgs=cfgs,
        )

    o_hbm_alias_q_hbm, kv_cache = run_rpa_kernel(configs.RpaCase.DECODE, q_hbm,
                                                 kv_cache)
    o_hbm_alias_q_hbm, kv_cache = run_rpa_kernel(configs.RpaCase.MIXED,
                                                 o_hbm_alias_q_hbm, kv_cache)

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    return o_hbm, kv_cache
