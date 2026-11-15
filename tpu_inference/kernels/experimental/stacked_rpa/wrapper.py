# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    block_sizes as decode_blocks
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    config as decode_config
from tpu_inference.kernels.experimental.stacked_rpa.decode.non_sliding_window import \
    kernel as non_sliding_window_kernel
from tpu_inference.kernels.experimental.stacked_rpa.decode.non_sliding_window import \
    schedule as non_sliding_window_schedule
from tpu_inference.kernels.experimental.stacked_rpa.decode.sliding_window import \
    kernel as sliding_window_kernel
from tpu_inference.kernels.experimental.stacked_rpa.decode.sliding_window import \
    schedule as sliding_window_schedule
from tpu_inference.kernels.experimental.stacked_rpa.prefill import \
    block_sizes as prefill_blocks
from tpu_inference.kernels.experimental.stacked_rpa.prefill import \
    config as prefill_config
from tpu_inference.kernels.experimental.stacked_rpa.prefill.non_sliding_window import \
    kernel as non_sliding_window_prefill_kernel
from tpu_inference.kernels.experimental.stacked_rpa.prefill.non_sliding_window import \
    schedule as non_sliding_window_prefill_schedule
from tpu_inference.kernels.experimental.stacked_rpa.prefill.sliding_window import \
    block_sizes as sliding_window_prefill_blocks
from tpu_inference.kernels.experimental.stacked_rpa.prefill.sliding_window import \
    kernel as sliding_window_prefill_kernel
from tpu_inference.kernels.experimental.stacked_rpa.prefill.sliding_window import \
    schedule as sliding_window_prefill_schedule


def prepare_queries(
    q: jax.Array,
    num_kv_heads: int,
    q_dtype: jnp.dtype,
) -> jax.Array:
    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    num_q_heads_per_kv_head = actual_num_q_heads // num_kv_heads
    q_packing = utils.get_dtype_packing(q_dtype)
    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_q_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    return (jnp.pad(
        q.reshape(
            total_q_tokens,
            num_kv_heads,
            num_q_heads_per_kv_head,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    ).swapaxes(0, 1))


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    prepacked_new_kv_hbm: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    kv_packing = utils.get_dtype_packing(kv_dtype)

    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_kv_head_dim = utils.align_to(actual_head_dim,
                                         num_sublanes * kv_packing)

    o_hbm_alias_q_hbm = prepare_queries(q, actual_num_kv_heads, q_dtype)

    actual_num_kv_heads_x2 = actual_num_kv_heads * 2

    if prepacked_new_kv_hbm is not None:
        new_kv_hbm = prepacked_new_kv_hbm
        expected_shape = (
            actual_num_kv_heads_x2,
            aligned_kv_head_dim,
            utils.align_to(total_q_tokens, num_lanes),
        )
        if new_kv_hbm.shape != expected_shape:
            raise ValueError(
                f"prepacked_new_kv_hbm has shape {new_kv_hbm.shape}, "
                f"expected {expected_shape}")
    else:
        new_kv_hbm = prepare_seq_along_lane_new_kv_hbm(k, v, kv_dtype=kv_dtype)
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_seq_along_lane_new_kv_hbm(
    k: jax.Array,
    v: jax.Array,
    *,
    kv_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Pack post-RoPE K/V directly into batched-RPA SEQ_ALONG_LANE layout.

    Inputs are the normal projection layout ``[tokens, kv_heads, head_dim]``.
    Output is ``[2 * kv_heads, hd_aligned, padded_tokens]`` (stacked keeps
    head_dim contiguous, unlike the batched 4D packed layout), with sequence on
    the lane/minor axis.
    """
    total_q_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    if v.shape != k.shape:
        raise ValueError(f"k/v shapes differ: {k.shape=} {v.shape=}")
    if kv_dtype is None:
        kv_dtype = k.dtype
    kv_packing = utils.get_dtype_packing(kv_dtype)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_kv_head_dim = utils.align_to(actual_head_dim,
                                         num_sublanes * kv_packing)
    padded_total_tokens = utils.align_to(total_q_tokens, num_lanes)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    return (jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                 actual_num_kv_heads_x2,
                                                 actual_head_dim),
        (
            (0, padded_total_tokens - total_q_tokens),
            (0, 0),
            (0, aligned_kv_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        padded_total_tokens,
        actual_num_kv_heads_x2,
        aligned_kv_head_dim,
    ).transpose(1, 2, 0))


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
    # page_size is fixed at cache-allocation time and must be a multiple of 128.
    # Larger pages reduce KV DMA descriptors but can overcompute a partially used
    # final page, so callers should choose a size appropriate for their contexts.
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        actual_num_kv_heads * 2,
        utils.align_to(actual_head_dim, num_sublanes * kv_packing),
        page_size,
    )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_q_len: int = 1,
) -> tuple[decode_blocks.BlockSizes, prefill_blocks.BlockSizes]:
    """Choose the concrete decode policy and the prefill policy."""
    if model_cfgs.sliding_window is None:
        prefill = prefill_blocks.choose_block_sizes(model_cfgs, serve_cfgs,
                                                    vmem_limit_bytes)
    else:
        prefill = sliding_window_prefill_blocks.choose_block_sizes(
            model_cfgs, serve_cfgs, vmem_limit_bytes)
    return (
        decode_blocks.choose_block_sizes(model_cfgs, serve_cfgs,
                                         vmem_limit_bytes, decode_q_len),
        prefill,
    )


def _resolve_attn_static(
    queries: jax.Array,
    keys: jax.Array | None,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    decode_block_sizes: decode_blocks.BlockSizes | None,
    prefill_block_sizes: prefill_blocks.BlockSizes | None,
    decode_q_len: int = 1,
    num_kv_heads: int | None = None,
):
    """Resolve the static attention configs + effective block sizes.

    Shared by ``ragged_paged_attention`` and ``build_schedules`` so the
    The concrete decode and prefill configs used to precompute schedules are
    byte-identical to those used by their kernels (the schedule embeds ``cfgs``
    statically, so any drift would be a correctness bug).
    """
    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    if kv_cache.ndim != 4:
        raise ValueError(
            "stacked_rpa requires a rank-4 KV cache with shape "
            "[num_pages, 2 * num_kv_heads, aligned_head_dim, page_size], "
            f"but got {kv_cache.shape}.")
    page_size = kv_cache.shape[3]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    if num_kv_heads is None:
        if keys is None:
            raise ValueError("num_kv_heads must be provided when keys is None")
        num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
    )

    if model_cfgs.sliding_window is None:
        effective_decode = decode_blocks.choose_non_sliding_window_block_sizes(
            model_cfgs,
            serve_cfgs,
            vmem_limit_bytes,
            decode_q_len,
            override=decode_block_sizes,
        )
    else:
        effective_decode = decode_blocks.choose_sliding_window_block_sizes(
            model_cfgs,
            serve_cfgs,
            vmem_limit_bytes,
            decode_q_len,
            override=decode_block_sizes,
        )
    if model_cfgs.sliding_window is None:
        effective_prefill = prefill_blocks.choose_block_sizes(
            model_cfgs,
            serve_cfgs,
            vmem_limit_bytes,
            override=prefill_block_sizes,
        )
    else:
        effective_prefill = sliding_window_prefill_blocks.choose_block_sizes(
            model_cfgs,
            serve_cfgs,
            vmem_limit_bytes,
            override=prefill_block_sizes,
        )

    return (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        out_dtype,
        mask_value,
    )


def _validate_prepacked_inputs(
    cfgs: decode_config.DecodeConfig | prefill_config.PrefillConfig,
    q: jax.Array,
    prepacked_new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> None:
    """Validate the prepacked-KV entry point without requiring normal K/V arrays."""
    if q.ndim != 3:
        raise ValueError(f"Expected 3D array for {q.shape=}")

    expected_kv_shape = (
        cfgs.model.num_kv_heads * 2,
        cfgs.aligned_kv_head_dim,
        utils.align_to(q.shape[0],
                       pltpu.get_tpu_info().num_lanes),
    )
    if prepacked_new_kv_hbm.shape != expected_kv_shape:
        raise ValueError(
            f"Expected {prepacked_new_kv_hbm.shape=} to be equal to"
            f" {expected_kv_shape=}")
    if prepacked_new_kv_hbm.dtype != kv_cache.dtype:
        raise ValueError(
            "Expected prepacked KV dtype and KV cache dtype to match, but got"
            f" {prepacked_new_kv_hbm.dtype=} and {kv_cache.dtype=}")

    expected_kv_cache_shape = (
        kv_cache.shape[0],
        cfgs.model.num_kv_heads * 2,
        cfgs.aligned_kv_head_dim,
        cfgs.serve.page_size,
    )
    if kv_cache.shape != expected_kv_cache_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
        )
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")
    if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
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


def _empty_schedule_like(schedule_type, cfgs, *, plan=None):
    """Return a concrete zero-step schedule without launching its builder."""
    template = schedule_type.create_hbm_shape_dtype(cfgs, plan=plan)
    return jax.tree.map(lambda value: jnp.zeros(value.shape, value.dtype),
                        template)


def build_schedules(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    visibility: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    vmem_limit_bytes: int | None = None,
    out_dtype: jnp.dtype | None = None,
    decode_block_sizes: decode_blocks.BlockSizes | None = None,
    prefill_block_sizes: prefill_blocks.BlockSizes | None = None,
    update_kv_cache: bool = True,
    dispatch_hint: str = "auto",
):
    """Precompute the (DECODE, MIXED) RPA schedules once for a forward step.

    The schedule depends on ``cu_q_lens, kv_lens, page_indices, distribution`` +
    static cfgs. ``page_indices`` is folded into the emitted DMA offsets, so a
    precomputed schedule may only be reused by layers sharing the same metadata
    stream. ``values`` is accepted for a signature symmetric with
    ``ragged_paged_attention`` (only shapes/dtypes of q/k/kv_cache are needed
    for cfgs).
    """
    del values
    if visibility is not None:
        raise ValueError("visibility is not supported by stacked RPA.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        _out_dtype,
        _mask_value,
    ) = _resolve_attn_static(
        queries,
        keys,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        decode_q_len=decode_q_len,
    )
    decode_cfgs = decode_config.make_config(
        model_cfgs,
        serve_cfgs,
        effective_decode,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = prefill_config.make_config(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_prefill,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
    )
    if model_cfgs.sliding_window is None:
        decode_schedule_module = non_sliding_window_schedule
        decode_schedule_type = non_sliding_window_schedule.DecodeSchedule
        mixed_schedule_module = non_sliding_window_prefill_schedule
        mixed_schedule_type = non_sliding_window_prefill_schedule.PrefillSchedule
    else:
        decode_schedule_module = sliding_window_schedule
        decode_schedule_type = sliding_window_schedule.SlidingWindowSchedule
        mixed_schedule_module = sliding_window_prefill_schedule
        mixed_schedule_type = sliding_window_prefill_schedule.SlidingWindowSchedule

    build_decode = dispatch_hint in ("auto", "decode_only")
    build_mixed = dispatch_hint in ("auto", "mixed_only")
    decode_schedule_hbm = (jax.lax.cond(
        distribution[0] > 0,
        lambda: decode_schedule_module.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=decode_cfgs,
        ),
        lambda: _empty_schedule_like(decode_schedule_type, decode_cfgs),
    ) if build_decode else None)
    mixed_schedule = (jax.lax.cond(
        distribution[2] > distribution[1],
        lambda: mixed_schedule_module.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=mixed_cfgs,
        ),
        lambda: _empty_schedule_like(mixed_schedule_type, mixed_cfgs),
    ) if build_mixed else None)
    return decode_schedule_hbm, mixed_schedule


def build_schedules_shared(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices_list: list[jax.Array],
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
    decode_q_len: int = 1,
    vmem_limit_bytes: int | None = None,
    out_dtype: jnp.dtype | None = None,
    decode_block_sizes: decode_blocks.BlockSizes | None = None,
    prefill_block_sizes: prefill_blocks.BlockSizes | None = None,
    update_kv_cache: bool = True,
):
    """Build local-group schedules with one shared SW-decode launch.

    Sliding-window prefill remains a per-group fallback because it has a
    different compact schedule and can contain a dynamic number of Q blocks.
    """
    del values
    if sliding_window is None:
        raise ValueError("Shared schedules require sliding-window attention.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    page_indices_list = list(page_indices_list)
    num_groups = len(page_indices_list)
    if num_groups < 1:
        raise ValueError(
            "page_indices_list must contain at least one cache group.")
    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        _out_dtype,
        _mask_value,
    ) = _resolve_attn_static(
        queries,
        keys,
        kv_cache,
        kv_lens,
        page_indices_list[0],
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        decode_q_len=decode_q_len,
    )
    decode_cfgs = decode_config.make_config(
        model_cfgs,
        serve_cfgs,
        effective_decode,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = prefill_config.make_config(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_prefill,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
    )

    shared_plan = sliding_window_schedule.SchedulePlan.create(
        decode_cfgs,
        num_shared_groups=num_groups,
    )
    decode_schedules = jax.lax.cond(
        distribution[0] > 0,
        lambda: sliding_window_schedule.generate_rpa_metadata_shared(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices_list,
            cfgs=decode_cfgs,
        ),
        lambda: tuple(
            _empty_schedule_like(
                sliding_window_schedule.SlidingWindowSchedule,
                decode_cfgs,
                plan=shared_plan,
            ) for _ in range(num_groups)),
    )
    mixed_schedules = [
        jax.lax.cond(
            distribution[2] > distribution[1],
            lambda page_indices=page_indices:
            (sliding_window_prefill_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=mixed_cfgs,
            )),
            lambda: _empty_schedule_like(
                sliding_window_prefill_schedule.SlidingWindowSchedule,
                mixed_cfgs,
            ),
        ) for page_indices in page_indices_list
    ]
    return [(decode_schedules[group], mixed_schedules[group])
            for group in range(num_groups)]


def build_schedules_prepacked_kv(
    queries: jax.Array,
    prepacked_new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    visibility: jax.Array | None = None,
    *,
    num_kv_heads: int,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    vmem_limit_bytes: int | None = None,
    out_dtype: jnp.dtype | None = None,
    decode_block_sizes: decode_blocks.BlockSizes | None = None,
    prefill_block_sizes: prefill_blocks.BlockSizes | None = None,
    update_kv_cache: bool = True,
):
    """Precompute schedules for the prepacked-KV SEQ_ALONG_LANE entry point."""
    if visibility is not None:
        raise ValueError("visibility is not supported by stacked RPA.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        _out_dtype,
        _mask_value,
    ) = _resolve_attn_static(
        queries,
        None,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        num_kv_heads=num_kv_heads,
        decode_q_len=decode_q_len,
    )
    decode_cfgs = decode_config.make_config(
        model_cfgs,
        serve_cfgs,
        effective_decode,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = prefill_config.make_config(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_prefill,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
    )
    _validate_prepacked_inputs(
        decode_cfgs,
        queries,
        prepacked_new_kv_hbm,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
    # Stacked RPA is SEQ_ALONG_LANE-only and runs native multi-token (spec)
    # decode on the stacked DECODE kernel, so spec-decode sequences stay in the
    # caller's decode bucket without forcing decode_q_len > 1 into MIXED.
    if model_cfgs.sliding_window is None:
        decode_schedule_hbm = non_sliding_window_schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=decode_cfgs,
        )
    else:
        decode_schedule_hbm = sliding_window_schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=decode_cfgs,
        )
    if model_cfgs.sliding_window is None:
        mixed_schedule = non_sliding_window_prefill_schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=mixed_cfgs,
        )
    else:
        mixed_schedule = sliding_window_prefill_schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            page_indices,
            cfgs=mixed_cfgs,
        )
    return decode_schedule_hbm, mixed_schedule


@jax.jit(
    static_argnames=(
        "num_kv_heads",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "decode_q_len",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
        "dispatch_hint",
    ),
    donate_argnames=("queries", "kv_cache"),
)
def ragged_paged_attention_prepacked_kv(
    queries: jax.Array,
    prepacked_new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    visibility: jax.Array | None = None,
    *,
    num_kv_heads: int,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    decode_block_sizes: decode_blocks.BlockSizes | None = None,
    prefill_block_sizes: prefill_blocks.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    precomputed_schedules: tuple | None = None,
    dispatch_hint: str = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Perform stacked RPA with K/V already in SEQ_ALONG_LANE ``new_kv_hbm``."""

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if attention_sink is not None:
        raise ValueError("attention_sink is not supported by stacked RPA.")
    if visibility is not None:
        raise ValueError("visibility is not supported by stacked RPA.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")

    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        out_dtype,
        mask_value,
    ) = _resolve_attn_static(
        queries,
        None,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        num_kv_heads=num_kv_heads,
        decode_q_len=decode_q_len,
    )
    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]

    q_hbm = prepare_queries(queries, num_kv_heads, queries.dtype)

    decode_cfgs = decode_config.make_config(
        model_cfgs,
        serve_cfgs,
        effective_decode,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = prefill_config.make_config(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_prefill,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
    )
    for cfgs in (decode_cfgs, mixed_cfgs):
        _validate_prepacked_inputs(
            cfgs,
            queries,
            prepacked_new_kv_hbm,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

    def run_decode(args):
        o_hbm_alias_q_hbm, current_kv_cache = args
        if precomputed_schedules is not None:
            schedule_hbm = precomputed_schedules[0]
        elif model_cfgs.sliding_window is None:
            schedule_hbm = non_sliding_window_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=decode_cfgs,
            )
        else:
            schedule_hbm = sliding_window_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=decode_cfgs,
            )

        if model_cfgs.sliding_window is None:
            if not isinstance(schedule_hbm,
                              non_sliding_window_schedule.DecodeSchedule):
                raise TypeError(
                    "Global decode requires DecodeSchedule metadata.")
            return non_sliding_window_kernel.rpa_kernel(
                cu_q_lens,
                kv_lens,
                schedule_hbm,
                o_hbm_alias_q_hbm,
                prepacked_new_kv_hbm,
                current_kv_cache,
                cfgs=decode_cfgs,
            )
        if not isinstance(schedule_hbm,
                          sliding_window_schedule.SlidingWindowSchedule):
            raise TypeError(
                "Sliding-window decode requires SlidingWindowSchedule.")
        return sliding_window_kernel.rpa_kernel(
            schedule_hbm,
            o_hbm_alias_q_hbm,
            prepacked_new_kv_hbm,
            current_kv_cache,
            cfgs=decode_cfgs,
        )

    def run_mixed(args):
        o_hbm_alias_q_hbm, current_kv_cache = args
        if precomputed_schedules is not None:
            schedule_hbm = precomputed_schedules[1]
        elif model_cfgs.sliding_window is None:
            schedule_hbm = non_sliding_window_prefill_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=mixed_cfgs,
            )
        else:
            schedule_hbm = sliding_window_prefill_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=mixed_cfgs,
            )
        if model_cfgs.sliding_window is None:
            if not isinstance(
                    schedule_hbm,
                    non_sliding_window_prefill_schedule.PrefillSchedule):
                raise TypeError(
                    "Global mixed attention requires PrefillSchedule.")
            return non_sliding_window_prefill_kernel.rpa_kernel(
                cu_q_lens,
                kv_lens,
                schedule_hbm,
                o_hbm_alias_q_hbm,
                prepacked_new_kv_hbm,
                current_kv_cache,
                cfgs=mixed_cfgs,
            )
        if not isinstance(
                schedule_hbm,
                sliding_window_prefill_schedule.SlidingWindowSchedule):
            raise TypeError(
                "Sliding-window mixed attention requires SlidingWindowSchedule."
            )
        return sliding_window_prefill_kernel.rpa_kernel(
            schedule_hbm,
            o_hbm_alias_q_hbm,
            prepacked_new_kv_hbm,
            current_kv_cache,
            cfgs=mixed_cfgs,
        )

    if dispatch_hint == "mixed_only":
        o_hbm_alias_q_hbm = q_hbm
    else:
        decode_start, decode_end = configs.RpaCase.DECODE.get_range(
            distribution)
        o_hbm_alias_q_hbm, kv_cache = jax.lax.cond(
            decode_end > decode_start,
            run_decode,
            lambda args: args,
            (q_hbm, kv_cache),
        )
    if dispatch_hint != "decode_only":
        mixed_start, mixed_end = configs.RpaCase.MIXED.get_range(distribution)
        o_hbm_alias_q_hbm, kv_cache = jax.lax.cond(
            mixed_end > mixed_start,
            run_mixed,
            lambda args: args,
            (o_hbm_alias_q_hbm, kv_cache),
        )

    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    return o_hbm, kv_cache


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
        "decode_q_len",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
        "dispatch_hint",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
    compiler_options={},
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
    attention_sink: jax.Array | None = None,
    visibility: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_q_len: int = 1,
    decode_block_sizes: decode_blocks.BlockSizes | None = None,
    prefill_block_sizes: prefill_blocks.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    precomputed_schedules: tuple | None = None,
    prepacked_new_kv_hbm: jax.Array | None = None,
    dispatch_hint: str = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    ``precomputed_schedules``: optional ``(decode_schedule, mixed_schedule)``
    tuple of mode-specific schedules from ``build_schedules(...)``. When
    provided, the per-call ``generate_rpa_metadata`` is skipped and these are
    used instead (lets a caller hoist the schedule build to once-per-step). It
    is a runtime (traced) value, so it is NOT a static argument.

    ``prepacked_new_kv_hbm``: optional post-RoPE K/V already packed in the
    SEQ_ALONG_LANE ``new_kv_hbm`` layout. When provided, ``prepare_inputs`` skips
    the K/V concat+pad+transpose and only prepares Q.

    Args:
        queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
        keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
        values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
        kv_cache: [num_pages, 2 * num_kv_heads, aligned_head_dim, page_size].
            Stores existing K/V cache data with K and V interleaved on the head axis.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
            page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
            sequences. distribution[2] represents total number of sequences.
        attention_sink: Not supported by stacked RPA.
        visibility: Not supported by stacked RPA.
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
        new_kv_cache: Same shape as ``kv_cache``.
    """

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if attention_sink is not None:
        raise ValueError("attention_sink is not supported by stacked RPA.")
    if visibility is not None:
        raise ValueError("visibility is not supported by stacked RPA.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")

    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        out_dtype,
        mask_value,
    ) = _resolve_attn_static(
        queries,
        keys,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        decode_q_len=decode_q_len,
    )
    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        prepacked_new_kv_hbm=prepacked_new_kv_hbm,
    )

    decode_cfgs = decode_config.make_config(
        model_cfgs,
        serve_cfgs,
        effective_decode,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = prefill_config.make_config(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_prefill,
        vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
    )
    for cfgs in (decode_cfgs, mixed_cfgs):
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
        )

    def run_decode(args):
        o_hbm_alias_q_hbm, current_kv_cache = args
        if precomputed_schedules is not None:
            schedule_hbm = precomputed_schedules[0]
        elif model_cfgs.sliding_window is None:
            schedule_hbm = non_sliding_window_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=decode_cfgs,
            )
        else:
            schedule_hbm = sliding_window_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=decode_cfgs,
            )

        if model_cfgs.sliding_window is None:
            if not isinstance(schedule_hbm,
                              non_sliding_window_schedule.DecodeSchedule):
                raise TypeError(
                    "Global decode requires DecodeSchedule metadata.")
            return non_sliding_window_kernel.rpa_kernel(
                cu_q_lens,
                kv_lens,
                schedule_hbm,
                o_hbm_alias_q_hbm,
                new_kv_hbm,
                current_kv_cache,
                cfgs=decode_cfgs,
            )
        if not isinstance(schedule_hbm,
                          sliding_window_schedule.SlidingWindowSchedule):
            raise TypeError(
                "Sliding-window decode requires SlidingWindowSchedule.")
        return sliding_window_kernel.rpa_kernel(
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            current_kv_cache,
            cfgs=decode_cfgs,
        )

    def run_mixed(args):
        o_hbm_alias_q_hbm, current_kv_cache = args
        if precomputed_schedules is not None:
            schedule_hbm = precomputed_schedules[1]
        elif model_cfgs.sliding_window is None:
            schedule_hbm = non_sliding_window_prefill_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=mixed_cfgs,
            )
        else:
            schedule_hbm = sliding_window_prefill_schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=mixed_cfgs,
            )
        if model_cfgs.sliding_window is None:
            if not isinstance(
                    schedule_hbm,
                    non_sliding_window_prefill_schedule.PrefillSchedule):
                raise TypeError(
                    "Global mixed attention requires PrefillSchedule.")
            return non_sliding_window_prefill_kernel.rpa_kernel(
                cu_q_lens,
                kv_lens,
                schedule_hbm,
                o_hbm_alias_q_hbm,
                new_kv_hbm,
                current_kv_cache,
                cfgs=mixed_cfgs,
            )
        if not isinstance(
                schedule_hbm,
                sliding_window_prefill_schedule.SlidingWindowSchedule):
            raise TypeError(
                "Sliding-window mixed attention requires SlidingWindowSchedule."
            )
        return sliding_window_prefill_kernel.rpa_kernel(
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            current_kv_cache,
            cfgs=mixed_cfgs,
        )

    if dispatch_hint == "mixed_only":
        o_hbm_alias_q_hbm = q_hbm
    else:
        decode_start, decode_end = configs.RpaCase.DECODE.get_range(
            distribution)
        o_hbm_alias_q_hbm, kv_cache = jax.lax.cond(
            decode_end > decode_start,
            run_decode,
            lambda args: args,
            (q_hbm, kv_cache),
        )
    if dispatch_hint != "decode_only":
        mixed_start, mixed_end = configs.RpaCase.MIXED.get_range(distribution)
        o_hbm_alias_q_hbm, kv_cache = jax.lax.cond(
            mixed_end > mixed_start,
            run_mixed,
            lambda args: args,
            (o_hbm_alias_q_hbm, kv_cache),
        )

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    return o_hbm, kv_cache
