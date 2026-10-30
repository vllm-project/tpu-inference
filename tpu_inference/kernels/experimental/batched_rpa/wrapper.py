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

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference import envs
from tpu_inference.kernels.experimental.batched_rpa import (configs, kernel,
                                                            schedule, utils)
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import \
    get_tuned_params


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    k_scale: jax.Array | None = None,
    v_scale: jax.Array | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
    attention_sink: jax.Array | None = None,  # f32[actual_num_q_heads]
) -> tuple[jax.Array, jax.Array, jax.Array | None]:

    total_q_tokens, actual_num_q_heads, actual_q_head_dim = q.shape
    _, actual_num_kv_heads, actual_kv_head_dim = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_q_head_dim, num_lanes)

    # Compute aligned kv head dimension, accounting for per-token scale if
    # present.
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if k_scale is not None and v_scale is not None:
            scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
            aligned_kv_head_dim = utils.align_to(
                actual_kv_head_dim + num_scale_channels,
                num_sublanes * kv_packing)
        else:
            aligned_kv_head_dim = utils.align_to(actual_kv_head_dim,
                                                 num_sublanes * kv_packing)
    else:
        aligned_kv_head_dim = utils.align_to(actual_kv_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (jnp.pad(
        q.reshape(
            total_q_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head,
            actual_q_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_q_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        actual_num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    ).swapaxes(0, 1))

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2,
                                             kv_packing)

    if k_scale is not None and v_scale is not None:
        k_scale = k_scale.reshape(total_q_tokens, actual_num_kv_heads)
        v_scale = v_scale.reshape(total_q_tokens, actual_num_kv_heads)

        scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        num_scale_channels = max(1, scale_bits // kv_bits)

        # Bitcast the scale factors to the kv dtype and reshape scale factors to
        # (T, H_kv, num_scale_channels).
        k_scale_split = jax.lax.bitcast_convert_type(k_scale, k.dtype).reshape(
            total_q_tokens, actual_num_kv_heads, num_scale_channels)
        v_scale_split = jax.lax.bitcast_convert_type(v_scale, v.dtype).reshape(
            total_q_tokens, actual_num_kv_heads, num_scale_channels)

        # k/v have shape (T, H_kv, D) -> (T, H_kv, D + num_scale_channels)
        k = jnp.concatenate([k, k_scale_split], axis=-1)
        v = jnp.concatenate([v, v_scale_split], axis=-1)

        actual_kv_head_dim += num_scale_channels

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        align_tokens = max(num_lanes, page_size)
        padded_total_tokens = utils.align_to(total_q_tokens, align_tokens)
        new_kv_hbm = (jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_kv_head_dim),
            (
                (0, padded_total_tokens - total_q_tokens),
                (0, 0),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        ).reshape(
            padded_total_tokens,
            actual_num_kv_heads_x2,
            aligned_kv_head_dim // kv_packing,
            kv_packing,
        ).transpose(1, 2, 3, 0))
    else:
        new_kv_hbm = jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_kv_head_dim),
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_q_tokens,
            num_kv_heads_x2_aligned // kv_packing,
            kv_packing,
            aligned_kv_head_dim,
        )

    if attention_sink is not None:
        attention_sink = jnp.pad(
            attention_sink.reshape(actual_num_kv_heads,
                                   num_q_heads_per_kv_head),
            (
                (0, 0),
                (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            ),
            constant_values=-jnp.inf,
        )
        attention_sink = jnp.broadcast_to(attention_sink[..., None],
                                          (*attention_sink.shape, num_lanes))

    return o_hbm_alias_q_hbm, new_kv_hbm, attention_sink


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def _default_kv_layout() -> configs.KVLayout:
    """KV layout selected by the USE_BATCHED_RPA_SEQ_ON_LANE env flag."""
    if envs.USE_BATCHED_RPA_SEQ_ON_LANE:
        return configs.KVLayout.SEQ_ALONG_LANE
    return configs.KVLayout.HEAD_ALONG_SUBLANE


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    kv_layout: configs.KVLayout | None = None,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
    use_per_token_scale: bool = False,
    scale_dtype: jnp.dtype | None = None,
):
    if kv_layout is None:
        kv_layout = _default_kv_layout()
    chip_info = pltpu.get_tpu_info_for_chip(chip_version, 1)
    num_lanes, num_sublanes = chip_info.num_lanes, chip_info.num_sublanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_scale_channels = 0
        if use_per_token_scale:
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            scale_bits = jax.dtypes.itemsize_bits(scale_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
        base_dim = actual_head_dim + num_scale_channels
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            utils.align_to(base_dim, num_sublanes * kv_packing) // kv_packing,
            kv_packing,
            page_size,
        )
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "use_per_token_scale",
        "per_token_scale_dtype",
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
        "skip_kv_update",
        "update_kv_cache",
        "kv_layout",
        "decode_query_size",
        "cp_group_size",
        "attention_scope",
        "return_lse",
        "quantize_p",
    ),
    # Donation of transient inputs can fail for some runtime buffer layouts in
    # the experimental tuning path. Keep donation only for kv_cache, which is
    # the intended long-lived mutable state.
    donate_argnames=("kv_cache", ),
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
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    use_per_token_scale: bool = False,
    per_token_scale_dtype: jnp.dtype | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    skip_kv_update: bool = False,
    update_kv_cache: bool = True,
    kv_layout: configs.KVLayout | None = None,
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    quantize_p: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
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
            sequences (cumulative end indices for decode, prefill-only, and mixed).
            The prefill-only segment is handled by the mixed kernel call.
            distribution[2] represents total number of sequences.
        attention_sink: [num_q_heads]. Per-head logit that joins the softmax
            denominator without contributing to the output.
        sm_scale: Softmax scale value.
        sliding_window: Size of sliding window (also known as local attention). kvs
            outside of the window is not fetched from hbm and masked out during
            computation.
        soft_cap: Cap values of softmax inputs.
        mask_value: Value to use for causal masking. Defaults to smallest
            representable value of the activation dtype.
        use_per_token_scale: Whether to use per-token quantization.
        per_token_scale_dtype: Dtype of per-token scale. Defaults to None.
        q_scale: Quantization scale value of queries.
        k_scale: Per-tensor quantization scale value of keys.
        v_scale: Per-tensor quantization scale value of values.
        dynamic_k_scale: Per-token quantization scale value of keys.
        dynamic_v_scale: Per-token quantization scale value of values.
        chunk_prefill_size: Not used.
        decode_block_sizes: Kernel block size to use during decode.
        prefill_block_sizes: Kernel block size to use during prefill.
        vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
            size of the hardware.
        debug_mode: Not used.
        out_dtype: Dtype of output. Defaults to dtype of queries.
        use_causal_mask: Not used.
        skip_kv_update: If True, don't write keys/values back to kv_cache
            (KV-share). All kv, including this step's tokens, is read from kv_cache.
        update_kv_cache: update_kv_cache=False (KV-share) is the same as
            skip_kv_update=True. Defaults to True, which leaves skip_kv_update as
            passed.
        kv_layout: KV cache layout. Defaults to the layout selected by the
            USE_BATCHED_RPA_SEQ_ON_LANE env flag.
        decode_query_size: Number of query tokens in decode (1 by default, can be
            higher in case of speculative decoding).
        cp_group_size: Size of the context parallelism (CP) group. KV cache is
            sharded across devices in this group. Defaults to None.
        cp_rank: Rank of the current device within the CP group, which determine the
            token ownership. Defaults to None.
        attention_scope: Which KV positions to attend to. FULL attends all
            positions, CACHE_ONLY skips new tokens, NEW_TOKENS_ONLY skips cached
            tokens. Defaults to FULL.
        return_lse: If True, return log-sum-exp (lse) values along with the output.
            Defaults to False.
        quantize_p: If True, cast the softmax probabilities down to the kv cache
            dtype before the pv matmul. Required to get a native fp8 pv matmul when
            the kv cache is fp8, at the cost of pv precision. Defaults to False.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Result of new kv cache where k & vs are
            concatenated along num kv heads dim.
        lse (only when return_lse=True): [max_num_tokens, num_q_heads].
            Log-sum-exp values (m + log(l)) for each query token and head,
            needed for merging partial attention results in CP.
    """

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    if kv_layout is None:
        kv_layout = _default_kv_layout()

    # update_kv_cache=False (KV-share, as passed by tpu-inference) is the same
    # as skip_kv_update=True.
    skip_kv_update = skip_kv_update or not update_kv_cache

    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
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

    if k_scale is not None and dynamic_k_scale is not None:
        raise ValueError(
            "Only one of k_scale or dynamic_k_scale can be set. Got"
            f" {k_scale=} and {dynamic_k_scale=}")
    if v_scale is not None and dynamic_v_scale is not None:
        raise ValueError(
            "Only one of v_scale or dynamic_v_scale can be set. Got"
            f" {v_scale=} and {dynamic_v_scale=}")

    k_scale_config = k_scale
    v_scale_config = v_scale
    k_scale_tensor = dynamic_k_scale
    v_scale_tensor = dynamic_v_scale

    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        scale_q=q_scale,
        scale_k=k_scale_config,
        scale_v=v_scale_config,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        skip_kv_update=skip_kv_update,
        quantize_p=quantize_p,
    )

    q_hbm, new_kv_hbm, attention_sink_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        k_scale_tensor,
        v_scale_tensor,
        kv_layout=kv_layout,
        page_size=page_size,
        attention_sink=attention_sink,
    )

    # Pre-allocate LSE buffer.
    lse_hbm_init: jax.Array | None = None
    if return_lse:
        num_lanes = pltpu.get_tpu_info().num_lanes
        q_packing = utils.get_dtype_packing(queries.dtype)
        num_q_heads_per_kv_head = num_q_heads // num_kv_heads
        aligned_num_q_heads_per_kv_head = utils.align_to(
            num_q_heads_per_kv_head, q_packing)
        max_tokens = queries.shape[0]
        lse_hbm_init = jnp.full(
            [
                num_kv_heads,
                max_tokens,
                aligned_num_q_heads_per_kv_head // q_packing,
                q_packing,
                num_lanes,
            ],
            -jnp.inf,
            dtype=out_dtype,
        )

    # Compute per-sequence length parameters for the kernel.
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    global_kv_cache_lens = kv_lens - q_lens

    if attention_scope == configs.AttentionScope.CACHE_ONLY:
        if cp_group_size is not None:
            rank = cp_rank[0] if cp_rank is not None else 0
            kv_cache_lens = utils.cp_local_cache_len(global_kv_cache_lens,
                                                     cp_group_size, rank,
                                                     page_size)
        else:
            kv_cache_lens = global_kv_cache_lens
        kv_new_lens = jnp.zeros_like(q_lens)
        q_offsets = kv_cache_lens
    elif attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
        kv_cache_lens = global_kv_cache_lens
        kv_new_lens = q_lens
        q_offsets = global_kv_cache_lens
    else:  # FULL
        kv_cache_lens = global_kv_cache_lens
        kv_new_lens = q_lens
        q_offsets = global_kv_cache_lens

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
        lse_hbm_in: jax.Array | None,
    ):
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or get_tuned_params(
                model_cfgs,
                serve_cfgs,
                case="decode",
                vmem_limit_bytes=vmem_limit_bytes,
                decode_query_size=decode_query_size,
            )
        else:
            effective_blocks = prefill_block_sizes or get_tuned_params(
                model_cfgs,
                serve_cfgs,
                case="prefill",
                vmem_limit_bytes=vmem_limit_bytes,
            )

        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=attention_sink,
        )
        # Select metadata computer class.
        if cp_group_size is not None:
            # TODO(kwang3939): schedule_cp.py has not been synced from google3 yet.
            from tpu_inference.kernels.experimental.batched_rpa import \
                schedule_cp
            computer_cls = schedule_cp.CPMetadataComputer
            extra_scalars = (cp_rank, ) if cp_rank is not None else ()
        else:
            computer_cls = schedule.BaseMetadataComputer
            extra_scalars = ()

        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            distribution,
            cfgs=cfgs,
            computer_cls=computer_cls,
            extra_scalars=extra_scalars,
        )
        result = kernel.rpa_kernel(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            page_indices,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            lse_hbm_in,
            attention_sink_hbm,
            cfgs=cfgs,
            computer_cls=computer_cls,
        )
        if return_lse:
            o_out, kv_out, lse_out = result
        else:
            o_out, kv_out, _ = result
            lse_out = None
        return o_out, kv_out, lse_out

    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init)
    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm)

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    if not return_lse:
        return o_hbm, kv_cache

    # Reshape LSE from [num_kv_heads, max_tokens, q_per_kv // q_packing, q_packing
    # , num_lanes] to [max_tokens, num_q_heads].
    max_tokens = queries.shape[0]
    # Extract first lane (scalar LSE value per token-head pair).
    lse = lse_hbm.reshape(num_kv_heads, max_tokens,
                          aligned_num_q_heads_per_kv_head,
                          num_lanes)[:, :max_tokens, :num_q_heads_per_kv_head,
                                     0]
    lse = lse.swapaxes(0, 1).reshape(max_tokens, num_q_heads)

    return o_hbm, kv_cache, lse
