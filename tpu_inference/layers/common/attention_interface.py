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

import functools
import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import Sharding

import tpu_inference.kernels.experimental.rpa_v3_cp.kernel as rpa_v3_cp
import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from tpu_inference import envs
from tpu_inference.kernels.flash_attention.kernel import (
    encoder_only_flash_attention, flash_attention)
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.mla.v2.tuned_params import (TuningKey,
                                                       get_tuned_params)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_megacore, get_mesh_shape_product

logger = init_logger(__name__)

MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.

# NOTE: this kernel is experimental and not fully tested.  See
# tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py
# for details
if envs.USE_BATCHED_RPA_KERNEL:
    import tpu_inference.kernels.experimental.batched_rpa.wrapper as rpa
    logger.info_once("Using experimental batched RPA kernel")
else:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
    logger.info_once("Using default RPA kernel")

ragged_paged_attention = rpa.ragged_paged_attention
get_kv_cache_shape = rpa.get_kv_cache_shape

ragged_paged_attention_hd64 = rpa_hd64.ragged_paged_attention_hd64
get_kv_cache_shape_hd64 = rpa_hd64.get_kv_cache_shape


def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    vmem_limit_bytes: int | None = None,
    use_attention_bias: bool = False,
    batch_axis="data",
    head_axis="model",
) -> Callable[..., Any]:
    if use_attention_bias:
        in_specs = (
            P(batch_axis, head_axis, None, None),  # q
            P(batch_axis, head_axis, None, None),  # k
            P(batch_axis, head_axis, None, None),  # v
            P(batch_axis, head_axis, None, None),  # attention_bias
            P(batch_axis,
              None),  # segment_ids (B matches q's B, so shard 'data')
        )
        out_specs = P(batch_axis, head_axis, None, None)

        def _flash_attention_use_ab(q, k, v, attention_bias, segment_ids):
            return flash_attention(q,
                                   k,
                                   v,
                                   ab=attention_bias,
                                   segment_ids=segment_ids,
                                   sm_scale=sm_scale,
                                   causal=causal,
                                   vmem_limit_bytes=vmem_limit_bytes)

        attn_fn = _flash_attention_use_ab
    else:
        in_specs = (
            P(batch_axis, head_axis, None, None),  # q
            P(batch_axis, head_axis, None, None),  # k
            P(batch_axis, head_axis, None, None),  # v
            P(batch_axis,
              None),  # segment_ids (B matches q's B, so shard 'data')
        )
        out_specs = P(batch_axis, head_axis, None, None)

        def _flash_attention(q, k, v, segment_ids):
            return flash_attention(q,
                                   k,
                                   v,
                                   segment_ids=segment_ids,
                                   sm_scale=sm_scale,
                                   causal=causal,
                                   vmem_limit_bytes=vmem_limit_bytes)

        attn_fn = _flash_attention

    return jax.jit(
        jax.shard_map(attn_fn,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))


def sharded_encoder_only_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    vmem_limit_bytes: int | None = None,
) -> Callable[..., Any]:
    in_specs = (
        P(None, "model", None),  # q: [q_len, num_heads, head_size]
        P(None, "model", None),  # k: [k_len, num_kv_heads, head_size]
        P(None, "model", None),  # v: [k_len, num_kv_heads, head_size]
        P(),  # seq_lens: [batch_size]
    )
    out_specs = P(None, "model", None)

    def _flash_attention(q, k, v, seq_lens):
        return encoder_only_flash_attention(
            q,
            k,
            v,
            seq_lens,
            causal=causal,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    return jax.jit(
        jax.shard_map(
            _flash_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: Optional[float] = None,
) -> Callable[..., Any]:
    """Shards GQA PagedAttention along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P("model", None, None, None),  # k
        P("model", None, None, None),  # v
        P(),  # lengths
        P(),  # page_indices
    )
    out_specs = P(None, "model", None)

    def _paged_attention_fn(q, k, v, lengths, page_indices):
        if page_indices.size > MAX_ALLOWED_PAGE_INDICES_N:
            raise ValueError(
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` to run with minibatches."
            )
        return paged_attention(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        jax.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


# TODO(xiangxu): merge this with sharded_paged_attention
@jax.jit(static_argnames=["paged_attention_kernel"])
def paged_attention_with_guarded_smem(
    paged_attention_kernel: Callable,
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
):
    # Addresses b/336316706. Summary:
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent. Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths,
                                      page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(q[i], k_pages, v_pages, seq_lens[i],
                                   block_indices[i]))

    outputs = outputs.reshape((batch_size, *outputs.shape[2:]))

    return outputs


# ruff: noqa: E741
def update_cache(
    is_prefill,
    cache,
    indices,
    operand,
    prefill_seq_len=None,
    sliding_window=None,
) -> jax.Array:

    # (8, 55640, 32, 128) (1, 8, 256, 128) -> K (8, 8, 32, 128)
    # I = B * T // S
    # k cache, operand

    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op inserted by
    #    the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from actual prompt, not from the padded tokens.
        if sliding_window and T > sliding_window:
            assert B == 1
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            operand = jax.lax.dynamic_slice_in_dim(
                operand, start_index, sliding_window,
                axis=2)  # TODO: @pooyam Perf check this.
            T = sliding_window

        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)
        indices = indices.reshape(I)
        cache = cache.at[:, indices, :, :].set(operand)
    else:
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        cache = cache.reshape(K, L * S, H)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache


@jax.jit(static_argnames=["window_size", "attn_logits_soft_cap", "is_mqa"])
def apply_splash(q, k, v, window_size, attn_logits_soft_cap,
                 is_mqa) -> jax.Array:
    # q: (batch_size, num_heads, seq_len, head_dim)
    num_heads = q.shape[1]
    q_seq_len = q.shape[2]
    kv_seq_len = k.shape[2]
    assert kv_seq_len >= q_seq_len

    masks = [
        mask_lib.LocalMask((q_seq_len, kv_seq_len), (window_size, 0),
                           kv_seq_len - q_seq_len) for _ in range(num_heads)
    ]
    mask = mask_lib.MultiHeadMask(tuple((m for m in masks)))
    block_sizes = splash.BlockSizes.get_default()

    if is_mqa:
        attn = splash.make_splash_mqa_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    else:
        attn = splash.make_splash_mha_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    attn = jax.vmap(attn)
    outputs = attn(q, k, v, None)

    return outputs


def sharded_splash_attention(
    mesh: Mesh,
    window_size: Optional[int] = None,
    attn_logits_soft_cap: Optional[float] = None,
    is_mqa: bool = False,
) -> Callable[..., Any]:
    in_specs = (
        P("data", "model", None, None),  # q
        P("data", "model", None, None),  # k
        P("data", "model", None, None),  # vx
    )
    out_specs = P("data", "model", None, None)
    return jax.jit(
        jax.shard_map(
            functools.partial(
                apply_splash,
                window_size=window_size,
                attn_logits_soft_cap=attn_logits_soft_cap,
                is_mqa=is_mqa,
            ),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


def sharded_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    use_causal_mask: bool = True,
):
    """Shards along KV heads."""
    # Handle GQA/MQA where num_kv_heads < tp_size
    # We replicate KV heads to match tp_size so that we can shard them evenly.
    # TODO (ranlihao): This is not performant and introduces extra overhead during inference. We need to handle this during weight loading
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    if tp_size > 1:
        num_kv_heads = k.shape[1]
        if num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"For GQA/MQA, tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}"
                )
            factor = tp_size // num_kv_heads
            k = jnp.repeat(k, factor, axis=1)
            v = jnp.repeat(v, factor, axis=1)

    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD, None, None)
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        kv_cache_spec,  # kv cache
        P(ShardingAxisName.ATTN_DATA),  # kv_lens
        P(ShardingAxisName.ATTN_DATA),  # page_indices
        P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
        P(ShardingAxisName.ATTN_DATA),  # distribution
    )
    out_specs = (qkv_spec, kv_cache_spec)

    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    use_hd64 = q.shape[-1] == 64
    func = ragged_paged_attention_hd64 if use_hd64 else ragged_paged_attention

    if attention_sink is not None:
        if not use_hd64:
            raise NotImplementedError(
                "Attention sink support is only available when head_dim==64")

        in_specs += (P(ShardingAxisName.ATTN_HEAD), )
        args += (attention_sink, )

    # update_kv_cache=False (KV-share) is supported by the v3 default RPA
    # kernel and by the experimental batched RPA kernel. The hd64 path
    # doesn't accept it; fail loud rather than silently ignoring.
    if use_hd64 and not update_kv_cache:
        raise NotImplementedError(
            "update_kv_cache=False (KV-share) is not supported on the "
            "head_dim==64 RPA kernel.")

    def _ragged_paged_attention(*args):
        kwargs = dict(
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # update_kv_cache is supported by both the v3 default and batched
        # RPA kernels; only the hd64 path doesn't accept it. Default True
        # is a no-op so we don't forward it to the hd64 signature.
        if not use_hd64:
            kwargs["update_kv_cache"] = update_kv_cache
            kwargs["use_causal_mask"] = use_causal_mask
        return func(*args, **kwargs)

    return jax.shard_map(
        _ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def sharded_ragged_paged_attention_experimental(
        mesh: Mesh,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_cache: jax.Array,
        kv_lens: jax.Array,
        paged_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        attention_sink: jax.Array | None,
        sm_scale: float,
        attention_chunk_size: int | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        # kv_cache_lens: jax.Array | None = None,
        # Flags for CP
        update_kv_cache: bool = True,
        return_lse: bool = False,
        skip_cache_attn: bool = False,
        skip_current_attn: bool = False,
        is_context_phase: bool = False,
        use_causal_mask: bool = True):
    # Determine the Pallas kernel block sizes.
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    if tp_size > 1:
        num_kv_heads = k.shape[1]
        if num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"For GQA/MQA, tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}"
                )
            factor = tp_size // num_kv_heads
            k = jnp.repeat(k, factor, axis=1)
            v = jnp.repeat(v, factor, axis=1)

    if is_context_phase:
        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_CACHE_HEAD,
                   None)
        o_spec = P(('data', 'attn_dp', 'dcp'), ShardingAxisName.KV_CACHE_HEAD,
                   None)
    else:
        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD,
                   None)
        o_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD,
                   None)
    # Define the sharding specs.
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    # KV cache is sharded across DCP and TP.
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT,
                      ShardingAxisName.KV_CACHE_HEAD, None, None)
    print(f"page_size={kv_cache.shape[1]}")

    # Build a global cp_rank array of shape (dcp_size,) sharded along 'dcp'.
    # Inside shard_map each device receives a (1,) slice containing its rank.
    dcp_size = mesh.shape['dcp']
    cp_rank_global = jnp.arange(dcp_size, dtype=jnp.int32)

    in_specs = [
        q_spec,  # q
        kv_spec,  # k
        kv_spec,  # v
        kv_cache_spec,  # kv cache
        P(ShardingAxisName.ATTN_DATA),  # kv_lens
        P(ShardingAxisName.ATTN_DATA),  # page_indices
        P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
        P(ShardingAxisName.ATTN_DATA),  # distribution
        P(ShardingAxisName.CONTEXT),  # cp_rank
    ]

    args = [
        q, k, v, kv_cache, kv_lens, paged_indices, cu_q_lens, distribution,
        cp_rank_global
    ]

    lse_spec = (P(
        ('data', 'attn_dp',
         'dcp'), ShardingAxisName.KV_CACHE_HEAD) if is_context_phase else P(
             ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD))
    out_specs = [o_spec, kv_cache_spec]
    if return_lse:
        out_specs.append(lse_spec)

    def _ragged_paged_attention_wrapper(*args):
        *kernel_args, cp_rank = args  # cp_rank is (1,) for this device
        cp_group_size = mesh.shape['dcp']

        kwargs = dict(sm_scale=sm_scale,
                      sliding_window=attention_chunk_size,
                      q_scale=q_scale,
                      k_scale=k_scale,
                      v_scale=v_scale,
                      cp_rank=cp_rank,
                      cp_group_size=cp_group_size,
                      update_kv_cache=update_kv_cache,
                      skip_cache_attn=skip_cache_attn,
                      skip_current_attn=skip_current_attn,
                      return_lse=return_lse,
                      use_causal_mask=use_causal_mask)
        return rpa_v3_cp.ragged_paged_attention(*kernel_args, **kwargs)

    return jax.shard_map(
        _ragged_paged_attention_wrapper,
        mesh=mesh,
        in_specs=tuple(in_specs),
        out_specs=tuple(out_specs),
        check_vma=False,
    )(*args)


def dcp_alltoall(
    attn_out: jax.
    Array,  # P('dcp', 'model'): local_shape=(max_num_tokens, heads/model, head_dim)
    lse: jax.
    Array,  # P('dcp', 'model'): local_shape=(max_num_tokens, heads/model)
    mesh: Mesh,
    dcp_axis: str = 'dcp',
    model_axis: str = 'model',
) -> tuple[jax.Array, jax.Array]:

    def _inner(attn_out, lse):
        dcp_size = jax.lax.psum(1, axis_name=dcp_axis)
        max_num_tokens = attn_out.shape[0]
        local_heads = attn_out.shape[1]
        head_dim = attn_out.shape[2]

        # Step 1: all-to-all across dcp
        attn_gathered = jax.lax.all_to_all(
            attn_out,
            axis_name=dcp_axis,
            split_axis=1,  # split heads
            concat_axis=0,  # concat tokens
            tiled=True,
        )  # -> (max_num_tokens*dcp, heads/(model*dcp), head_dim)

        lse_gathered = jax.lax.all_to_all(
            lse,
            axis_name=dcp_axis,
            split_axis=1,  # split heads
            concat_axis=0,  # concat tokens
            tiled=True,
        )  # -> (max_num_tokens*dcp, heads/(model*dcp))

        # Step 2: Reshape and make shape[0]=dcp_size
        new_local_heads = local_heads // dcp_size
        attn_chunks = attn_gathered.reshape(dcp_size, max_num_tokens,
                                            new_local_heads, head_dim)
        lse_chunks = lse_gathered.reshape(dcp_size, max_num_tokens,
                                          new_local_heads)

        # Step 3: Local lse correction
        combined_lse = jax.nn.logsumexp(lse_chunks, axis=0)

        weights = jnp.exp(lse_chunks - combined_lse[None])
        # NOTE(weiyulin): When all ranks have -inf LSE (e.g. prefill seqs with no cached tokens),
        # combined_lse=-inf and weights=exp(-inf-(-inf))=NaN.  Zero them out so
        # combined_out stays 0 and combined_lse stays -inf for those tokens,
        # letting merge_attn_states fall back to the query-phase result.
        weights = jnp.where(jnp.isneginf(combined_lse[None, ...]), 0.0,
                            weights)

        combined_out = jnp.einsum('d t h, d t h f -> t h f', weights,
                                  attn_chunks)
        # (max_num_tokens, new_local_heads, head_dim)

        return combined_out, combined_lse

    return jax.shard_map(
        _inner,
        mesh=mesh,
        in_specs=(
            P(dcp_axis, ShardingAxisName.KV_CACHE_HEAD, None),
            P(dcp_axis, ShardingAxisName.KV_CACHE_HEAD),
        ),
        out_specs=(
            P(None, ShardingAxisName.ATTN_HEAD, None),
            P(None, ShardingAxisName.ATTN_HEAD),
        ),
        check_vma=False,
    )(attn_out, lse)


def merge_attn_states(context_out: jax.Array, context_lse: jax.Array,
                      query_out: jax.Array,
                      query_lse: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Merged attn results based on Context (Cache) Query (Current)'s LSE
        context_out = [seq, local_heads, head_dim]
        context_lse = [seq, local_heads]
        query_out = [seq, local_heads, head_dim]
        query_lse = [seq, local_heads]
    """
    max_lse = jnp.maximum(context_lse, query_lse)
    exp_context = jnp.exp(context_lse - max_lse)
    exp_query = jnp.exp(query_lse - max_lse)

    sum_exp = exp_context + exp_query

    merged_out = (context_out * exp_context[..., None] +
                  query_out * exp_query[..., None]) / sum_exp[..., None]
    merged_lse = max_lse + jnp.log(sum_exp)

    return merged_out, merged_lse


def forward_with_dcp(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,
    sm_scale: float | None = None,
    sinks: jax.Array | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """
    DCP Attention forward pass.

    Phase 1: Context Attention (Attending to KV caches)
    Phase 2: Query Attention (Attending to current tokens K, V)
    Phase 3: Combine and Sharding
    """
    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata

    # ==========================================================================
    # Phase 1: Context Attention (Attending to KV caches)
    # `kv_caches` is not modified in the phase.
    # ==========================================================================

    context_attn_out, kv_cache, context_lse = sharded_ragged_paged_attention_experimental(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=md.seq_lens,
        paged_indices=md.block_tables,
        cu_q_lens=md.query_start_loc,
        distribution=md.request_distribution,
        attention_sink=sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=False,
        is_context_phase=True,
        return_lse=True,
        skip_current_attn=True,
        use_causal_mask=False)

    context_attn_out_cor, context_lse_cor = dcp_alltoall(context_attn_out,
                                                         context_lse,
                                                         mesh=mesh)

    # ==========================================================================
    # Phase 2: Query Attention (Attending to current tokens K, V)
    # ==========================================================================
    query_attn_out, updated_kv_cache, query_lse = sharded_ragged_paged_attention_experimental(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=md.seq_lens,
        paged_indices=md.block_tables,
        cu_q_lens=md.query_start_loc,
        distribution=md.request_distribution,
        attention_sink=sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=True,
        skip_cache_attn=True,
        return_lse=True,
    )

    # ==========================================================================
    # Phase 3: Combine Context and Query results
    # ==========================================================================
    final_output, _ = merge_attn_states(
        context_attn_out_cor,
        context_lse_cor,
        query_attn_out,
        query_lse,
    )

    return updated_kv_cache, final_output


def attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,  # before padding,
    sm_scale: float | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sinks: jax.Array | None = None,
    update_kv_cache: bool = True,
    use_causal_mask: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    # T: seq_len
    # N: num_heads
    # K: num_kv_heads
    # D: hidden_size
    # H: head_dim
    # L: num_blocks
    # S: block_size

    # TODO(jevinjiang, cuiq): transpose q weight offline.
    # q: (T, N, H)
    # k,v: (T, K, H)

    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata

    if mesh.shape['dcp'] > 1:
        return forward_with_dcp(
            kv_cache=kv_cache,
            q=q,
            k=k,
            v=v,
            attention_metadata=md,
            mesh=mesh,
            head_dim_original=head_dim_original,
            sm_scale=sm_scale,
            sinks=sinks,
            attention_chunk_size=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # (T, N, H)
    output, kv_cache = sharded_ragged_paged_attention(
        mesh,
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=update_kv_cache,
        use_causal_mask=use_causal_mask,
    )

    return kv_cache, output


def mla_attention(
        q_NTA: jax.Array,
        q_rope_TNH: jax.Array,
        k_SA: jax.Array,
        k_rope_SH: jax.Array,
        kv_cache: jax.Array,
        md: AttentionMetadata,
        mesh: Mesh,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        query_nth_sharding: Sharding | None = None,
        query_tnh_sharding: Sharding | None = None,
        keyvalue_skh_sharding: Sharding | None = None,
        attn_o_nth_sharding: Sharding | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        sm_scale: float | None = None) -> Tuple[jax.Array, jax.Array]:
    """
    Main shared interface for MLA attention.  Computes the sharded attention
    output and kv cache update.

    Args:
        q_NTA: (num_query_heads, tokens_query, q_lora_rank) # head-major output from q_nope einsum projection.
        q_rope_TNH: (tokens_query, num_query_heads, head_dim)
        k_SA: (tokens_kv, q_lora_rank)
        k_rope_SH: (tokens_kv, head_dim)
        kv_cache: KV cache to be retrieved from/updated
        md: attention metadata
        mesh: Mesh
        num_attention_heads: number of attention heads
        qk_nope_head_dim: head dim for QK without rope
        query_nth_sharding: sharding to use for q_nope for the shard map (MLA kernel)
        query_tnh_sharding: sharding to use for q_rope for the shard map (MLA kernel)
        keyvalue_skh_sharding: sharding to use for k/k_rope for the shard map (MLA kernel)
        attn_o_nth_sharding: sharding to use for the attention output for the shard map (MLA kernel)
        q_scale: scale to apply to q (if quantized)
        k_scale: scale to apply to k (if quantized)
        v_scale: scale to apply to v (if quantized)
        sm_scale: softmax scale
    """
    in_specs = (
        query_nth_sharding
        or P(None, ShardingAxisName.MLP_TENSOR, None),  # q (head-major)
        query_tnh_sharding
        or P(ShardingAxisName.MLP_TENSOR, None, None),  # q_rope (token-major)
        keyvalue_skh_sharding or P(ShardingAxisName.MLP_TENSOR, None),  # k
        keyvalue_skh_sharding
        or P(ShardingAxisName.MLP_TENSOR, None),  # k_rope
        P(ShardingAxisName.BATCH),  # kv_cache
        P(ShardingAxisName.ATTN_DATA),  # md.seq_lens
        P(ShardingAxisName.ATTN_DATA),  # md.page_indices_flat
        P(ShardingAxisName.ATTN_DATA),  # md.query_start_loc
        P(ShardingAxisName.ATTN_DATA),  # md.distribution
    )
    out_specs = (
        P(ShardingAxisName.BATCH),  # kv cache
        attn_o_nth_sharding
        or P(None, ShardingAxisName.MLP_TENSOR, None)  # attn output
    )

    def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, *args):
        dp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA)
        batched_decode_tuning_key = TuningKey(
            case="batched_decode",
            max_num_tokens=q.shape[1],
            actual_num_q_heads=q.shape[0],
            actual_lkv_dim=q.shape[2],
            actual_r_dim=q_rope.shape[2],
            kv_dtype=cache.dtype.name,
            q_dtype=q.dtype.name,
            page_size_per_kv_packing=cache.shape[1],
            kv_packing=cache.shape[2],
            max_num_seqs=md.padded_num_reqs // dp_size,
            pages_per_seq=args[1].shape[0] // args[0].shape[0],
        )
        batched_decode_tuned_params = get_tuned_params(
            batched_decode_tuning_key)
        num_kv_pages_per_block = (
            batched_decode_tuned_params.num_kv_pages_per_block, 1, 1)
        num_queries_per_block = (
            batched_decode_tuned_params.num_queries_per_block, 16, 16)
        decode_batch_size = batched_decode_tuned_params.decode_batch_size
        logger.info(
            f"Using MLA tuned block sizes for batched decode: {batched_decode_tuned_params} for input shapes: {batched_decode_tuning_key}"
        )

        out, new_cache = mla_ragged_paged_attention(
            q,
            q_rope,
            k,
            k_rope,
            cache,
            *args,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            decode_batch_size=decode_batch_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            transpose_kv_cache=envs.MLA_TRANSPOSE_KV_CACHE)

        return new_cache, out

    kv_cache, output_TNA = jax.jit(
        jax.shard_map(_mla_ragged_paged_attention,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))(q_NTA, q_rope_TNH, k_SA, k_rope_SH,
                                        kv_cache, md.seq_lens, md.block_tables,
                                        md.query_start_loc,
                                        md.request_distribution)
    return kv_cache, output_TNA


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "sm_scale",
        "sliding_window",
    ),
)
def encoder_only_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
) -> jax.Array:
    kernel = sharded_encoder_only_attention(
        mesh=mesh,
        causal=False,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
    )
    return kernel(q, k, v, attention_metadata.seq_lens)
