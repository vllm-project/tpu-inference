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

"""Context-parallel attention: DCP (Distributed CP) and PCP (Prefill CP)."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import tpu_inference.kernels.experimental.rpa_v3_cp.kernel as rpa_v3_cp
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, PCPMetadata)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


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
        update_kv_cache: bool = True,
        return_lse: bool = False,
        skip_cache_attn: bool = False,
        skip_current_attn: bool = False,
        is_context_phase: bool = False,
        use_causal_mask: bool = True):
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
        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
        o_spec = P(('data', 'attn_dp', 'dcp'), ShardingAxisName.KV_HEAD, None)
    else:
        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD,
                   None)
        o_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD,
                   None)
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)
    print(f"page_size={kv_cache.shape[1]}")

    dcp_size = mesh.shape['dcp']
    cp_rank_global = jnp.arange(dcp_size, dtype=jnp.int32)

    in_specs = [
        q_spec,
        kv_spec,
        kv_spec,
        kv_cache_spec,
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.KV_CONTEXT),
    ]

    args = [
        q, k, v, kv_cache, kv_lens, paged_indices, cu_q_lens, distribution,
        cp_rank_global
    ]

    lse_spec = (P(
        ('data', 'attn_dp',
         'dcp'), ShardingAxisName.KV_HEAD) if is_context_phase else P(
             ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD))
    out_specs = [o_spec, kv_cache_spec]
    if return_lse:
        out_specs.append(lse_spec)

    def _ragged_paged_attention_wrapper(*args):
        *kernel_args, cp_rank = args
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
    attn_out: jax.Array,
    lse: jax.Array,
    mesh: Mesh,
    dcp_axis: str = 'dcp',
    model_axis: str = 'model',
) -> tuple[jax.Array, jax.Array]:

    def _inner(attn_out, lse):
        dcp_size = jax.lax.psum(1, axis_name=dcp_axis)
        max_num_tokens = attn_out.shape[0]
        local_heads = attn_out.shape[1]
        head_dim = attn_out.shape[2]

        attn_gathered = jax.lax.all_to_all(
            attn_out,
            axis_name=dcp_axis,
            split_axis=1,
            concat_axis=0,
            tiled=True,
        )

        lse_gathered = jax.lax.all_to_all(
            lse,
            axis_name=dcp_axis,
            split_axis=1,
            concat_axis=0,
            tiled=True,
        )

        new_local_heads = local_heads // dcp_size
        attn_chunks = attn_gathered.reshape(dcp_size, max_num_tokens,
                                            new_local_heads, head_dim)
        lse_chunks = lse_gathered.reshape(dcp_size, max_num_tokens,
                                          new_local_heads)

        combined_lse = jax.nn.logsumexp(lse_chunks, axis=0)

        weights = jnp.exp(lse_chunks - combined_lse[None])
        # When all ranks have -inf LSE (prefill seqs with no cached tokens),
        # combined_lse=-inf and weights=NaN. Zero them out so combined_out
        # stays 0 and merge_attn_states falls back to the query-phase result.
        weights = jnp.where(jnp.isneginf(combined_lse[None, ...]), 0.0,
                            weights)

        combined_out = jnp.einsum('d t h, d t h f -> t h f', weights,
                                  attn_chunks)

        return combined_out, combined_lse

    return jax.shard_map(
        _inner,
        mesh=mesh,
        in_specs=(
            P(dcp_axis, ShardingAxisName.KV_HEAD, None),
            P(dcp_axis, ShardingAxisName.KV_HEAD),
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
    """Merge attention outputs from disjoint KV spans using LSE weighting.

    Both guards (max_lse_safe, denom) are required for the case where both
    context_lse and query_lse are -inf (padding tokens with no attended keys).
    """
    max_lse = jnp.maximum(context_lse, query_lse)
    max_lse_safe = jnp.where(jnp.isinf(max_lse), 0.0, max_lse)
    exp_context = jnp.exp(context_lse - max_lse_safe)
    exp_query = jnp.exp(query_lse - max_lse_safe)

    sum_exp = exp_context + exp_query
    denom = jnp.where(sum_exp == 0.0, 1.0, sum_exp)

    merged_out = (context_out * exp_context[..., None] +
                  query_out * exp_query[..., None]) / denom[..., None]
    merged_lse = jnp.where(sum_exp == 0.0, -jnp.inf,
                           max_lse_safe + jnp.log(denom))

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
    """DCP attention: context phase → alltoall → query phase → merge."""
    if head_dim_original is None:
        head_dim_original = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata

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

    final_output, _ = merge_attn_states(
        context_attn_out_cor,
        context_lse_cor,
        query_attn_out,
        query_lse,
    )

    return updated_kv_cache, final_output


def _lse_reduce_scatter(o: jax.Array, lse: jax.Array, axis: str,
                        axis_size: int):
    """
    o:   [axis_size*blk, nq, hd]  ->  [blk, nq, hd]
    lse: [axis_size*blk, nq]      ->  [blk, nq]
    """
    n = o.shape[0] // axis_size
    m = lax.pmax(lse, axis)
    m_safe = jnp.where(jnp.isinf(m), 0.0, m)
    w = jnp.exp(lse - m_safe)
    w_o = w[..., None].astype(o.dtype)
    o_w_sum = lax.psum_scatter(o * w_o, axis, scatter_dimension=0, tiled=True)
    denom = lax.psum_scatter(w, axis, scatter_dimension=0, tiled=True)
    m_own = lax.dynamic_slice_in_dim(m_safe, lax.axis_index(axis) * n, n, 0)
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[..., None]
    o_merged = o_w_sum.astype(denom.dtype) / denom_safe
    lse_merged = jnp.where(denom == 0.0, -jnp.inf, m_own + jnp.log(denom))
    return o_merged, lse_merged


def pcp_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    distribution: jax.Array,
    pcp: PCPMetadata,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    use_causal_mask: bool = True,
):
    """Single-request prefill context-parallel (PCP) attention.

    The one request's S = 2*pcp*C current tokens are split head-tail into
    2*pcp chunks of size C; rank r holds chunk r (head) and chunk 2*pcp-1-r
    (tail), so its local buffer is [head_chunk | tail_chunk].

    Two kernel launches per rank, LSE-combined:
      * cache phase:  all-gather Q, attend the pcp-strided cache (non-causal,
        no write), LSE reduce-scatter over pcp ranks.
      * current phase: local Q attends all-gathered current KV (causal,
        per-chunk q_pos_offset); tail chunk writes the strided cache.
    """
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT
    pcp_size = get_mesh_shape_product(mesh, pcp_axis)
    two_p = 2 * pcp_size
    padded_q_len = q.shape[0]
    C = padded_q_len // two_p

    _row = [c for r in range(pcp_size) for c in (r, two_p - 1 - r)]
    _inv = [0] * two_p
    for _i, _c in enumerate(_row):
        _inv[_c] = _i
    inv_row = jnp.array(_inv, jnp.int32)

    q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)
    pcp_spec = P(pcp_axis, None)
    repl = P()

    def _rank_i32(r):
        return jnp.reshape(r, (1, )).astype(jnp.int32)

    def _fn(q_l, k_l, v_l, kvc, kvl, kvcl, pi, dist2, pcp_qsl, pcp_qpos):
        r = lax.axis_index(pcp_axis)

        def ag(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        def to_token_order(x):
            g = ag(x).reshape(two_p, C, *x.shape[1:])
            return g[inv_row].reshape(padded_q_len, *x.shape[1:])

        common = dict(cp_rank=_rank_i32(r),
                      cp_group_size=pcp_size,
                      return_lse=True,
                      sm_scale=sm_scale,
                      q_scale=q_scale,
                      k_scale=k_scale,
                      v_scale=v_scale)

        # cache phase
        ag_q = ag(q_l)
        cu_cache = jnp.zeros_like(pcp_qsl[0]).at[1:].set(padded_q_len)
        dist_cache = jnp.array([0, 0, 1], jnp.int32)
        o1, kvc1, l1 = rpa_v3_cp.ragged_paged_attention(
            ag_q,
            k_l,
            v_l,
            kvc,
            kvl,
            pi,
            cu_cache,
            dist_cache,
            kv_cache_lens=kvcl,
            skip_current_attn=True,
            use_causal_mask=False,
            update_kv_cache=False,
            **common)
        o1, l1 = _lse_reduce_scatter(o1, l1, pcp_axis, pcp_size)

        # current phase
        page_size_local = kvc.shape[1]
        remap_kv = (C >= page_size_local) and (C % page_size_local == 0)
        if remap_kv:
            k_cur, v_cur = ag(k_l), ag(v_l)
        else:
            k_cur, v_cur = to_token_order(k_l), to_token_order(v_l)
        o2, kvc2, l2 = rpa_v3_cp.ragged_paged_attention(
            q_l,
            k_cur,
            v_cur,
            kvc1,
            kvl,
            pi,
            pcp_qsl[0],
            dist2,
            kv_cache_lens=kvcl,
            q_pos_offsets=pcp_qpos[0],
            pcp_chunk_size=(C if remap_kv else None),
            skip_cache_attn=True,
            use_causal_mask=use_causal_mask,
            update_kv_cache=update_kv_cache,
            write_last_seq_only=True,
            **common)

        out, _ = merge_attn_states(o1, l1, o2, l2)
        return out.astype(q.dtype), kvc2

    return jax.shard_map(
        _fn,
        mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec, kv_cache_spec, repl, repl, repl,
                  repl, pcp_spec, pcp_spec),
        out_specs=(q_spec, kv_cache_spec),
        check_vma=False,
    )(q, k, v, kv_cache, kv_lens, pcp.kv_cache_lens, page_indices,
      distribution, pcp.query_start_loc, pcp.q_pos_offsets)
