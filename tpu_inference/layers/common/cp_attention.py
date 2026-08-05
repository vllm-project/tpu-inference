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
"""Context-parallel (CP) attention: DCP (Decode Context Parallelism) and PCP (Prefill Context Parallelism)"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import tpu_inference.kernels.experimental.rpa_v3_cp.kernel as rpa_v3_cp
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product

# ── Shared utilities ──────────────────────────────────────────────────────────


def merge_attn_states(
    cache_out: jax.Array,
    cache_lse: jax.Array,
    query_out: jax.Array,
    query_lse: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """LSE-weighted merge of two disjoint attention spans.

    Both guards are required when both LSEs are -inf (padding tokens):
      max_lse_safe: prevents (-inf) - (-inf) = NaN in exp
      denom:        prevents 0 / 0 = NaN in weighted sum
    """
    max_lse = jnp.maximum(cache_lse, query_lse)
    max_lse_safe = jnp.where(jnp.isinf(max_lse), 0.0, max_lse)
    exp_cache = jnp.exp(cache_lse - max_lse_safe)
    exp_query = jnp.exp(query_lse - max_lse_safe)
    sum_exp = exp_cache + exp_query
    denom = jnp.where(sum_exp == 0.0, 1.0, sum_exp)
    merged_out = (cache_out * exp_cache[..., None] +
                  query_out * exp_query[..., None]) / denom[..., None]
    merged_lse = jnp.where(sum_exp == 0.0, -jnp.inf,
                           max_lse_safe + jnp.log(denom))
    return merged_out, merged_lse


def _rpa_cp_call(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    cp_rank: jax.Array,
    cp_group_size: int,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    **flags,
):
    """Call rpa_v3_cp with shared CP params; always returns LSE."""
    return rpa_v3_cp.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        cp_rank=cp_rank,
        cp_group_size=cp_group_size,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        return_lse=True,
        **flags,
    )


def _dcp_a2a_reduce(
    o: jax.Array,
    lse: jax.Array,
    axis: str,
    axis_size: int,
) -> tuple[jax.Array, jax.Array]:
    """All-to-all across DCP: exchange head shards for token shards, merge LSE.

    Called inside a DCP shard_map body after the cache phase.

    Input  (per rank, before exchange):
      o:   [local_tokens, heads_full, head_dim]   heads_full = H / model
      lse: [local_tokens, heads_full]
    Output (per rank, after exchange):
      o:   [local_tokens, heads_local, head_dim]  heads_local = H / (model * dcp)
      lse: [local_tokens, heads_local]
    """
    local_tokens = o.shape[0]
    local_heads = o.shape[1]
    head_dim = o.shape[2]

    o_gathered = lax.all_to_all(o,
                                axis,
                                split_axis=1,
                                concat_axis=0,
                                tiled=True)
    lse_gathered = lax.all_to_all(lse,
                                  axis,
                                  split_axis=1,
                                  concat_axis=0,
                                  tiled=True)
    # shapes: (local_tokens * axis_size, local_heads // axis_size, ...)

    heads_per_rank = local_heads // axis_size
    o_chunks = o_gathered.reshape(axis_size, local_tokens, heads_per_rank,
                                  head_dim)
    lse_chunks = lse_gathered.reshape(axis_size, local_tokens, heads_per_rank)

    out_lse = jax.nn.logsumexp(lse_chunks, axis=0)
    weights = jnp.exp(lse_chunks - out_lse[None])
    # Guard: when all ranks return -inf LSE (no cached tokens for these prefill
    # seqs), weights become NaN. Zero them; merge_attn_states falls back to the
    # query-phase result.
    weights = jnp.where(jnp.isneginf(out_lse[None]), 0.0, weights)
    out_merge = jnp.einsum('d t h, d t h f -> t h f', weights, o_chunks)
    return out_merge, out_lse


def dcp_forward(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    md: AttentionMetadata,
    head_dim_original: int | None = None,
    sm_scale: float | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """DCP attention forward — single shard_map over the 'dcp' axis.

    Inside the shard_map body:
      1. all_gather Q heads    cache phase needs full head view per rank
      2. cache phase           attend full Q against this rank's KV cache
      3. _dcp_a2a_reduce       all_to_all: head slices become token slices
      4. current phase         attend local Q against this rank's new tokens
      5. merge_attn_states     lse-weighted combine
    """
    if head_dim_original is None:
        head_dim_original = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    dcp_axis = 'dcp'
    dcp_size = mesh.shape[dcp_axis]

    # GQA/MQA: replicate KV heads to match ATTN_HEAD sharding before shard_map.
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    if tp_size > 1:
        num_kv_heads = k.shape[1]
        if num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}"
                )
            factor = tp_size // num_kv_heads
            k = jnp.repeat(k, factor, axis=1)
            v = jnp.repeat(v, factor, axis=1)

    cp_rank_global = jnp.arange(dcp_size, dtype=jnp.int32)

    q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)
    print(f"page_size={kv_cache.shape[1]}")

    common = dict(sm_scale=sm_scale,
                  q_scale=q_scale,
                  k_scale=k_scale,
                  v_scale=v_scale,
                  sliding_window=attention_chunk_size)

    def _shard_fn(q_local, k_local, v_local, kv_cache_local, kv_lens_local,
                  page_indices_local, cu_q_lens_local, distribution_local,
                  cp_rank):
        # Context phase: all_gather Q heads so every rank attends with dcp times of heads.
        # ATTN_HEAD includes 'dcp', so q_local has heads / (model * dcp).
        # After all_gather along heads axis: heads / model  (= KV_HEAD sharding).
        q_all_heads = lax.all_gather(q_local, dcp_axis, axis=1, tiled=True)

        context_out, kv_cache_temp, context_lse = _rpa_cp_call(
            q_all_heads,
            k_local,
            v_local,
            kv_cache_local,
            kv_lens_local,
            page_indices_local,
            cu_q_lens_local,
            distribution_local,
            cp_rank=cp_rank,
            cp_group_size=dcp_size,
            skip_current_attn=True,
            use_causal_mask=False,
            update_kv_cache=False,
            **common)

        # Rank reduce: swap head shards for token shards, merge partial LSE.
        context_out, context_lse = _dcp_a2a_reduce(context_out, context_lse,
                                                   dcp_axis, dcp_size)

        # Current phase: local Q (head-sharded by ATTN_HEAD) attends new tokens.
        curr_out, kv_cache_updated, curr_lse = _rpa_cp_call(
            q_local,
            k_local,
            v_local,
            kv_cache_temp,
            kv_lens_local,
            page_indices_local,
            cu_q_lens_local,
            distribution_local,
            cp_rank=cp_rank,
            cp_group_size=dcp_size,
            skip_cache_attn=True,
            update_kv_cache=True,
            **common)

        out, _ = merge_attn_states(context_out, context_lse, curr_out,
                                   curr_lse)
        return kv_cache_updated, out.astype(q.dtype)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(ShardingAxisName.ATTN_DATA),  # kv_lens
            P(ShardingAxisName.ATTN_DATA),  # page_indices
            P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
            P(ShardingAxisName.ATTN_DATA),  # distribution
            P(ShardingAxisName.KV_CONTEXT),  # cp_rank_global
        ),
        out_specs=(kv_cache_spec, q_spec),
        check_vma=False,
    )(q, k, v, kv_cache, md.seq_lens, md.block_tables, md.query_start_loc,
      md.request_distribution, cp_rank_global)


def _pcp_rs_reduce(
    o: jax.Array,
    lse: jax.Array,
    axis: str,
    axis_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Reduce-scatter across PCP: each rank collects its own token chunk.

    Called inside a PCP shard_map body after the cache phase.

    Input  (per rank, all-gathered result from cache phase):
      o:   [axis_size * chunk, heads, head_dim]
      lse: [axis_size * chunk, heads]
    Output (per rank, own chunk only):
      o:   [chunk, heads, head_dim]
      lse: [chunk, heads]
    """
    chunk = o.shape[0] // axis_size
    max_lse = lax.pmax(lse, axis)
    max_lse_safe = jnp.where(jnp.isinf(max_lse), 0.0, max_lse)
    weights = jnp.exp(lse - max_lse_safe)
    o_weighted_sum = lax.psum_scatter(o * weights[..., None].astype(o.dtype),
                                      axis,
                                      scatter_dimension=0,
                                      tiled=True)
    denom = lax.psum_scatter(weights, axis, scatter_dimension=0, tiled=True)
    max_lse_own = lax.dynamic_slice_in_dim(max_lse_safe,
                                           lax.axis_index(axis) * chunk, chunk,
                                           0)
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[..., None]
    out_merged = o_weighted_sum.astype(denom.dtype) / denom_safe
    lse_merged = jnp.where(denom == 0.0, -jnp.inf,
                           max_lse_own + jnp.log(denom))
    return out_merged, lse_merged


def pcp_forward(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    md: AttentionMetadata,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """PCP attention forward.

    Inside the shard_map body:
      1. all_gather Q tokens   cache phase needs full sequence view per rank
      2. cache phase           attend full Q against this rank's KV cache shard
      3. _pcp_rs_reduce        reduce_scatter: each rank collects its token chunk
      4. current phase         local Q (head+tail) attends all-gathered current KV
      5. merge_attn_states     lse-weighted combine
    """
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT
    pcp_size = get_mesh_shape_product(mesh, pcp_axis)
    two_p = 2 * pcp_size
    padded_q_len = q.shape[0]
    C = padded_q_len // two_p

    # Precompute inv_row on host: maps rank-order chunk index → token order.
    _row = [c for r in range(pcp_size) for c in (r, two_p - 1 - r)]
    _inv = [0] * two_p
    for _i, _c in enumerate(_row):
        _inv[_c] = _i
    inv_row = jnp.array(_inv, jnp.int32)

    q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)

    common = dict(sm_scale=sm_scale,
                  q_scale=q_scale,
                  k_scale=k_scale,
                  v_scale=v_scale)

    # Cache-phase strategy, decided per compile from static comm estimates.
    # all_gather and reduce_scatter are duals and move the same (p-1)/p
    # fraction, so gather-Q counts both legs; gather-KV moves K and V but has
    # no output collective.  Volume alone puts the crossover at
    # ctx = chunk*NQ/NKV, but gather-Q issues TWO collective rounds (two sync
    # points) plus an LSE reweight, so empirically it needs ~2x the raw volume
    # advantage before it actually wins.
    cache_pages = md.pcp.cache_pages
    _GATHER_Q_OVERHEAD = 2.0
    comm_q = 2 * padded_q_len * q.shape[1] * q.shape[2]
    comm_kv = 2 * (cache_pages * kv_cache.shape[1]) * k.shape[1] * q.shape[2]
    use_gather_kv = comm_kv < _GATHER_Q_OVERHEAD * comm_q

    def _shard_fn(q_local, k_local, v_local, kv_cache_local, kv_lens_local,
                  kv_cache_lens_local, page_indices_local, distribution_local,
                  pcp_cu_q_lens_local, pcp_q_pos_offsets_local):
        axis_idx = lax.axis_index(pcp_axis)
        cp_rank = jnp.reshape(axis_idx, (1, )).astype(jnp.int32)

        def all_gather_tokens(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        def to_token_order(x):  # rank-order chunks -> global token order
            return all_gather_tokens(x).reshape(two_p, C,
                                                *x.shape[1:])[inv_row].reshape(
                                                    padded_q_len, *x.shape[1:])

        # ---- Cache phase --------------------------------------------------
        # Two ways to give every rank what it needs:
        #   gather-Q : all_gather Q, attend this rank's KV shard, then
        #              reduce_scatter the partials.  Comm ~ 2*chunk*NQ, i.e.
        #              INDEPENDENT of context length.
        #   gather-KV: all_gather the striped KV cache into global token order
        #              so every rank holds the full cache, then attend the
        #              LOCAL Q against it as ordinary paged attention.  No
        #              output collective at all.  Comm ~ ctx*NKV.
        # gather-KV wins at short/medium context, gather-Q once ctx is long.
        # The cache phase never writes the cache (update_kv_cache=False), so
        # the current phase starts from the untouched local shard unless the
        # gather-Q path threaded a copy through.
        kv_cache_temp = kv_cache_local
        if cache_pages == 0:
            # Nothing cached (first chunk of a chunked prefill): the cache
            # phase would attend an empty cache, be fully masked, and have its
            # -inf result discarded by merge_attn_states.  Skip it outright.
            context_out = context_lse = None
        elif use_gather_kv:
            # Compact to this request's live pages, then all_gather with the
            # pcp axis INNERMOST: for local page p, offset o, rank j the cache
            # holds global token (p*page_l + o)*pcp + j, so flattening
            # (p, o, j) is already global token order.  Regroup into pages of
            # the ORIGINAL width (a pure reshape) and the cache phase becomes
            # plain paged attention with cp_group_size=1.
            local_q = q_local.shape[0]
            cu_kv = jnp.zeros_like(pcp_cu_q_lens_local[0]).at[1:].set(local_q)
            max_seqs = kv_lens_local.shape[0]
            kv_src = jnp.take(kv_cache_local,
                              page_indices_local[:cache_pages],
                              axis=0)
            kv_tok = lax.all_gather(kv_src, pcp_axis, axis=2, tiled=False)
            n_pages_tok = kv_src.shape[0] * pcp_size
            kv_tok = kv_tok.reshape(n_pages_tok, kv_src.shape[1],
                                    *kv_src.shape[2:])
            pi_tok = jnp.tile(
                jnp.arange(n_pages_tok, dtype=page_indices_local.dtype),
                max_seqs)
            context_out, _, context_lse = _rpa_cp_call(
                q_local,
                k_local,
                v_local,
                kv_tok,
                kv_lens_local,
                pi_tok,
                cu_kv,
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=jnp.zeros((1, ), jnp.int32),
                cp_group_size=1,
                kv_cache_lens=kv_cache_lens_local,
                skip_current_attn=True,
                use_causal_mask=False,
                update_kv_cache=False,
                **common)
        else:
            # Cache phase: all_gather Q tokens so every rank sees the full
            # sequence.  PCP local q_local has 2*C tokens (head + tail chunk
            # for this rank).  After all_gather along the tokens axis:
            # pcp_size * 2 * C = padded_q_len.
            q_all_tokens = all_gather_tokens(q_local)
            cu_cache = jnp.zeros_like(
                pcp_cu_q_lens_local[0]).at[1:].set(padded_q_len)
            context_out, kv_cache_temp, context_lse = _rpa_cp_call(
                q_all_tokens,
                k_local,
                v_local,
                kv_cache_local,
                kv_lens_local,
                page_indices_local,
                cu_cache,
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=cp_rank,
                cp_group_size=pcp_size,
                kv_cache_lens=kv_cache_lens_local,
                skip_current_attn=True,
                use_causal_mask=False,
                update_kv_cache=False,
                **common)

            # Rank reduce: reduce_scatter so each rank gets its own 2*C chunk.
            context_out, context_lse = _pcp_rs_reduce(context_out, context_lse,
                                                      pcp_axis, pcp_size)

        # Current phase: local Q (head+tail chunks) attends all-gathered current KV.
        # pcp_cu_q_lens_local[0] = [0, chunk, chunk+tail_real]; pcp_q_pos_offsets_local[0] = [head_offset, tail_offset].
        # remap_kv: if C aligns with page_size, all_gather_tokens() avoids an extra gather-reorder.
        page_size = kv_cache_local.shape[1]
        remap_kv = (C >= page_size) and (C % page_size == 0)
        k_curr = all_gather_tokens(k_local) if remap_kv else to_token_order(
            k_local)
        v_curr = all_gather_tokens(v_local) if remap_kv else to_token_order(
            v_local)
        curr_out, kv_cache_updated, curr_lse = _rpa_cp_call(
            q_local,
            k_curr,
            v_curr,
            kv_cache_temp,
            kv_lens_local,
            page_indices_local,
            pcp_cu_q_lens_local[0],
            distribution_local,
            cp_rank=cp_rank,
            cp_group_size=pcp_size,
            kv_cache_lens=kv_cache_lens_local,
            q_pos_offsets=pcp_q_pos_offsets_local[0],
            pcp_chunk_size=(C if remap_kv else None),
            skip_cache_attn=True,
            use_causal_mask=use_causal_mask,
            update_kv_cache=update_kv_cache,
            write_last_seq_only=True,
            **common)

        # With nothing cached the current phase already IS the answer.
        if context_out is None:
            out = curr_out
        else:
            out, _ = merge_attn_states(context_out, context_lse, curr_out,
                                       curr_lse)
        return kv_cache_updated, out.astype(q.dtype)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(),  # kv_lens: replicated
            P(),  # pcp.kv_cache_lens: replicated
            P(),  # page_indices: replicated
            P(),  # distribution: replicated
            P(pcp_axis, None),  # pcp.query_start_loc: per-rank cu_q_lens
            P(pcp_axis, None),  # pcp.q_pos_offsets: per-rank position offsets
        ),
        out_specs=(kv_cache_spec, q_spec),
        check_vma=False,
    )(q, k, v, kv_cache, md.seq_lens, md.pcp.kv_cache_lens, md.block_tables,
      md.request_distribution, md.pcp.query_start_loc, md.pcp.q_pos_offsets)
