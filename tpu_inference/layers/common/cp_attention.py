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

"""Context-parallel (CP) attention: DCP and PCP.

Both modes share the same three-phase structure inside a single shard_map:

  Phase 1 – cache:       attend Q against this rank's KV cache shard.
  Rank reduce:           combine partial outputs across CP ranks.
  Phase 2 – current:     attend Q against this rank's new tokens.
  Merge:                 LSE-weighted combine of cache + current outputs.

Structural symmetry between DCP and PCP
────────────────────────────────────────
                     DCP                         PCP
  CP axis            'dcp'                        'pcp'
  What is split      requests + KV pages          one request (head-tail chunks)
  Cache-phase Q      all_gather(q, axis=heads)    all_gather(q, axis=tokens)
  Rank reduce        all_to_all (heads ↔ tokens)  reduce_scatter (tokens → chunk)
  Current-phase KV   local (this rank's tokens)   all_gather (all ranks' tokens)
"""

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


def _rpa_cp_call(q, k, v, kvc, kvl, pi, cu_q_lens, dist, *, cp_rank,
                 cp_group_size, sm_scale, q_scale, k_scale, v_scale, **flags):
    """Call rpa_v3_cp with shared CP params; always returns LSE."""
    return rpa_v3_cp.ragged_paged_attention(
        q, k, v, kvc, kvl, pi, cu_q_lens, dist,
        cp_rank=cp_rank,
        cp_group_size=cp_group_size,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        return_lse=True,
        **flags,
    )


# ── DCP: Distributed Context Parallelism ─────────────────────────────────────
#
# Each rank owns its own requests with contiguous KV pages.  Q is sharded by
# ATTN_HEAD which includes 'dcp', so each rank starts with a head slice.
#
# Cache phase needs all Q heads to attend against each rank's KV shard, so we
# all_gather Q heads across 'dcp' first.  After attending, all_to_all swaps
# axes (heads → tokens, tokens → heads) and merges partial LSE results.


def _dcp_rank_reduce(
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

    o_gathered = lax.all_to_all(o, axis, split_axis=1, concat_axis=0, tiled=True)
    lse_gathered = lax.all_to_all(lse, axis, split_axis=1, concat_axis=0,
                                  tiled=True)
    # shapes: (local_tokens * axis_size, local_heads // axis_size, ...)

    heads_per_rank = local_heads // axis_size
    o_chunks = o_gathered.reshape(axis_size, local_tokens, heads_per_rank,
                                  head_dim)
    lse_chunks = lse_gathered.reshape(axis_size, local_tokens, heads_per_rank)

    combined_lse = jax.nn.logsumexp(lse_chunks, axis=0)
    weights = jnp.exp(lse_chunks - combined_lse[None])
    # Guard: when all ranks return -inf LSE (no cached tokens for these prefill
    # seqs), weights become NaN. Zero them; merge_attn_states falls back to the
    # query-phase result.
    weights = jnp.where(jnp.isneginf(combined_lse[None]), 0.0, weights)
    combined_out = jnp.einsum('d t h, d t h f -> t h f', weights, o_chunks)
    return combined_out, combined_lse


def dcp_forward(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    md: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,
    sm_scale: float | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """DCP attention forward — single shard_map over the 'dcp' axis.

    Inside the shard_map body:
      1. all_gather Q heads    → cache phase needs full head view per rank
      2. cache phase           → attend full Q against this rank's KV cache
      3. _dcp_rank_reduce      → all_to_all: head slices become token slices
      4. current phase         → attend local Q against this rank's new tokens
      5. merge_attn_states     → LSE-weighted combine
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

    def _fn(q_l, k_l, v_l, kvc, kvl, pi, cu_q_lens, dist, cp_rank):
        # Cache phase: all_gather Q heads so every rank attends with full heads.
        # ATTN_HEAD includes 'dcp', so q_l has heads / (model * dcp).
        # After all_gather along heads axis: heads / model  (= KV_HEAD sharding).
        q_full = lax.all_gather(q_l, dcp_axis, axis=1, tiled=True)

        o1, kvc1, l1 = _rpa_cp_call(
            q_full, k_l, v_l, kvc, kvl, pi, cu_q_lens, dist,
            cp_rank=cp_rank,
            cp_group_size=dcp_size,
            skip_current_attn=True,
            use_causal_mask=False,
            update_kv_cache=False,
            **common)

        # Rank reduce: swap head shards for token shards, merge partial LSE.
        o1, l1 = _dcp_rank_reduce(o1, l1, dcp_axis, dcp_size)

        # Current phase: local Q (head-sharded by ATTN_HEAD) attends new tokens.
        o2, kvc2, l2 = _rpa_cp_call(
            q_l, k_l, v_l, kvc1, kvl, pi, cu_q_lens, dist,
            cp_rank=cp_rank,
            cp_group_size=dcp_size,
            skip_cache_attn=True,
            update_kv_cache=True,
            **common)

        out, _ = merge_attn_states(o1, l1, o2, l2)
        return out.astype(q.dtype), kvc2

    return jax.shard_map(
        _fn,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(ShardingAxisName.ATTN_DATA),   # kv_lens
            P(ShardingAxisName.ATTN_DATA),   # page_indices
            P(ShardingAxisName.ATTN_DATA),   # cu_q_lens
            P(ShardingAxisName.ATTN_DATA),   # distribution
            P(ShardingAxisName.KV_CONTEXT),  # cp_rank_global
        ),
        out_specs=(q_spec, kv_cache_spec),
        check_vma=False,
    )(q, k, v, kv_cache, md.seq_lens, md.block_tables, md.query_start_loc,
      md.request_distribution, cp_rank_global)


# ── PCP: Prefill Context Parallelism ─────────────────────────────────────────
#
# One long prefill request split head-tail across ranks.  Rank r owns token
# chunks r (head) and 2P-1-r (tail), local buffer = [head | tail].
#
# Cache phase needs all Q tokens (full sequence) to attend against each rank's
# strided KV cache shard, so we all_gather Q tokens across 'pcp' first.
# After attending, reduce_scatter gives each rank its own token chunk's result.


def _pcp_rank_reduce(
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
    m = lax.pmax(lse, axis)
    m_safe = jnp.where(jnp.isinf(m), 0.0, m)
    w = jnp.exp(lse - m_safe)
    o_w_sum = lax.psum_scatter(o * w[..., None].astype(o.dtype),
                               axis,
                               scatter_dimension=0,
                               tiled=True)
    denom = lax.psum_scatter(w, axis, scatter_dimension=0, tiled=True)
    m_own = lax.dynamic_slice_in_dim(m_safe, lax.axis_index(axis) * chunk,
                                     chunk, 0)
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[..., None]
    o_merged = o_w_sum.astype(denom.dtype) / denom_safe
    lse_merged = jnp.where(denom == 0.0, -jnp.inf, m_own + jnp.log(denom))
    return o_merged, lse_merged


def pcp_forward(
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
) -> tuple[jax.Array, jax.Array]:
    """PCP attention forward — single shard_map over the 'pcp' axis.

    Inside the shard_map body:
      1. all_gather Q tokens   → cache phase needs full sequence view per rank
      2. cache phase           → attend full Q against this rank's KV cache shard
      3. _pcp_rank_reduce      → reduce_scatter: each rank collects its token chunk
      4. current phase         → local Q (head+tail) attends all-gathered current KV
      5. merge_attn_states     → LSE-weighted combine
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

    def _fn(q_l, k_l, v_l, kvc, kvl, kvcl, pi, dist, pcp_qsl, pcp_qpos):
        r = lax.axis_index(pcp_axis)
        cp_rank = jnp.reshape(r, (1,)).astype(jnp.int32)

        def ag(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        def to_token_order(x):
            return ag(x).reshape(two_p, C, *x.shape[1:])[inv_row].reshape(
                padded_q_len, *x.shape[1:])

        # Cache phase: all_gather Q tokens so every rank sees the full sequence.
        # PCP local q_l has 2*C tokens (head + tail chunk for this rank).
        # After all_gather along tokens axis: pcp_size * 2 * C = padded_q_len.
        q_full = ag(q_l)
        cu_cache = jnp.zeros_like(pcp_qsl[0]).at[1:].set(padded_q_len)
        o1, kvc1, l1 = _rpa_cp_call(
            q_full, k_l, v_l, kvc, kvl, pi,
            cu_cache, jnp.array([0, 0, 1], jnp.int32),
            cp_rank=cp_rank,
            cp_group_size=pcp_size,
            kv_cache_lens=kvcl,
            skip_current_attn=True,
            use_causal_mask=False,
            update_kv_cache=False,
            **common)

        # Rank reduce: reduce_scatter so each rank gets its own 2*C token chunk.
        o1, l1 = _pcp_rank_reduce(o1, l1, pcp_axis, pcp_size)

        # Current phase: local Q (head+tail chunks) attends all-gathered current KV.
        # remap_kv: if C aligns with page_size, ag() avoids an extra gather-reorder.
        page_size = kvc.shape[1]
        remap_kv = (C >= page_size) and (C % page_size == 0)
        k_cur = ag(k_l) if remap_kv else to_token_order(k_l)
        v_cur = ag(v_l) if remap_kv else to_token_order(v_l)
        o2, kvc2, l2 = _rpa_cp_call(
            q_l, k_cur, v_cur, kvc1, kvl, pi,
            pcp_qsl[0], dist,
            cp_rank=cp_rank,
            cp_group_size=pcp_size,
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
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(),                # kv_lens: replicated
            P(),                # pcp.kv_cache_lens: replicated
            P(),                # page_indices: replicated
            P(),                # distribution: replicated
            P(pcp_axis, None),  # pcp.query_start_loc: per-rank cu_q_lens
            P(pcp_axis, None),  # pcp.q_pos_offsets: per-rank position offsets
        ),
        out_specs=(q_spec, kv_cache_spec),
        check_vma=False,
    )(q, k, v, kv_cache, kv_lens, pcp.kv_cache_lens, page_indices,
      distribution, pcp.query_start_loc, pcp.q_pos_offsets)
