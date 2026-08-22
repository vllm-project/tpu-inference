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
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

# ── Shared utilities ──────────────────────────────────────────────────────────

logger = init_logger(__name__)


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


def _pcp_write_new_kv(kv_cache, new_kv_all, kv_len, cache_len, page_indices,
                      max_num_seqs, rank):
    """Write this rank's pages of the current chunk's KV into the
    page-interleaved cache (global page p lives on rank p % P at local page
    p // P), in place: one gather + one scatter over the owned pages.

    new_kv_all: [P, 2C, ...] in kv-cache row layout, rank order (row r =
    rank r's [head chunk r | tail chunk 2P-1-r]); the real tokens are global
    positions [cache_len, kv_len). The chunk touches a statically bounded
    run of global pages; this rank owns every P-th of them. Each owned page
    is read, patched where the rows are real, and scattered back (pages
    past the real tokens get out-of-bounds indices and are dropped).
    """
    pcp, two_c = new_kv_all.shape[:2]
    C = two_c // 2
    page = kv_cache.shape[1]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    # Rank order -> token order: chunk c < P is rank c's head (row 2c),
    # chunk c >= P is rank 2P-1-c's tail (row 2(2P-1-c)+1).
    rows = new_kv_all.reshape(2 * pcp, C, *new_kv_all.shape[2:])
    order = [
        2 * c if c < pcp else 2 * (2 * pcp - 1 - c) + 1 for c in range(2 * pcp)
    ]
    tok = rows[jnp.array(order, jnp.int32)].reshape(2 * pcp * C,
                                                    *new_kv_all.shape[2:])
    zpad = jnp.zeros((page, *tok.shape[1:]), tok.dtype)
    tok_pad = jnp.concatenate([zpad, tok, zpad])
    num_current = kv_len - cache_len
    first_gp = cache_len // page
    # Owned global pages gp = first_owned + P*j, first_owned = the first
    # page >= first_gp with gp % P == rank; m bounds how many the chunk can
    # touch.
    m = cdiv(cdiv(2 * pcp * C, page) + 1, pcp) + 1
    first_owned = first_gp + (rank - first_gp) % pcp
    gps = first_owned + pcp * jnp.arange(m, dtype=jnp.int32)  # [m]
    valid = gps * page < cache_len + num_current
    lp = jnp.minimum(gps // pcp, pages_per_seq - 1)
    # Invalid pages: unique out-of-bounds indices (dropped by the scatter,
    # zero-filled by the gather).
    phys = jnp.where(valid, page_indices[lp],
                     kv_cache.shape[0] + jnp.arange(m, dtype=jnp.int32))
    starts = jnp.clip(gps * page - cache_len + page, 0,
                      tok_pad.shape[0] - page)
    new_pages = tok_pad[starts[:, None] +
                        jnp.arange(page)[None, :]]  # [m, page, ...]
    cur = kv_cache.at[phys].get(mode="fill", fill_value=0)  # [m, page, ...]
    pos = gps[:, None] * page + jnp.arange(page)[None, :]
    keep = (pos >= cache_len) & (pos < cache_len + num_current)
    val = jnp.where(keep.reshape((m, page) + (1, ) * (cur.ndim - 2)),
                    new_pages, cur)
    return kv_cache.at[phys].set(val, mode="drop", unique_indices=True)


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
    """
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT
    pcp_size = get_mesh_shape_product(mesh, pcp_axis)
    two_p = 2 * pcp_size
    padded_q_len = q.shape[0]
    C = padded_q_len // two_p

    q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)

    common = dict(sm_scale=sm_scale,
                  q_scale=q_scale,
                  k_scale=k_scale,
                  v_scale=v_scale)

    def _shard_fn(q_local, k_local, v_local, kv_cache_local, kv_lens_local,
                  kv_cache_lens_local, page_indices_local, distribution_local,
                  pcp_cu_q_lens_local, pcp_q_pos_offsets_local):
        axis_idx = lax.axis_index(pcp_axis)
        cp_rank = jnp.reshape(axis_idx, (1, )).astype(jnp.int32)

        # ONE seq: this rank's [head chunk | tail chunk] (2C rows, tail pad
        # rows discarded by the caller); pcp_q_pos_offsets_local[0] =
        # [head_offset, tail_offset] places both halves.
        cu = jnp.zeros_like(pcp_cu_q_lens_local[0]).at[1:].set(2 * C)
        # k/v stay LOCAL (this rank's head+tail chunks, same layout as q):
        # the kernel rotates [cache shard | own new KV] around the ring.
        # The kernel only reads the cache; it returns every rank's own new KV
        # (new_kv_all [P, 2C, ...], rank order) collected as it rotated by,
        # and this rank writes the pages it owns below with plain in-place
        # updates. (A communicating kernel that mutates the cache in place
        # makes XLA copy the whole cache in and out, per layer.)
        out, new_kv_all, _ = _rpa_cp_call(
            q_local,
            k_local,
            v_local,
            kv_cache_local,
            kv_lens_local,
            page_indices_local,
            cu,
            jnp.array([0, 0, 1], jnp.int32),
            cp_rank=cp_rank,
            cp_group_size=pcp_size,
            kv_cache_lens=kv_cache_lens_local,
            q_pos_offsets=pcp_q_pos_offsets_local[0],
            pcp_ring_axis_name=pcp_axis,
            pcp_ring_mesh_axis_names=tuple(mesh.axis_names),
            use_causal_mask=use_causal_mask,
            update_kv_cache=update_kv_cache,
            **common)
        if update_kv_cache:
            kv_cache_local = _pcp_write_new_kv(kv_cache_local, new_kv_all,
                                               kv_lens_local[0],
                                               kv_cache_lens_local[0],
                                               page_indices_local,
                                               kv_lens_local.shape[0],
                                               axis_idx)
        return kv_cache_local, out.astype(q.dtype)

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
