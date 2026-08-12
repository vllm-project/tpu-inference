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
      1. cache phase           in-kernel ring: KV shards rotate around the pcp
                               axis while each rank attends with its local Q;
                               one online softmax accumulates all rounds
      2. current phase         local Q (head+tail) attends all-gathered current KV
      3. merge_attn_states     lse-weighted combine
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

    cache_pages = md.pcp.cache_pages

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
        if cache_pages == 0:
            # Nothing cached (first chunk of a chunked prefill): the cache
            # phase would attend an empty cache, be fully masked, and have its
            # -inf result discarded by merge_attn_states.  Skip it outright.
            context_out = context_lse = None
        else:
            cu_ring = jnp.zeros_like(pcp_cu_q_lens_local[0]).at[1:].set(
                q_local.shape[0])
            context_out, _, context_lse = _rpa_cp_call(
                q_local,
                k_local,
                v_local,
                kv_cache_local,
                kv_lens_local,
                page_indices_local,
                cu_ring,
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=cp_rank,
                cp_group_size=pcp_size,
                kv_cache_lens=kv_cache_lens_local,
                pcp_ring_axis_name=pcp_axis,
                pcp_ring_mesh_axis_names=tuple(mesh.axis_names),
                skip_current_attn=True,
                use_causal_mask=False,
                update_kv_cache=False,
                **common)

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
            kv_cache_local,
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
