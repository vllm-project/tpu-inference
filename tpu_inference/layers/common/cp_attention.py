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
    cache_phase: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """PCP attention forward (multi-request).

    The token buffer holds `num_reqs` fixed request LANES, each of
    `Lreq = 2*pcp*C` tokens. Every request is head-tail split into `2*pcp`
    chunks and the global buffer is laid out RANK-MAJOR then LANE then
    head|tail, so each pcp rank's contiguous shard is
    `concat_i [head_r,i | tail_r,i]`. Request `i`'s local Q is therefore the
    static slice `q_local[i*2C : (i+1)*2C]`.

    Per request (a lane), inside the shard_map body:
      1. all_gather Q tokens   cache phase needs full sequence view per rank
      2. cache phase           attend full Q against this rank's KV cache shard
      3. _pcp_rs_reduce        reduce_scatter: each rank collects its token chunk
      4. current phase         local Q (head+tail) attends all-gathered current KV
      5. merge_attn_states     lse-weighted combine

    Each per-lane call presents exactly ONE request as two seqs (head, tail), so
    `write_last_seq_only` and the scalar `pcp_chunk_size` remap stay valid with
    no kernel changes. The kv cache is threaded through the lanes; requests
    write disjoint pages via their own block-table rows.
    """
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT
    pcp_size = get_mesh_shape_product(mesh, pcp_axis)
    two_p = 2 * pcp_size
    padded_q_len = q.shape[0]
    num_reqs = int(md.padded_num_reqs)
    assert padded_q_len % (num_reqs * two_p) == 0, (
        f"padded tokens {padded_q_len} not divisible by "
        f"num_reqs*2*pcp {num_reqs * two_p}")
    per_req_tokens = padded_q_len // num_reqs  # one lane = 2*pcp*C
    C = per_req_tokens // two_p
    local_per_req = 2 * C  # this rank's tokens for one lane
    pages_per_seq = md.block_tables.shape[0] // num_reqs

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

    # Cache-phase strategy. The PER-REQUEST choice is made by the runner (it
    # knows each request's own cached length) and arrives as the static split
    # point `num_gather_kv`; see PCPMetadata. `cache_phase` overrides it and
    # forces one algorithm for every lane (tests, A/B).
    cache_pages = md.pcp.cache_pages
    num_gather_kv = md.pcp.num_gather_kv
    phase = None if cache_phase is None else cache_phase.lower()
    assert phase in (None, "gather_kv", "gather_q", "ring"), (
        f"cache_phase must be 'gather_kv', 'gather_q' or 'ring', got {phase!r}"
    )

    def _shard_fn(q_local, k_local, v_local, kv_cache_local, kv_lens_local,
                  kv_cache_lens_local, page_indices_local):
        axis_idx = lax.axis_index(pcp_axis)
        cp_rank = jnp.reshape(axis_idx, (1, )).astype(jnp.int32)

        def all_gather_tokens(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        def to_token_order(x):  # rank-order chunks -> global token order
            # `x` is ONE lane's local tokens [2C, ...]: the all-gather rebuilds
            # that request's 2*pcp chunks, then un-permute into token order.
            return all_gather_tokens(x).reshape(two_p, C,
                                                *x.shape[1:])[inv_row].reshape(
                                                    per_req_tokens, *x.shape[1:])

        page_size = kv_cache_local.shape[1]
        # remap_kv: if C aligns with page_size, all_gather_tokens() avoids an
        # extra gather-reorder.
        remap_kv = (C >= page_size) and (C % page_size == 0)
        # Head/tail geometry for THIS rank. Identical for every lane; the only
        # per-lane part is how much of the tail chunk is real.
        head_off = (axis_idx * C).astype(jnp.int32)
        tail_off = ((two_p - 1 - axis_idx) * C).astype(jnp.int32)
        q_pos_offsets = jnp.stack([head_off, tail_off])
        # The fused current phase presents head+tail as two prefill seqs.
        dist_one = jnp.array([0, 0, 1], jnp.int32)
        dist_two = jnp.array([0, 0, 2], jnp.int32)

        # ---- Cache-phase algorithm groups --------------------------------
        # The runner orders lanes so each cache-phase algorithm owns a
        # CONTIGUOUS range -- the same shape as the RPA kernel's
        # decode / prefill / mixed split -- and hands us the split point:
        #   lanes [0, num_gather_kv)        -> gather-KV
        #   lanes [num_gather_kv, num_reqs) -> gather-Q
        # The split is STATIC, so each group gathers exactly what it uses and
        # each lane picks its algorithm at trace time (no masking, no wasted
        # collective). gather-KV in particular materialises the all-gathered
        # context on every rank, so it must NOT be sized for the whole batch.
        n_gather_kv = min(num_gather_kv, num_reqs)

        def _lane_phase(i):
            if phase is not None:  # explicitly forced for every lane
                return phase
            return "gather_kv" if i < n_gather_kv else "gather_q"

        def _one_lane(i, kv_cache_in):
            """One request lane: cache phase + current phase, LSE-merged.

            Returns (out[2C, ...], updated kv cache). The kernel sees exactly
            one request as two seqs (head, tail).
            """
            lo = i * local_per_req
            q_i = lax.dynamic_slice_in_dim(q_local, lo, local_per_req, axis=0)
            k_i = lax.dynamic_slice_in_dim(k_local, lo, local_per_req, axis=0)
            v_i = lax.dynamic_slice_in_dim(v_local, lo, local_per_req, axis=0)

            # This lane's request lengths, replicated over its two fused seqs.
            total_len = kv_lens_local[i]
            num_computed = kv_cache_lens_local[i]
            kv_lens_i = jnp.stack([total_len,
                                   total_len]).astype(kv_lens_local.dtype)
            kv_cache_lens_i = jnp.stack(
                [num_computed, num_computed]).astype(kv_cache_lens_local.dtype)
            # Both fused seqs are the same request, so the tail (the writing
            # seq) needs a COPY of this request's block-table row.
            pi_i = lax.dynamic_slice_in_dim(page_indices_local,
                                            i * pages_per_seq,
                                            pages_per_seq,
                                            axis=0)
            pi_i2 = jnp.concatenate([pi_i, pi_i], axis=0)
            tail_real = jnp.clip(total_len - num_computed - tail_off, 0, C)
            # cu_q_lens must be max_num_seqs+1 == 3 entries here.
            cu_i = jnp.array([0, C, C], jnp.int32).at[2].set(C + tail_real)

            # ---- Cache phase ------------------------------------------------
            lane_phase = _lane_phase(i)
            kv_cache_temp = kv_cache_in
            if cache_pages == 0:
                # Nothing cached anywhere in this batch (first chunk of a
                # chunked prefill): the cache phase would attend an empty
                # cache, be fully masked, and have its -inf result discarded
                # by merge_attn_states. Skip it outright.
                context_out = context_lse = None
            elif lane_phase == "ring":
                cu_ring = jnp.array([0, local_per_req, local_per_req],
                                    jnp.int32)
                context_out, _, context_lse = _rpa_cp_call(
                    q_i,
                    k_i,
                    v_i,
                    kv_cache_in,
                    kv_lens_i,
                    pi_i2,
                    cu_ring,
                    dist_one,
                    cp_rank=cp_rank,
                    cp_group_size=pcp_size,
                    kv_cache_lens=kv_cache_lens_i,
                    pcp_ring_axis_name=pcp_axis,
                    pcp_ring_mesh_axis_names=tuple(mesh.axis_names),
                    skip_current_attn=True,
                    use_causal_mask=False,
                    update_kv_cache=False,
                    **common)
            elif lane_phase == "gather_kv":
                cu_kv = jnp.array([0, local_per_req, local_per_req], jnp.int32)
                kv_src = jnp.take(kv_cache_in, pi_i[:cache_pages], axis=0)
                kv_tok = lax.all_gather(kv_src, pcp_axis, axis=2, tiled=False)
                n_pages_tok = kv_src.shape[0] * pcp_size
                kv_tok = kv_tok.reshape(n_pages_tok, kv_src.shape[1],
                                        *kv_src.shape[2:])
                # Two seqs in this call, so the block table is tiled twice.
                pi_tok = jnp.tile(
                    jnp.arange(n_pages_tok, dtype=page_indices_local.dtype), 2)
                context_out, _, context_lse = _rpa_cp_call(
                    q_i,
                    k_i,
                    v_i,
                    kv_tok,
                    kv_lens_i,
                    pi_tok,
                    cu_kv,
                    dist_one,
                    cp_rank=jnp.zeros((1, ), jnp.int32),
                    cp_group_size=1,
                    kv_cache_lens=kv_cache_lens_i,
                    skip_current_attn=True,
                    use_causal_mask=False,
                    update_kv_cache=False,
                    **common)
            else:
                # gather-Q: all_gather this lane's Q tokens so every rank sees
                # the request's full sequence (pcp_size * 2C = per_req_tokens).
                q_all_tokens = all_gather_tokens(q_i)
                cu_cache = jnp.array([0, per_req_tokens, per_req_tokens],
                                     jnp.int32)
                context_out, kv_cache_temp, context_lse = _rpa_cp_call(
                    q_all_tokens,
                    k_i,
                    v_i,
                    kv_cache_in,
                    kv_lens_i,
                    pi_i2,
                    cu_cache,
                    dist_one,
                    cp_rank=cp_rank,
                    cp_group_size=pcp_size,
                    kv_cache_lens=kv_cache_lens_i,
                    skip_current_attn=True,
                    use_causal_mask=False,
                    update_kv_cache=False,
                    **common)
                # reduce_scatter so each rank gets back its own 2*C chunk.
                context_out, context_lse = _pcp_rs_reduce(
                    context_out, context_lse, pcp_axis, pcp_size)

            # ---- Current phase ----------------------------------------------
            k_curr = all_gather_tokens(k_i) if remap_kv else to_token_order(k_i)
            v_curr = all_gather_tokens(v_i) if remap_kv else to_token_order(v_i)
            curr_out, kv_cache_out, curr_lse = _rpa_cp_call(
                q_i,
                k_curr,
                v_curr,
                kv_cache_temp,
                kv_lens_i,
                pi_i2,
                cu_i,
                dist_two,
                cp_rank=cp_rank,
                cp_group_size=pcp_size,
                kv_cache_lens=kv_cache_lens_i,
                q_pos_offsets=q_pos_offsets,
                pcp_chunk_size=(C if remap_kv else None),
                skip_cache_attn=True,
                use_causal_mask=use_causal_mask,
                update_kv_cache=update_kv_cache,
                write_last_seq_only=True,
                **common)

            # With nothing cached the current phase already IS the answer.
            if context_out is None:
                out_i = curr_out
            else:
                out_i, _ = merge_attn_states(context_out, context_lse, curr_out,
                                             curr_lse)
            return out_i, kv_cache_out

        outs = []
        kv_cache_cur = kv_cache_local
        for i in range(num_reqs):
            out_i, kv_cache_cur = _one_lane(i, kv_cache_cur)
            outs.append(out_i)
        out = outs[0] if num_reqs == 1 else jnp.concatenate(outs, axis=0)
        return kv_cache_cur, out.astype(q.dtype)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(),  # kv_lens (per request): replicated
            P(),  # pcp.kv_cache_lens (per request): replicated
            P(),  # page_indices: replicated
        ),
        out_specs=(kv_cache_spec, q_spec),
        check_vma=False,
    )(q, k, v, kv_cache, md.seq_lens, md.pcp.kv_cache_lens, md.block_tables)
