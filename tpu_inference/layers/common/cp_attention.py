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

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import merge_kv
import tpu_inference.kernels.experimental.rpa_v3_cp.write_kv as write_kv_pallas
from tpu_inference import envs
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

if envs.USE_BATCHED_RPA_KERNEL:
    import tpu_inference.kernels.experimental.batched_rpa.wrapper as rpa_cp
    logger.info_once("Using experimental batched RPA kernel")
else:
    import tpu_inference.kernels.experimental.rpa_v3_cp.kernel as rpa_cp
    logger.info_once("Using default RPA kernel")

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
    return_lse: bool = True,
    **flags,
):
    """Call with shared CP params"""
    return rpa_cp.ragged_paged_attention(
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
        return_lse=return_lse,
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


def dcp_forward_two_phase(
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
    """Two-phase DCP attention forward (cache phase + current phase)."""
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


def dcp_forward_decode_only(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    attention_metadata: AttentionMetadata,
    head_dim_original: int | None = None,
    sm_scale: float | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """DCP decode forward pass (q_len=1 per sequence) using a single shard_map."""
    if head_dim_original is None:
        head_dim_original = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata
    dcp_axis = 'dcp'
    dcp_size = mesh.shape[dcp_axis]
    proj_replicate = envs.DCP_PROJ_REPLICATE

    cp_rank_global = jnp.arange(dcp_size, dtype=jnp.int32)

    if proj_replicate:
        # Q and K/V projections are replicated across DCP ranks: every DCP rank
        # holds the full model-shard heads. No GQA expansion needed and no
        # all-gather inside shard_map — the projection already produced the right
        # head count (num_q_heads/model and num_kv_heads/model respectively).
        q_in_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
        kv_spec   = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
    else:
        # GQA/MQA: replicate KV heads to match ATTN_HEAD sharding before shard_map.
        # all-gather + stride inside _shard_fn undoes the duplication.
        tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
        num_kv_heads = k.shape[1]
        if tp_size > 1 and num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}"
                )
            k = jnp.repeat(k, tp_size // num_kv_heads, axis=1)
            v = jnp.repeat(v, tp_size // num_kv_heads, axis=1)
        q_in_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
        kv_spec   = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)

    # Output is always ATTN_HEAD-sharded: _dcp_a2a_reduce converts
    # (tokens, q_heads/model, head_dim) → (tokens*dcp, q_heads/(model*dcp), head_dim).
    q_out_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)

    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)

    common = dict(
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sliding_window=attention_chunk_size,
    )

    def _shard_fn(q_local, k_local, v_local, kv_cache_local, kv_lens_local,
                  page_indices_local, cu_q_lens_local, distribution_local,
                  cp_rank):
        # rpa_v3_cp uses token-level interleaved sharding.
        cp_kv_cache_interleaved_size = 1

        if envs.DCP_PROJ_REPLICATE:
            # Q and K/V are model-only sharded (replicated across DCP). Each DCP
            # rank already has the correct head count for the cache write and for
            # RPA — no all-gather or stride needed.
            pass
        else:
            # K/V are sharded by ATTN_HEAD (model × dcp). When num_kv_heads < tp_size,
            # jnp.repeat in the outer scope duplicates heads so each DCP rank at the
            # same model rank holds a copy of the same head. All-gather across dcp
            # collects those dcp copies; stride-slicing by (gathered / cache_heads)
            # picks the unique heads, giving num_kv_heads/model to match the cache.
            # kv_cache_local.shape[2] = num_kv_heads/model (from KV_HEAD sharding).
            k_gathered = lax.all_gather(k_local, dcp_axis, axis=1, tiled=True)
            v_gathered = lax.all_gather(v_local, dcp_axis, axis=1, tiled=True)
            cache_kv_heads = max(1, kv_cache_local.shape[2])
            kv_head_stride = max(1, k_gathered.shape[1] // cache_kv_heads)
            k_local = k_gathered[:, ::kv_head_stride, :]
            v_local = v_gathered[:, ::kv_head_stride, :]

        # Write new decode token first, then attend to the full updated cache.
        merged_kv = merge_kv(k_local, v_local)
        kv_cache_updated = write_kv_pallas.write_decode_kv(
            merged_kv=merged_kv,
            kv_cache=kv_cache_local,
            kv_lens=kv_lens_local,
            page_indices=page_indices_local,
            cu_q_lens=cu_q_lens_local,
            distribution=distribution_local,
            cp_rank=cp_rank,
            cp_group_size=dcp_size,
            cp_kv_cache_interleaved_size=cp_kv_cache_interleaved_size,
        )

        if envs.DCP_PROJ_REPLICATE:
            # Q is model-only sharded: already has num_q_heads/model heads.
            q_all_heads = q_local
        else:
            q_all_heads = lax.all_gather(q_local, dcp_axis, axis=1, tiled=True)

        # RPA use kv_len - q_len to decide cache_len, we +1 here to include new tokens.
        attn_out, _, lse = _rpa_cp_call(
            q_all_heads,
            k_local,
            v_local,
            kv_cache_updated,
            kv_lens_local + 1,
            page_indices_local,
            cu_q_lens_local,
            distribution_local,
            cp_rank=cp_rank,
            cp_group_size=dcp_size,
            update_kv_cache=False,
            return_lse=True,
            skip_cache_attn=False,
            skip_current_attn=True,
            **common,
        )

        final_output, _ = _dcp_a2a_reduce(attn_out, lse, dcp_axis, dcp_size)
        return kv_cache_updated, final_output.astype(q.dtype)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            q_in_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(ShardingAxisName.ATTN_DATA),  # kv_lens
            P(ShardingAxisName.ATTN_DATA),  # page_indices
            P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
            P(ShardingAxisName.ATTN_DATA),  # distribution
            P(ShardingAxisName.KV_CONTEXT),  # cp_rank_global
        ),
        out_specs=(kv_cache_spec, q_out_spec),
        check_vma=False,
    )(q, k, v, kv_cache, md.seq_lens, md.block_tables, md.query_start_loc,
      md.request_distribution, cp_rank_global)


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
    is_decode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """DCP attention forward.

    When envs.DCP_DECODE_ONLY_OPT is enabled, dynamically dispatches via jax.lax.cond
    based on distribution[0] == distribution[2] (i.e. num_decode_reqs == num_reqs).
    """
    if envs.DCP_DECODE_ONLY_OPT:
        is_decode_only = (md.request_distribution[0] == md.request_distribution[2])
        # Only return the attention output through lax.cond. Both branches update
        # kv_cache in-place in HBM. Returning kv_cache as part of the conditional phi
        # causes XLA copy_insertion (IndicesToCopyForConditional) to insert a 1.25 GB
        # DeepCopyInstruction on every layer (~500us/layer, 32ms/step).
        out = jax.lax.cond(
            is_decode_only,
            lambda: dcp_forward_decode_only(
                mesh=mesh,
                q=q,
                k=k,
                v=v,
                kv_cache=kv_cache,
                attention_metadata=md,
                head_dim_original=head_dim_original,
                sm_scale=sm_scale,
                attention_chunk_size=attention_chunk_size,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )[1],
            lambda: dcp_forward_two_phase(
                mesh=mesh,
                q=q,
                k=k,
                v=v,
                kv_cache=kv_cache,
                md=md,
                head_dim_original=head_dim_original,
                sm_scale=sm_scale,
                attention_chunk_size=attention_chunk_size,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )[1],
        )
        return kv_cache, out

    return dcp_forward_two_phase(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        md=md,
        head_dim_original=head_dim_original,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


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
    C = q.shape[0] // (2 * pcp_size)

    q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
                      ShardingAxisName.KV_HEAD, None, None)

    common = dict(sm_scale=sm_scale,
                  q_scale=q_scale,
                  k_scale=k_scale,
                  v_scale=v_scale)

    has_cached_kv = md.pcp.has_cached_kv
    num_reqs = md.pcp.num_reqs
    multi_req = num_reqs > 1

    def _shard_fn(q_local, k_local, v_local, kv_cache_local, kv_lens_local,
                  kv_cache_lens_local, page_indices_local, distribution_local,
                  pcp_cu_q_lens_local, pcp_q_pos_offsets_local,
                  kv_new_starts_local, kv_token_order_local):
        axis_idx = lax.axis_index(pcp_axis)
        cp_rank = jnp.reshape(axis_idx, (1, )).astype(jnp.int32)

        def all_gather_tokens(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        # ---- Cache phase --------------------------------------------------
        if not has_cached_kv:
            # Nothing cached (first chunk of a chunked prefill): the cache
            # phase would attend an empty cache, be fully masked, and have its
            # -inf result discarded by merge_attn_states.  Skip it outright.
            context_out = context_lse = None
        else:
            if not multi_req:
                # Single request: one seq spanning this rank's whole local
                # buffer (head + tail + padding).  Valid for any current-phase
                # cu_q_lens, including the per-rank clipped-tail form that
                # single-request callers build.
                cu_cache = jnp.zeros_like(pcp_cu_q_lens_local[0]).at[1:].set(
                    q_local.shape[0])
                kv_lens_cache = kv_lens_local
                kv_cache_lens_cache = kv_cache_lens_local
                page_indices_cache = page_indices_local
                distribution_cache = jnp.array([0, 0, 1], jnp.int32)
            else:
                # One cache-phase seq per request spanning its head+tail run
                # (no causal mask here, so the two need not be told apart):
                # every array is the current-phase array with the head/tail
                # duplication undone, and the seq count is half.  The ring
                # runs in lock-step across ranks, so everything passed here is
                # rank-invariant; q_pos_offsets, the one per-rank array, is
                # not passed.
                cu_cache = pcp_cu_q_lens_local[0][0::2]
                kv_lens_cache = kv_lens_local[0::2]
                kv_cache_lens_cache = kv_cache_lens_local[0::2]
                pps = page_indices_local.shape[0] // kv_lens_local.shape[0]
                page_indices_cache = page_indices_local.reshape(
                    -1, pps)[0::2].reshape(-1)
                distribution_cache = jnp.zeros_like(
                    distribution_local).at[2].set(distribution_local[2] // 2)
            context_out, _, context_lse = _rpa_cp_call(
                q_local,
                k_local,
                v_local,
                kv_cache_local,
                kv_lens_cache,
                page_indices_cache,
                cu_cache,
                distribution_cache,
                cp_rank=cp_rank,
                cp_group_size=pcp_size,
                kv_cache_lens=kv_cache_lens_cache,
                pcp_ring_axis_name=pcp_axis,
                pcp_ring_mesh_axis_names=tuple(mesh.axis_names),
                skip_current_attn=True,
                use_causal_mask=False,
                update_kv_cache=False,
                **common)

        # ---- Current phase ------------------------------------------------
        # Local Q (head+tail chunks) attends the all-gathered current K/V.  A
        # single request with a page-aligned chunk hands the kernel the
        # rank-order buffer and lets it remap addresses (pcp_chunk_size);
        # otherwise the gathered K/V is reordered into request-major token
        # order here, which is what kv_new_starts indexes.
        page_size = kv_cache_local.shape[1]
        remap_kv = not multi_req and C >= page_size and C % page_size == 0
        k_curr = all_gather_tokens(k_local)
        v_curr = all_gather_tokens(v_local)
        if not remap_kv:
            k_curr = jnp.take(k_curr, kv_token_order_local, axis=0)
            v_curr = jnp.take(v_curr, kv_token_order_local, axis=0)
        # Each request's tail seq performs the fused strided KV write.
        max_seqs = kv_lens_local.shape[0]
        kv_write_seq_mask = jnp.zeros(max_seqs,
                                      jnp.int32).at[1:2 * num_reqs:2].set(1)
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
            kv_new_starts=None if remap_kv else kv_new_starts_local,
            kv_write_seq_mask=kv_write_seq_mask,
            pcp_chunk_size=C if remap_kv else None,
            skip_cache_attn=True,
            use_causal_mask=use_causal_mask,
            update_kv_cache=update_kv_cache,
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
            P(),  # pcp.kv_new_starts: replicated
            P(),  # pcp.kv_token_order: replicated
        ),
        out_specs=(kv_cache_spec, q_spec),
        check_vma=False,
    )(q, k, v, kv_cache, md.seq_lens, md.pcp.kv_cache_lens, md.block_tables,
      md.request_distribution, md.pcp.query_start_loc, md.pcp.q_pos_offsets,
      md.pcp.kv_new_starts, md.pcp.kv_token_order)
