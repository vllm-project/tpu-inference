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
"""Serving layer for the fused expert-parallel MoE kernel.

fused_ep_moe_v2 wraps the kernel builder in a shard_map over the
expert-parallel mesh axis: quantize, dispatch, kernel, combine.
"""
import threading

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK, MAX_ROUTING_BLOCK, QB4, ROWBLK, WeightFormat,
    act_scale_slab_rows, align_up, build_routing_tables, expert_visit_list,
    local_slab_rows, ragged_stride_bound, routing_block, shard_count_vector,
    shard_expert_slabs, shard_push_tables_in_rows, shard_token_gather,
    shard_transport_tables_in_blocks, weight_form, weight_format_of_dtype)
from tpu_inference.kernels.fused_moe.v2.kernel import (
    build_fused_ep_moe_kernel, rowquant_fp8)
from tpu_inference.kernels.fused_moe.v2.router_ops import pallas_select


def _combine_arrivals(arrivals, arrival_scales, pos, topk_weights, out_dtype):
    """Combine arrivals [rows, lane blocks, 128] into [t_local, hidden]."""
    t_local, topk = pos.shape
    _, lane_blocks, lanes = arrivals.shape
    hidden = lane_blocks * lanes
    arrival_rows = pos.T.reshape(-1)  # k-major [K*t_local]
    rows_fp8 = arrivals[arrival_rows]  # f8 [K*t, lane blocks, 128]
    row_scales = arrival_scales[arrival_rows][:, :1]  # f32 row scales
    weights_f32 = topk_weights.astype(jnp.float32)
    terms = []
    for k in range(topk):
        slot_rows = lax.slice(rows_fp8, (k * t_local, 0, 0),
                              ((k + 1) * t_local, lane_blocks, lanes))
        slot_scales = lax.slice(row_scales, (k * t_local, 0),
                                ((k + 1) * t_local, 1))
        terms.append(
            slot_rows.astype(jnp.float32) * slot_scales[:, :, None] *
            weights_f32[:, k:k + 1, None])
    return sum(terms).astype(out_dtype).reshape(t_local, hidden)


def _pack_routing_blob(topk_idx, row_scale):
    """Fold idx [t, K] and row scales [t, 1] into one [t, K+1] i32 blob.

    row_scale is None where the rows were never quantized and so carry no
    scale; the blob is then the indices alone, [t, K].
    """
    # The router weight stays local: the destination combine applies it in
    # f32 from its own copy, so the wire never carries it.
    if row_scale is None:
        return topk_idx.astype(jnp.int32)
    return jnp.concatenate([
        topk_idx.astype(jnp.int32),
        lax.bitcast_convert_type(row_scale.astype(jnp.float32), jnp.int32)
    ],
                           axis=1)


def _unpack_routing_blob(blob_g, topk):
    """Inverse of _pack_routing_blob (bit-exact round trip).

    Returns a row scale of None for the blob that carries none.
    """
    topk_idx_g = blob_g[:, :topk]
    if blob_g.shape[1] == topk:
        return topk_idx_g, None
    row_scale_g = lax.bitcast_convert_type(blob_g[:, topk:], jnp.float32)
    return topk_idx_g, row_scale_g


# Per-config cache of the shard_map'd MoE callable; a fresh shard_map
# body would re-trace the whole kernel for every hidden layer.
_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def fused_ep_moe_v2(x,
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    gating,
                    w1_bias=None,
                    w2_bias=None,
                    *,
                    topk,
                    renormalize,
                    mesh,
                    capacity,
                    block=None,
                    ragged_stride=None,
                    weight_format=WeightFormat.FP8,
                    rhs_qb=None,
                    act_fn="silu"):
    """Run one MoE layer through the fused expert-parallel kernel.

    x [tokens, hidden], w1 [experts, hidden, 2 * inter], w2 [experts,
    inter, hidden] and gating [tokens, experts], all sharded over the mesh
    axis. Returns [tokens, hidden], sharded like x.

    weight_format names one of the kernel's accepted weight forms and
    decides everything that follows from it. w1_scale and w2_scale are per
    output channel for the per-channel forms (fp8 e4m3, int8), [experts,
    2 * inter] and [experts, hidden], and per contraction block for fp4
    e2m1, [experts, blocks, 2 * inter] and [experts, blocks, hidden]; on an
    unquantized weight (bf16) there are no scales and both must be None.
    Both are taken in the kernel's own operand layout, so a table shaped
    for a loader rather than for the kernel is refused. The token rows are
    quantized to fp8 only where the format's matmuls take fp8 rows: an
    unquantized weight gets unquantized activations, with no quantize and
    dequantize pair inserted around a model that never asked for one.

    w1_bias [experts, 1, 2 * inter] and w2_bias [experts, 1, hidden] are
    optional and independent. The gate and up halves are added to the first
    matmul's post-scale accumulator before the activation; the down bias is
    added to the second matmul's post-scale row before it is quantized for
    the wire. Under expert parallelism the second matmul is sharded on the
    expert axis rather than on its contraction, so a routed row's whole down
    projection is produced on one shard and the bias enters it exactly once;
    the combine then weights it by that row's router weight, which is the
    per-selected-expert weighting the sum wants.

    block and ragged_stride default to the values this function derives;
    an explicitly passed one still has to divide tokens // ep * topk and to
    reach the no-drop bound respectively. act_fn selects the fused FFN
    activation from the kernel's ACT_FNS and is a build-time constant.
    """
    form = weight_form(weight_format)
    rhs_qb = QB4 if rhs_qb is None else int(rhs_qb)
    # The weights themselves, against the format they were named as. This
    # validated only that the SCALES matched, which was largely
    # self-limiting while fp8 and fp4 were the whole table: four-bit weights
    # stream as packed u32 words, so a wrong format name died on the
    # ref-level bitcast or on a shape mismatch soon after. int8 and bf16
    # declare IDENTICAL slab shapes and differ only in element size, one
    # byte against two, so naming one and passing the other is a half-slab
    # read with correct-looking shapes all the way down. This is the
    # documented public entry, so the check belongs where the operands come
    # in rather than only in the serving adapter above it.
    for name, w in (("w1", w1), ("w2", w2)):
        actual = weight_format_of_dtype(w.dtype)
        if actual != weight_format:
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert weights and "
                f"{name} is {jnp.dtype(w.dtype).name}"
                f"{f', which is the {actual!r} form' if actual else ''}")
    if form.has_scales != (w1_scale is not None):
        raise ValueError(f"weight format {weight_format!r} carries "
                         f"{form.scale_layout} weight scales and w1_scale is "
                         f"{'present' if w1_scale is not None else 'absent'}")
    if (w1_scale is None) != (w2_scale is None):
        raise ValueError(
            "both weight scales are supplied together or neither is")
    T, hidden = x.shape
    e_total = w1.shape[0]
    inter = w1.shape[2] // 2
    # Where the weight scales' layout is fixed: [E, N] per output channel,
    # [E, blocks, N] per contraction block, which is what the kernel's own
    # operands are. The scales are constants of the weights, so whatever
    # prepares the weights gives them that shape once; a reshape here would
    # instead stand between the parameter and the kernel call on every call,
    # and it is a layout change rather than free. A table carrying a
    # loader's singleton axes is refused rather than reshaped.
    if form.has_scales:
        ndim = 3 if form.scale_layout == "per_contraction_block" else 2
        for name, s, n in (("w1_scale", w1_scale, 2 * inter),
                           ("w2_scale", w2_scale, hidden)):
            if s.ndim != ndim or s.shape[0] != e_total or s.shape[-1] != n:
                want = (e_total, "blocks", n) if ndim == 3 else (e_total, n)
                raise ValueError(f"{name} layout {tuple(s.shape)} is not the "
                                 f"{form.scale_layout} form {want} the kernel "
                                 f"takes")
    (ax, ) = mesh.axis_names
    ep = mesh.shape[ax]
    g_local = e_total // ep
    t_local = T // ep
    P = jax.sharding.PartitionSpec
    stride_bound = ragged_stride_bound(T, topk, e_total, capacity)
    if ragged_stride is None:
        ragged_stride = stride_bound
    if block is None:
        block = routing_block(t_local, topk)
    if ragged_stride % capacity != 0:
        raise ValueError(
            f"ragged_stride must be a multiple of capacity {capacity}; got "
            f"{ragged_stride}. Pass {stride_bound}.")
    if ragged_stride < stride_bound:
        raise ValueError(
            f"ragged_stride {ragged_stride} is below the no-drop bound "
            f"{stride_bound}, so one shard's slab could bleed into the "
            f"next. Pass {stride_bound} or more.")
    has_w1_bias = w1_bias is not None
    has_w2_bias = w2_bias is not None
    kfn = build_fused_ep_moe_kernel(g_local=g_local,
                                    capacity=capacity,
                                    hidden=hidden,
                                    inter=inter,
                                    ep=ep,
                                    ragged_rows_alloc=ragged_stride,
                                    weight_format=weight_format,
                                    rhs_qb=rhs_qb,
                                    act_fn=act_fn,
                                    has_w1_bias=has_w1_bias,
                                    has_w2_bias=has_w2_bias)

    def local_fn(x_l, w1_l, w2_l, *scales_gating_biases):
        """One shard's half of the layer: route, dispatch, kernel, combine."""
        # The scales come before the gating logits and the biases after
        # them, which is the operand order the scaled formats have always
        # had: a format that supplies no scales drops them out of the middle
        # rather than moving anything that stayed.
        operands = iter(scales_gating_biases)
        w1s_l = next(operands) if form.has_scales else None
        w2s_l = next(operands) if form.has_scales else None
        gate_l = next(operands)
        me = lax.axis_index(ax)
        rows_bf16 = x_l.astype(jnp.bfloat16)
        if form.quantized_activations:
            q_l, row_scale_l = rowquant_fp8(rows_bf16)
        else:
            # The rows go on the wire as they arrived. There is no scale to
            # carry, so nothing downstream builds or ships one.
            q_l, row_scale_l = rows_bf16, None
        q_g = lax.all_gather(q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK,
                                         HIDDEN_LANE_BLOCK),
                             ax,
                             axis=0,
                             tiled=True)

        scores = jax.nn.softmax(gate_l, axis=-1)
        select_rows = MAX_ROUTING_BLOCK
        while t_local % select_rows:
            select_rows //= 2
        topk_weights, topk_idx = pallas_select(scores,
                                               topk=topk,
                                               block_rows=select_rows)
        # A row of scores carrying no real value routes nowhere. It was
        # dropped by arithmetic accident: the selector gives every slot of
        # such a row a large negative sentinel, and the renormalization below
        # divided by their sum, which OVERFLOWED float32 to -inf and so
        # returned exactly zero. That outcome is a property of the
        # accumulation dtype and the association order, not of the design --
        # a wider accumulator, a reassociating compiler pass, or a different
        # renormalization would silently turn such a row from dropped into
        # routed at full weight to expert zero. Mask it explicitly instead.
        row_routes = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
        if renormalize:
            denom = topk_weights.sum(axis=-1, keepdims=True)
            topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
        topk_weights = jnp.where(row_routes, topk_weights, 0.0)
        # A count of the masked rows would be the other half of this: an
        # incident today is degraded output against a completely clean log.
        # It is not added here because the only in-trace channel for it,
        # jax.debug.print, makes the program refuse to lower for TPU from a
        # host, which is how this repository verifies that a change served
        # the same program. A counter belongs in a metrics channel outside
        # the trace.
        # One all-gather carries the indices and, where the rows were
        # quantized, their row scales, packed into one integer blob.
        blob_g = lax.all_gather(_pack_routing_blob(topk_idx, row_scale_l),
                                ax,
                                axis=0,
                                tiled=True)
        topk_idx_g, row_scale_g = _unpack_routing_blob(blob_g, topk)

        routing = build_routing_tables(topk_idx_g,
                                       e_total=e_total,
                                       ep=ep,
                                       t_local=t_local,
                                       block=block,
                                       tile_m=capacity,
                                       shard_stride=ragged_stride)
        block_tables = shard_transport_tables_in_blocks(routing,
                                                        me,
                                                        e_total=e_total,
                                                        ep=ep)
        row_tables = shard_push_tables_in_rows(routing,
                                               me,
                                               e_total=e_total,
                                               ep=ep)
        # Both slab tables scatter onto this shard's own slab. The rows of
        # the other shards are dropped where they are computed rather than
        # built into a replicated slab and sliced away afterwards.
        slab_row = local_slab_rows(routing, me, shard_stride=ragged_stride)
        token_gather = shard_token_gather(routing,
                                          me,
                                          shard_stride=ragged_stride)
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on. It exists only where the rows were quantized.
        #
        # The slab is built flat and handed over in the dense lane-block
        # view, which is the same bytes in the same order and so costs
        # nothing. A column would cost a copy of the whole slab: [rows, 1]
        # is padded out to a full lane block on the way to the kernel, and
        # that copy is the largest single piece of glue above a thousand
        # rows. The kernel rebuilds the column a tile at a time instead.
        #
        # The scatter runs the length of the view rather than of the slab.
        # Those extra elements are past the slab's last row, so a pair the
        # rebase sent past the end lands on one of them instead of being
        # dropped; nothing reads them, and the slab's own rows are the same
        # either way.
        if form.quantized_activations:
            scale_bits = lax.bitcast_convert_type(
                jnp.repeat(row_scale_g[:, 0], topk), jnp.int32)
            scale_rows = act_scale_slab_rows(ragged_stride)
            scale_slab = lax.bitcast_convert_type(
                jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK, ),
                          jnp.int32).at[slab_row].add(scale_bits, mode="drop"),
                jnp.float32).reshape(scale_rows, HIDDEN_LANE_BLOCK)
        else:
            scale_slab = None
        expert_rows, slab_base = shard_expert_slabs(routing,
                                                    me,
                                                    e_total=e_total,
                                                    ep=ep)
        recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
        # The bias tables are per expert and per output channel on every
        # weight format, so they take one layout: [G, N].
        bias_it = operands
        w1b_k = next(bias_it).reshape(g_local, 2 *
                                      inter) if has_w1_bias else None
        w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
        # The empty experts drop out of the visit list, so an empty
        # expert's weight slab never streams. How MANY are visited is a
        # row of the count table rather than its own reduction, so this
        # builder's second value is not read; it leaves the program with
        # the rest of the dead code, and its own test still sees it.
        visit, _ = expert_visit_list(expert_rows, g_local)
        # One pass builds every count the kernel needs, still spread over
        # the expert-parallel axis; the kernel's scalar core closes them.
        counts = shard_count_vector(routing,
                                    expert_rows,
                                    me,
                                    e_total=e_total,
                                    ep=ep)

        # The kernel takes five of the seven transport tables and two of
        # the three push tables. The two totals are rows of `counts`, and
        # the push source is the same table as the commit offset, so it
        # goes over once.
        kernel_tables = (*block_tables[:3], *block_tables[4:6],
                         *row_tables[:2])
        arrivals, arrival_scales = kfn(kernel_tables,
                                       token_gather,
                                       expert_rows,
                                       slab_base,
                                       q_g,
                                       scale_slab,
                                       w1_l,
                                       w2_l,
                                       w1s_l,
                                       w2s_l,
                                       w1b_k,
                                       w2b_k,
                                       recv_rows=recv_rows,
                                       visit=visit,
                                       counts=counts)

        # The destination's own table: one arrival row per selection slot.
        pos = lax.dynamic_slice(routing.arrival_row, (me * t_local, 0),
                                (t_local, topk))
        return _combine_arrivals(arrivals, arrival_scales, pos, topk_weights,
                                 x_l.dtype)

    # The biases ride the same expert-axis sharding as the weights they
    # belong to, so no shard ever holds a bias for an expert it does not own.
    bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    scale_args = (w1_scale, w2_scale) if form.has_scales else ()
    in_specs = (P(ax), ) * (4 + len(scale_args) + len(bias_args))
    args = (x, w1, w2) + scale_args + (gating, ) + bias_args
    NS = jax.sharding.NamedSharding
    args = tuple(
        jax.device_put(a, NS(mesh, sp)) for a, sp in zip(args, in_specs))
    # local_fn closes over config-static values only -- the per-call data
    # are the shard_map arguments -- so this key is exact.
    key = (mesh, T, hidden, e_total, inter, topk, bool(renormalize), capacity,
           block, ragged_stride, weight_format, rhs_qb, act_fn, x.dtype,
           w1.dtype, w2.dtype, form.has_scales
           and w1_scale.dtype, gating.dtype, has_w1_bias
           and w1_bias.dtype, has_w2_bias and w2_bias.dtype)
    sm = _LAYER_SM_CACHE.get(key)
    if sm is None:
        with _LAYER_SM_CACHE_LOCK:
            sm = _LAYER_SM_CACHE.get(key)
            if sm is None:
                sm = jax.shard_map(local_fn,
                                   mesh=mesh,
                                   in_specs=in_specs,
                                   out_specs=P(ax),
                                   check_vma=False)
                _LAYER_SM_CACHE[key] = sm
    return sm(*args)
