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
"""Fused expert-parallel MoE kernel: one Pallas TPU kernel computes the
expert half of an MoE layer for one expert-parallel shard and pushes each
result row to the shard that owns its token, so the layer needs no dense
combine reduce-scatter."""
import functools
import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# The layout constants this kernel is built against, and the host module
# whose accounting decides what it may declare.
from tpu_inference.kernels.fused_moe.v2 import host
from tpu_inference.kernels.fused_moe.v2.host import (FP4, FP8, FP8_MAX,
                                                     HIDDEN_LANE_BLOCK, NBUF,
                                                     OUT_PARITIES, PACK4, QB4,
                                                     ROWBLK,
                                                     WEIGHT_PREFETCH_DISTANCE)
# The clamped GPT-OSS activation is the grouped-matmul kernel's own tested
# implementation, called here rather than copied.
from tpu_inference.kernels.megablox.gmm_v2 import swigluoai
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

DN = (((1, ), (0, )), ((), ()))
# The single-axis mesh name the layer's shard_map uses.
AXIS = "d"
# Columns of the intermediate the requantization does at a time. Chunking
# it keeps the live bf16 intermediate down; the width is a tuned choice.
QCHUNK = 512
# DMA priority the weight refills take, so that they stay off the in-order
# queue the token gather issues on.
WEIGHT_DMA_PRIORITY = 1
# The collective this kernel's barrier and remote copies belong to. Shared
# with any other collective kernel built with the same id.
COLLECTIVE_ID = 0
# The jax whose private ref-level bitcast the four-bit weight stream binds
# to. Recorded so the failure names the version this was built against
# rather than only the version that broke it.
_REF_BITCAST_JAX_VERSION = "0.10.2"
# The scalar-prefetch operands, in the order the kernel body unpacks them.
# The grid spec is told the total, so a table cannot be added on one side
# without the other.
# These are what the KERNEL takes, which is no longer everything the
# builders return: the two totals and the visit count are rows of the
# count table instead, and the push source is the same table as the
# commit offset, so it rides once. The builders' own signatures are
# unchanged and the values the kernel stopped taking leave the program
# with the rest of the dead code.
N_TRANSPORT_TABLES = 5  # of the seven shard_transport_tables_in_blocks has
N_PUSH_TABLES = 2  # of the three shard_push_tables_in_rows has
N_GATHER_TABLES = 1  # the token-gather table
N_SLAB_TABLES = 2  # what shard_expert_slabs returns: (rows, base)
N_VISIT_TABLES = 1  # the visit list; its count is a row of the table below
N_COUNT_TABLES = 1  # shard_count_vector: [N_COUNTS, ep] i32
N_PREFETCH = (N_TRANSPORT_TABLES + N_PUSH_TABLES + N_GATHER_TABLES +
              N_SLAB_TABLES + N_VISIT_TABLES + N_COUNT_TABLES)
# The HBM operands, in the order the kernel body unpacks them. Only these
# three are on every build: the rest -- the activation row scale, the two
# weight scale tables, the two expert bias tables -- are there where the
# weight format and the model carry them, and each absent one has to drop
# out of the MIDDLE of the tuple, leaving the operands after it in place.
N_STAGING_INPUTS = 1  # the ungathered token buffer
N_WEIGHT_SLABS = 2  # the gate/up and down weight slabs
N_SCALE_TABLES = 2  # a weight scale table per matmul, where the format has
#                     them

# The FFN activations this kernel can fuse. "silu" is the default and the
# only one the kernel carried before the selector existed; "swigluoai" is
# the clamped GPT-OSS form, whose math is the grouped-matmul kernel's
# function imported above rather than a second copy.
ACT_FNS = ("silu", "swigluoai")


def _apply_act(gate, up, act_fn):
    """The FFN activation, on the POST-SCALE gate and up halves.

    Both weight forms reach here with the accumulator already multiplied
    by its scales, which the clamped form requires: its clip is defined on
    the true activation value, not on a raw accumulator.
    """
    if act_fn == "silu":
        return jax.nn.silu(gate) * up
    if act_fn == "swigluoai":
        return swigluoai(gate, up)
    raise NotImplementedError(
        f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}")


def _row_scale(amax):
    """The row scale and its inverse, with the rows that have no usable one.

    A row whose maximum is zero has nothing to scale by, and a row whose
    maximum is not finite has nothing USEFUL to scale by: the reciprocal of
    an infinite maximum is zero, so every finite value in that row would
    multiply to zero and only the infinity would survive, as a NaN. That
    leaves a plausible-looking mostly-zero row where the row's real content
    was, and sends an infinite scale over the wire for the combine at the
    destination shard to multiply back in. Both cases are answered the same
    way, by the caller zeroing the row whole against `usable`: an obviously
    empty row rather than a quietly wrong one.
    """
    scale = amax.astype(jnp.float32) / FP8_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    return scale, sinv, usable


def rowquant_fp8(x):
    """Per-row dynamic fp8 quant, reducing and applying in bf16."""
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale, sinv, usable = _row_scale(amax)
    return _quant_apply(x, sinv.astype(x.dtype), usable), scale


def _absmax_rows(x):
    return jnp.max(jnp.abs(x), axis=-1, keepdims=True)


def _quant_apply(x, sinv_native, usable):
    """Quantize a row, or zero it whole where its scale is unusable."""
    return jnp.where(usable, x * sinv_native, jnp.zeros_like(x)).astype(FP8)


def wire_row_and_scale(acc2,
                       mid_scale,
                       *,
                       w2s=None,
                       w2b=None,
                       wire_dtype=FP8,
                       tile_m=None):
    """One tile's rows as they go on the wire, and the scale that rides with
    them.

    The tile epilogue: apply the intermediate row scale, apply the
    per-channel weight scale where the format has one, add the down bias
    where the model carries one, and quantize for the wire where the wire is
    eight-bit. Every line is plain jnp on values, so it lives here rather
    than inside the Pallas body -- the down bias is the only operand
    placement in the layer with no host-side witness otherwise, and its
    add-once property is the one correctness claim the biases commit makes.

    mid_scale is None for the formats whose second matmul took bf16 rows and
    whose accumulator is therefore already the true value. w2s is the
    per-channel weight scale the four-bit path has already folded into its
    block sums and passes as None. w2b is the down bias, which lands on the
    POST-SCALE row and before the wire quantization: expert parallelism
    shards the second matmul on the expert axis rather than on its
    contraction, so a routed row's whole down projection is computed in one
    place and this add happens exactly once for it. The destination's
    per-slot router weight then weights the bias per selected expert, which
    is what the sum over selected experts wants.

    An unquantized wire carries the row as it is, against a scale mirror of
    exactly one, so the destination combine reads one transport for every
    format and multiplying by it changes no value. The mirror is 16 bytes
    per row against a row of 2 * hidden, so keeping it costs well under one
    percent of the wire.
    """
    down = acc2 if mid_scale is None else acc2 * mid_scale
    if w2s is not None:
        down = down * w2s
    if w2b is not None:
        down = down + w2b
    down_bf16 = down.astype(jnp.bfloat16)
    if wire_dtype is FP8:
        return rowquant_fp8(down_bf16)
    rows = tile_m if tile_m is not None else down_bf16.shape[0]
    return down_bf16, jnp.ones((rows, 1), jnp.float32)


def _intermediate_rows(acc1,
                       act_scale,
                       inter,
                       *,
                       w1s,
                       act_fn,
                       w1b,
                       requantize=True):
    """acc1's gate and up halves through the activation, as the second
    matmul's left-hand side.

    Returns that left-hand side and the row scale the caller still owes what
    the second matmul produces. Every weight format shares this; what
    differs is what the halves owe before the activation and whether the
    result goes back through a quantization on the way out.

    act_scale is the activation row scale a quantized activation owes its
    accumulator, and is None where the rows were never quantized. w1s is the
    per-channel weight scale the eight-bit and integer paths still owe their
    halves, which the four-bit path has already folded into acc1 and passes
    as None. requantize is False for the formats whose second matmul takes
    bf16 rows directly; those return a row scale of None, because the
    accumulator they produce is already the true value.
    """

    def half(lo, hi):
        """One post-scale half of the accumulator, columns [lo, hi)."""
        scaled = acc1[:, lo:hi]
        if act_scale is not None:
            scaled = scaled * act_scale
        if w1s is not None:
            scaled = scaled * w1s[:, lo:hi]
        return scaled

    mid_chunks, amax = [], None
    for c0 in range(0, inter, QCHUNK):
        c1 = min(c0 + QCHUNK, inter)
        gate = half(c0, c1)
        up = half(inter + c0, inter + c1)
        # The gate and up biases land on the POST-SCALE halves, before the
        # activation: the clamped form's clip is defined on the true
        # activation value, so a bias added to a raw accumulator would clip
        # against the wrong scale. w1b is the [1, 2 * inter] gate|up row.
        if w1b is not None:
            gate = gate + w1b[:, c0:c1]
            up = up + w1b[:, inter + c0:inter + c1]
        chunk = _apply_act(gate, up, act_fn).astype(jnp.bfloat16)
        mid_chunks.append(chunk)
        if requantize:
            chunk_amax = _absmax_rows(chunk)
            amax = chunk_amax if amax is None else jnp.maximum(
                amax, chunk_amax)
    if not requantize:
        return jnp.concatenate(mid_chunks, axis=-1), None
    mid_scale, mid_sinv, usable = _row_scale(amax)
    sinv_bf16 = mid_sinv.astype(jnp.bfloat16)
    mid_q = jnp.concatenate(
        [_quant_apply(chunk, sinv_bf16, usable) for chunk in mid_chunks],
        axis=-1)
    return mid_q, mid_scale


def _chunked_dot(lhs, rhs_chunk, n_chunks, kb):
    """Contract lhs against a weight handed over one contraction chunk at a
    time, summing the chunks' partial products.

    Four-bit and integer weights both reach the matrix unit through a
    widening, and a whole widened slab does not fit beside the buffered
    ones, so both contract in chunks. rhs_chunk(b) returns contraction chunk
    b already in an element type the matrix unit takes.
    """
    acc = None
    for b in range(n_chunks):
        part = lax.dot_general(lhs[:, b * kb:(b + 1) * kb],
                               rhs_chunk(b),
                               DN,
                               preferred_element_type=jnp.float32)
        acc = part if acc is None else acc + part
    return acc


def expert_ffn_fp8(act_q, act_scale, w1, w2, w1s, act_fn="silu", w1b=None):
    """One expert's FFN on eight-bit weights: the gate/up matmul, the
    activation, the intermediate requantization and the down matmul.

    act_q fp8 [m, k] and act_scale f32 [m, 1] are the row-quantized
    activation rows. Returns the down matmul's accumulator and the
    intermediate row scale the caller still has to apply to it.
    """
    inter = w1.shape[-1] // 2
    acc1 = lax.dot_general(act_q, w1, DN, preferred_element_type=jnp.float32)
    mid_q, mid_scale = _intermediate_rows(acc1,
                                          act_scale,
                                          inter,
                                          w1s=w1s,
                                          act_fn=act_fn,
                                          w1b=w1b)
    acc2 = lax.dot_general(mid_q, w2, DN, preferred_element_type=jnp.float32)
    return acc2, mid_scale


def expert_ffn_blockscale(act_q,
                          act_scale,
                          w1_block,
                          w2_block,
                          w1s_blocks,
                          w2s_blocks,
                          *,
                          qb=QB4,
                          act_fn="silu",
                          w1b=None):
    """One expert's FFN on block-scaled weights: the same body as
    expert_ffn_fp8 with each matmul summed over contraction blocks carrying
    their own scales, w1s_blocks [nb1, 2*inter] and w2s_blocks [nb2, hidden].

    Each contraction block carries its own scale, on blocks already widened
    to fp8: w1_block(b) and w2_block(b) return weight k-block b in fp8, and
    the widening is the CALLER's. It has to be, because it is a Mosaic
    bitcast on a VMEM reference, which has no host form and so cannot be
    shared with the eight-bit body the way the rest of this function is.
    Nothing below depends on what the block was stored as; every other line
    here runs on either, and a block-scaled integer-four form would reuse
    this body verbatim with a different bitcast in the reader.
    """
    n_blocks1, n_blocks2 = w1s_blocks.shape[0], w2s_blocks.shape[0]
    inter = w1s_blocks.shape[-1] // 2
    acc1 = None
    for b in range(n_blocks1):
        part = lax.dot_general(act_q[:, b * qb:(b + 1) * qb],
                               w1_block(b),
                               DN,
                               preferred_element_type=jnp.float32)
        part = part * w1s_blocks[b][None, :]
        acc1 = part if acc1 is None else acc1 + part
    # The block scales are already inside acc1, so the halves this builds
    # are the true activation values and owe no further weight scale.
    mid_q, mid_scale = _intermediate_rows(acc1,
                                          act_scale,
                                          inter,
                                          w1s=None,
                                          act_fn=act_fn,
                                          w1b=w1b)
    acc2 = None
    for b in range(n_blocks2):
        part = lax.dot_general(mid_q[:, b * qb:(b + 1) * qb],
                               w2_block(b),
                               DN,
                               preferred_element_type=jnp.float32)
        part = part * w2s_blocks[b][None, :]
        acc2 = part if acc2 is None else acc2 + part
    return acc2, mid_scale


def expert_ffn_bf16(act, w1, w2, act_fn="silu", w1b=None):
    """One expert's FFN on unquantized sixteen-bit weights.

    Nothing here is quantized and nothing is dequantized. The rows arrive as
    the caller produced them, the weights carry no scales, and the
    intermediate goes into the second matmul as the activation left it, so
    both accumulators are already the true values and there is no row scale
    to return: a model that did not ask for quantization gets none.

    Returns the down matmul's accumulator and None, where the quantized
    bodies return the intermediate row scale their caller still owes it.
    """
    inter = w1.shape[-1] // 2
    acc1 = lax.dot_general(act, w1, DN, preferred_element_type=jnp.float32)
    mid, _ = _intermediate_rows(acc1,
                                None,
                                inter,
                                w1s=None,
                                act_fn=act_fn,
                                w1b=w1b,
                                requantize=False)
    acc2 = lax.dot_general(mid, w2, DN, preferred_element_type=jnp.float32)
    return acc2, None


def expert_ffn_int8(act,
                    w1_chunk,
                    w2_chunk,
                    w1s,
                    *,
                    n_chunks1,
                    n_chunks2,
                    kb=host.WIDEN_KCHUNK,
                    act_fn="silu",
                    w1b=None):
    """One expert's FFN on eight-bit INTEGER weights with per-channel scales.

    TPU v7's matrix unit has no integer input type (a generation with
    integer input could add a native int8 body as another weight form), so an
    integer weight cannot be contracted as it is stored: w1_chunk(b) and
    w2_chunk(b) return contraction chunk b already widened to bf16, the way
    the four-bit body's callers widen a scale block to fp8. Widening a whole
    slab would need a second slab beside the buffered ones, which is why
    both matmuls contract in chunks.

    Because the widening is to bf16, the rows are contracted unquantized and
    the intermediate stays bf16 too. The scales are per output channel, so
    they apply after the contraction, exactly where the eight-bit float path
    applies its own: w1s [1, 2 * inter] here, and the caller applies the
    second matmul's own per-channel scale to what this returns.

    Returns the down matmul's accumulator and None, for the same reason
    expert_ffn_bf16 does.
    """
    inter = w1s.shape[-1] // 2
    acc1 = _chunked_dot(act, w1_chunk, n_chunks1, kb)
    mid, _ = _intermediate_rows(acc1,
                                None,
                                inter,
                                w1s=w1s,
                                act_fn=act_fn,
                                w1b=w1b,
                                requantize=False)
    acc2 = _chunked_dot(mid, w2_chunk, n_chunks2, kb)
    return acc2, None


def _all_pairs_barrier(ep):
    """Barrier over every shard: transport peers are not only neighbours."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(ep):
        pl.semaphore_signal(barrier_sem,
                            inc=1,
                            device_id=(jnp.int32(i), ),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, ep)

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _(second):
        for i in range(ep):
            pl.semaphore_signal(second,
                                inc=1,
                                device_id=(jnp.int32(i), ),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(second, ep)


def _slot_copy(src, vm, sems, slot):
    """A copy of `src` into buffer slot `slot`, on that slot's semaphore."""
    return pltpu.make_async_copy(src, vm.at[slot], sems.at[slot])


def _rows_wait(sem, ref, rows):
    """Block until `rows` rows' worth of DMAs on `sem` have landed."""
    pltpu.make_async_copy(ref.at[pl.ds(0, rows)], ref.at[pl.ds(0, rows)],
                          sem).wait()


def _and_nonempty(pred, rows):
    """`pred` and a nonzero length: a zero-length DMA must never issue."""
    return jnp.logical_and(pred, rows > 0)


def _tile_row_scales(window, row_base, tile_m):
    """One tile's activation row scales as the [tile_m, 1] column the FFN
    takes, out of the [sublanes, lanes] window its rows fall in.

    The slab holds one f32 per row in the dense lane-block layout the flat
    array already has, so row r is at sublane r / lanes, lane r % lanes.
    A tile's first row is row-block aligned but not lane-block aligned, so
    the tile straddles two sublanes: rotating the window's lanes by the
    first row's lane and selecting between each sublane and the one after
    it puts the tile's rows in order on one sublane, and the column the
    FFN wants is that sublane read the other way round.

    Every step here is a vector-unit operation on about a lane block of
    values, held beside two expert matmuls. The alternative is for the
    caller to build the column, which is a whole-slab retile in HBM.
    """
    lanes = HIDDEN_LANE_BLOCK
    resid = lax.rem(row_base, jnp.int32(lanes))
    rot = pltpu.roll(window, lanes - resid, 1)
    lane = lax.broadcasted_iota(jnp.int32, rot[:-1].shape, 1)
    rows = jnp.where(lane < lanes - resid, rot[:-1], rot[1:])
    return rows.reshape(-1)[:tile_m][:, None]


def _build_fused_ep_moe_kernel(*,
                               g_local,
                               capacity,
                               hidden,
                               inter,
                               ep,
                               weight_format=host.WeightFormat.FP8,
                               rhs_qb=QB4,
                               ragged_rows_alloc=None,
                               act_fn="silu",
                               has_w1_bias=False,
                               has_w2_bias=False):
    """Build the pallas_call and the function that invokes it."""
    if act_fn not in ACT_FNS:
        raise NotImplementedError(
            f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}")
    # The kernel fetches lhs rows by index from the ungathered token buffer
    # [tokens, lane blocks, 128], using the token-gather table, and ships
    # results over an all-to-all carrying unweighted rows; it returns
    # (arrival rows, arrival scales f32). The element type of the staging and
    # of the wire is the weight format's, not a literal here.
    form = host.weight_form(weight_format)
    rhs_packed4 = weight_format == host.WeightFormat.FP4
    assert 0 < WEIGHT_PREFETCH_DISTANCE < NBUF, (
        f"weight prefetch distance {WEIGHT_PREFETCH_DISTANCE} must satisfy "
        f"0 < distance < NBUF={NBUF}, so that distance + 1 consecutive "
        "experts occupy distinct weight slots")
    # Everything from here to the ragged-stride check below tests a value
    # that came IN from the caller -- a model shape, a mesh width, a serving
    # config -- so each one refuses rather than asserts. An assert is the
    # right tool for this function's own arithmetic, like the prefetch
    # distance above, and the wrong tool for the caller's operands: `python
    # -O` and PYTHONOPTIMIZE are ordinary things to find in a container
    # image, and under them an assert lets the illegal configuration build
    # and be cached.
    if capacity % ROWBLK:
        raise ValueError(
            f"tile height {capacity} is not a whole number of {ROWBLK}-row "
            "blocks, which is the unit every transport here moves")
    if rhs_packed4:
        # Tested before the modulo below rather than after it: every other
        # guard in this span was converted from an assert so that a caller's
        # operand is refused by name, and a block size of zero divided by
        # the operand first and came back as a ZeroDivisionError.
        if rhs_qb < 1:
            raise ValueError(
                f"the four-bit block size is {rhs_qb}; it is the number of "
                "contraction rows one weight scale covers and there is no "
                "block smaller than one row")
        if hidden % rhs_qb or inter % rhs_qb:
            raise ValueError(
                f"the four-bit block size {rhs_qb} has to divide both "
                f"hidden={hidden} and inter={inter}; the kernel blocks BOTH "
                "matmuls at one block size")
        # Packed rows per k-block must land on the u32 sublane tile.
        if rhs_qb % (host.U32_SUBLANE_TILE * PACK4):
            raise ValueError(
                f"the four-bit block size {rhs_qb} is not a whole number of "
                f"the packed-weight row tile "
                f"({host.U32_SUBLANE_TILE * PACK4} rows): {PACK4} four-bit "
                f"values pack into a 32-bit word and those words tile to "
                f"{host.U32_SUBLANE_TILE} sublanes")
    if weight_format == host.WeightFormat.INT8:
        # The widening chunk has to tile both contractions exactly.
        if hidden % host.WIDEN_KCHUNK or inter % host.WIDEN_KCHUNK:
            raise ValueError(
                f"integer weights widen one {host.WIDEN_KCHUNK}-row "
                f"contraction chunk at a time, which has to divide both "
                f"hidden={hidden} and inter={inter}")
    # True-length remote pushes: out_vm, contrib and recv take the per-row
    # [rows, lane blocks, 128] geometry, which row-granular offsets require.
    if (hidden < HIDDEN_LANE_BLOCK or hidden % HIDDEN_LANE_BLOCK
            or hidden > host.HIDDEN_MAX_BLOCKS * HIDDEN_LANE_BLOCK):
        raise ValueError(
            f"hidden {hidden} is not between one and "
            f"{host.HIDDEN_MAX_BLOCKS} whole {HIDDEN_LANE_BLOCK}-lane "
            "blocks, which the per-row transport geometry requires")
    # The intermediate axis had no floor at all, so an expert whose FFN has
    # no intermediate channels built a program and cached it: the estimate
    # for it clears the budget comfortably and every divisibility check
    # passes, because zero is a whole number of everything. The floor is one
    # lane block, the same rule and the same reason as the hidden axis
    # above -- an intermediate narrower than the vector unit's lane count is
    # not an FFN, and the diagnostic an operator eventually receives names a
    # reduction or a reshape rather than the weight that was empty.
    if inter < HIDDEN_LANE_BLOCK:
        raise ValueError(
            f"inter {inter} is narrower than one {HIDDEN_LANE_BLOCK}-lane "
            "block; an expert whose FFN has no intermediate channels to "
            "speak of is not a layer this kernel can serve")
    lane_blocks = host.row_lane_blocks(hidden)
    scale_lanes = host.scale_mirror_lanes(hidden)
    has_scales = form.has_scales
    has_act_scale = form.quantized_activations
    est = host.vmem_estimate_bytes(g_local,
                                   capacity,
                                   hidden,
                                   inter,
                                   nbuf=NBUF,
                                   weight_format=weight_format,
                                   rhs_qb=rhs_qb,
                                   has_w1_bias=has_w1_bias,
                                   has_w2_bias=has_w2_bias)
    limit = host.vmem_limit()
    if rhs_packed4:
        # Asked here, beside the other device reads, because the packed
        # four-bit layout is the one place a written-down device fact and a
        # queried one have to agree: the block-size check above uses the
        # constant and the buffers the accounting just sized used the
        # record.
        host.check_u32_sublane_tile()
    # The only thing standing between an over-budget model and a kernel that
    # will not fit. It must not be strippable.
    if est > limit:
        raise ValueError(
            f"the kernel's VMEM buffers need {est/2**20:.1f}MiB, over the "
            f"{limit/2**20:.1f}MiB budget, for {g_local} local experts of "
            f"{hidden}x{inter} at NBUF={NBUF} capacity={capacity}")
    # ragged_rows_alloc must cover every tile READ as well as every commit
    # -- the shard's total rows plus tile_m, aligned to tile_m -- because a
    # tail tile reads a full window.
    if (ragged_rows_alloc is None or ragged_rows_alloc % ROWBLK
            or ragged_rows_alloc % capacity):
        raise ValueError(
            f"ragged_rows_alloc {ragged_rows_alloc} must be a whole number "
            f"of both {ROWBLK}-row blocks and {capacity}-row tiles, because "
            "a tail tile reads a full window past the shard's last row")
    tile_m = capacity  # the tile height IS the capacity
    tile_blocks = tile_m // ROWBLK
    ls_window_rows = host.act_scale_window_rows(tile_m)

    def kernel(*refs):
        it = iter(refs)
        # Block-unit transport tables, from shard_transport_tables_in_blocks.
        # A push reads its source at the same offset the commit wrote it to,
        # so contrib_off_sm is also the push source and arrives once.
        (commit_start_sm, commit_len_sm, contrib_off_sm, push_len_sm,
         push_dst_sm) = (next(it) for _ in range(N_TRANSPORT_TABLES))
        # Row-unit push tables, from shard_push_tables_in_rows.
        (true_rows_sm, recv_row_off_sm) = (next(it)
                                           for _ in range(N_PUSH_TABLES))
        token_gather_sm = next(it)
        # (rows, base) i32 [G] row-unit tables from shard_expert_slabs.
        (expert_rows_sm, expert_base_sm) = (next(it)
                                            for _ in range(N_SLAB_TABLES))
        # visit_sm[visit_i] is the real expert at compacted step visit_i.
        visit_sm = next(it)
        # The five counts, each still spread over the expert-parallel axis.
        counts_sm = next(it)

        def count(row):
            """Close one count's reduction on the scalar core.

            The host stopped a step early and left `ep` lanes to add. `ep`
            is a build-time constant, so this is a fixed run of scalar
            loads and adds: no loop, no dynamic bound, and nothing that
            reaches vector memory.
            """
            total = counts_sm[row, 0]
            for d in range(1, ep):
                total = total + counts_sm[row, d]
            return total

        # Consumed at the head and all through the visit loop, so it is
        # closed once here. The transport totals are closed in the drain
        # instead, where they are read, because that is the end of the
        # kernel and their arithmetic has everything before it to hide in.
        n_visit = count(host.COUNT_VISITS)
        # lhs_hbm is the ungathered token buffer [tokens, lane blocks, 128].
        # The wire never ships the topk weight, so it is not an operand.
        lhs_hbm = next(it)
        # The activation row scale exists only where the rows were quantized.
        ls_hbm = next(it) if has_act_scale else None
        w1_hbm, w2_hbm = (next(it) for _ in range(2))
        # The weight scale tables exist only where the weights carry scales.
        w1s_hbm = next(it) if has_scales else None
        w2s_hbm = next(it) if has_scales else None
        # Optional per-expert bias tables, one row per local expert, in the
        # same [G, N] layout as the per-channel weight scales beside them.
        w1b_hbm = next(it) if has_w1_bias else None
        w2b_hbm = next(it) if has_w2_bias else None
        if rhs_packed4:
            # Packed-u32 weight stream: the four-bit [G, K, N] refs viewed as
            # [G, K//8, N] uint32, so the DMA moves half the bytes. This is
            # the ref-level view, and it is a PRIVATE jax method -- Pallas
            # exports pltpu.bitcast for the value-level widening below but
            # has no public equivalent for a ref. It carries no deprecation
            # entry, so if it moves it fails as a bare AttributeError from
            # inside a kernel body. Say what happened instead.
            if not hasattr(w1_hbm, "bitcast"):
                raise NotImplementedError(
                    f"the four-bit weight stream views a memory ref as "
                    f"packed 32-bit words through the private "
                    f"{type(w1_hbm).__name__}.bitcast, which the installed "
                    f"jax {jax.__version__} no longer carries; this kernel "
                    f"was built against jax {_REF_BITCAST_JAX_VERSION}. "
                    "There is no public Pallas equivalent for a ref-level "
                    "bitcast, so this needs a port rather than a rename.")
            w1_hbm = w1_hbm.bitcast(jnp.uint32)
            w2_hbm = w2_hbm.bitcast(jnp.uint32)
        recv_hbm = next(it)
        rscl_hbm = next(it)
        contrib_hbm = next(it)
        cscl_hbm = next(it)
        # In vmem_scratch_arrays order, which is what the scratch list is
        # built from: staging, weights, scale tables where they exist, the
        # bias tables each build asked for, the activation row scale where
        # it exists, then the wire buffer and its scale mirror.
        (lhs_vm, w1_vm, w2_vm) = (next(it) for _ in range(3))
        w1s_vm = next(it) if has_scales else None
        w2s_vm = next(it) if has_scales else None
        w1b_vm = next(it) if has_w1_bias else None
        w2b_vm = next(it) if has_w2_bias else None
        ls_vm = next(it) if has_act_scale else None
        out_vm = next(it)
        oscl_vm = next(it)  # [parities, tile blocks, ROWBLK, scale lanes] f32
        lhs_sems, w1_sems, w2_sems, cp_sem = (next(it) for _ in range(4))
        # One commit sem per out_vm parity: with a shared sem the other
        # parity's bytes could satisfy a wait, and order is not promised.
        commit_sems, send_sem, recv_sem = (next(it) for _ in range(3))
        # Scale-mirror DMAs need their own sems: waits are per-buffer.
        (commit_scl_sems, send_scl_sem, recv_scl_sem) = (next(it)
                                                         for _ in range(3))
        # Never cp_sem: its start+wait users must not consume these.
        mehop_sem = next(it)
        mehop_scl_sem = next(it)

        me = lax.axis_index(AXIS)

        def sync(src, dst):
            """Copy `src` into `dst` and wait for it before returning."""
            copy = pltpu.make_async_copy(src, dst, cp_sem)
            copy.start()
            copy.wait()

        def w1_copy(expert, slot):
            """The gate/up weight slab of `expert`, into weight slot `slot`."""
            return _slot_copy(w1_hbm.at[expert], w1_vm, w1_sems, slot)

        def w2_copy(expert, slot):
            """The down weight slab of `expert`, into weight slot `slot`."""
            return _slot_copy(w2_hbm.at[expert], w2_vm, w2_sems, slot)

        def prologue():
            """Land the resident tables and start the first weight refills."""
            if has_scales:
                sync(w1s_hbm, w1s_vm)
                sync(w2s_hbm, w2s_vm)
            if has_w1_bias:
                sync(w1b_hbm, w1b_vm)
            if has_w2_bias:
                sync(w2b_hbm, w2b_vm)
            # Guarded on b < n_visit, matching the head waits. The range is
            # clamped to the visit-list length: at one local expert,
            # visit_sm[1] would be a statically out-of-bounds read before
            # the predicate could save it.
            for b in range(min(WEIGHT_PREFETCH_DISTANCE, g_local)):

                @pl.when(jnp.int32(b) < n_visit)
                def _(b=b):
                    w1_copy(visit_sm[b], b).start(priority=WEIGHT_DMA_PRIORITY)
                    w2_copy(visit_sm[b], b).start(priority=WEIGHT_DMA_PRIORITY)

        prologue()
        _all_pairs_barrier(ep)

        def _fp4_block_readers(slot):
            """(w1_block, w2_block) fp4 block readers at weight slot `slot`.

            The one genuinely fp4 place in the kernel: everything around it
            is four-bit generic, and these two lines are where the packed
            words are read as fp4 (e2m1) rather than as some other four-bit
            element type.
            """
            # Each widens one k-block: u32 [qb/8, N] -> fp4 [qb, N] -> fp8.
            packed_rows = rhs_qb // PACK4

            def w1_block(b):
                return pltpu.bitcast(
                    w1_vm[slot, pl.ds(b * packed_rows, packed_rows), :],
                    FP4).astype(FP8)

            def w2_block(b):
                return pltpu.bitcast(
                    w2_vm[slot, pl.ds(b * packed_rows, packed_rows), :],
                    FP4).astype(FP8)

            return w1_block, w2_block

        def _int8_chunk_readers(slot):
            """(w1_chunk, w2_chunk) integer chunk readers at slot `slot`.

            Each widens one contraction chunk to bf16, which is what the
            matrix unit takes: TPU v7 has no integer input
            type, so the widening is not an optimization but the only way
            an integer weight reaches a matmul at all.
            """

            def w1_chunk(b):
                return w1_vm[
                    slot,
                    pl.ds(b * host.WIDEN_KCHUNK, host.WIDEN_KCHUNK), :].astype(
                        host.BF16)

            def w2_chunk(b):
                return w2_vm[
                    slot,
                    pl.ds(b * host.WIDEN_KCHUNK, host.WIDEN_KCHUNK), :].astype(
                        host.BF16)

            return w1_chunk, w2_chunk

        # ---- ragged tile machinery ----
        # One-tile lookahead, double-buffered on the global tile parity:
        # a tile's stream rides lhs_sems[parity], waited once at its head.
        def lhs_issue_tile(row_base, live_blocks, slot):
            """Issue one [tile_m] tile's input stream into slot `slot`."""

            def issue_block(i, _):
                """Fetch the eight token rows of block i by their numbers."""
                for r in range(ROWBLK):
                    token = token_gather_sm[row_base + i * ROWBLK + r]
                    pltpu.make_async_copy(lhs_hbm.at[token],
                                          lhs_vm.at[slot, i * ROWBLK + r],
                                          lhs_sems.at[slot]).start()
                return _

            lax.fori_loop(0, live_blocks, issue_block, jnp.int32(0))
            if has_act_scale:
                # ls rides along on the same sem. The slab is the dense
                # lane-block view of one f32 per row, so a tile's window is
                # the sublanes its rows fall in, not a row range.
                pltpu.make_async_copy(
                    ls_hbm.at[pl.ds(row_base // HIDDEN_LANE_BLOCK,
                                    ls_window_rows)], ls_vm.at[slot],
                    lhs_sems.at[slot]).start()

        def lhs_ready_tile(live_rows, slot):
            """Wait the token rows and the scale window of one tile."""
            # Sems count bytes, so the split wait is exact under any landing
            # order.
            pltpu.make_async_copy(lhs_vm.at[slot, pl.ds(0, live_rows)],
                                  lhs_vm.at[slot, pl.ds(0, live_rows)],
                                  lhs_sems.at[slot]).wait()
            if has_act_scale:
                pltpu.make_async_copy(ls_vm.at[slot], ls_vm.at[slot],
                                      lhs_sems.at[slot]).wait()

        def wait_commits(parity, live_blocks):
            """Wait one tile's commits on `parity`: data rows, then scales."""
            _rows_wait(commit_sems.at[parity], contrib_hbm,
                       live_blocks * ROWBLK)
            _rows_wait(commit_scl_sems.at[parity], cscl_hbm, live_blocks)

        def expert_tiles(e, visit_i, carry):
            """One expert step: a fori_loop over its [tile_m, H] tiles."""
            # carry = (tile_count, pending0, pending1): the global tile
            # counter plus the last committed block count per out_vm parity.
            # The weight slot indexes a contiguous counter so DISTANCE + 1 of
            # them occupy distinct slots; the DMA base stays the real expert.
            slot = lax.rem(visit_i, jnp.int32(NBUF))
            w1_copy(e, slot).wait()
            w2_copy(e, slot).wait()

            # Refill the slot for the expert DISTANCE ahead: the previous
            # expert last read it, and DISTANCE < NBUF keeps it off both live
            # readers, so no wait is needed.
            @pl.when(visit_i + WEIGHT_PREFETCH_DISTANCE < n_visit)
            def _():
                refill_slot = lax.rem(visit_i + WEIGHT_PREFETCH_DISTANCE,
                                      jnp.int32(NBUF))
                ahead = visit_sm[visit_i + WEIGHT_PREFETCH_DISTANCE]
                # Weight refills take their own DMA priority to stay off the
                # in-order token-gather queue.
                w1_copy(ahead, refill_slot).start(priority=WEIGHT_DMA_PRIORITY)
                w2_copy(ahead, refill_slot).start(priority=WEIGHT_DMA_PRIORITY)

            rows = expert_rows_sm[e]
            slab_base = expert_base_sm[e]
            n_tiles = -(-rows // tile_m)

            def commit_tile(parity, tile_block_base, live_blocks):
                """Commit the tile's intersection with each (expert, dest) run."""
                # The runs tile the expert slab contiguously, so the per-dest
                # lengths sum to exactly live_blocks. Data in rows, scales in
                # blocks.
                for d in range(ep):
                    run_start = commit_start_sm[e, d]
                    run_len = commit_len_sm[e, d]
                    lo = jnp.maximum(run_start, tile_block_base)
                    hi = jnp.minimum(run_start + run_len,
                                     tile_block_base + live_blocks)
                    overlap = jnp.maximum(hi - lo, 0)
                    # An empty intersection leaves lo unclamped, so the
                    # offsets below can point out of range; the copies are
                    # predicated out rather than issued at zero length. The
                    # commit waits count bytes, and a skipped copy
                    # contributes zero bytes either way.
                    lo = jnp.minimum(lo, hi)
                    src_block = lo - tile_block_base
                    dst_block = contrib_off_sm[e, d] + (lo - run_start)

                    @pl.when(overlap > 0)
                    def _(src_block=src_block,
                          dst_block=dst_block,
                          overlap=overlap,
                          parity=parity):
                        pltpu.make_async_copy(
                            out_vm.at[parity,
                                      pl.ds(src_block * ROWBLK, overlap *
                                            ROWBLK)],
                            contrib_hbm.at[pl.ds(dst_block * ROWBLK,
                                                 overlap * ROWBLK)],
                            commit_sems.at[parity]).start()
                        pltpu.make_async_copy(
                            oscl_vm.at[parity,
                                       pl.ds(src_block, overlap)],
                            cscl_hbm.at[pl.ds(dst_block, overlap)],
                            commit_scl_sems.at[parity]).start()

            def tile_body(t, carried):
                """Compute tile t, stage it for the wire and commit it."""
                tile_count, pending0, pending1 = carried
                row_base = slab_base + t * tile_m
                live_rows = jnp.minimum(rows - t * tile_m, tile_m)
                live_blocks = live_rows // ROWBLK
                tile_block_base = t * tile_blocks  # expert-local block base
                parity = lax.rem(tile_count, jnp.int32(OUT_PARITIES))

                # out_vm[parity] reuse guard. Predicated on pending > 0 rather
                # than on the tile count, because drain_commits zeroes them.
                pending = jnp.where(parity == 0, pending0, pending1)

                @pl.when(pending > 0)
                def _():
                    wait_commits(parity, pending)

                # Wait this tile's stream, then issue tile t+1's into the other
                # slot before compute, so the fetch runs under the MXU window.
                lhs_ready_tile(live_rows, parity)
                other_parity = jnp.int32(1) - parity

                @pl.when(t + 1 < n_tiles)
                def _():
                    next_rows = jnp.minimum(rows - (t + 1) * tile_m, tile_m)
                    lhs_issue_tile(row_base + tile_m, next_rows // ROWBLK,
                                   other_parity)

                # A tile always computes its full tile_m rows; the commits
                # below span only its true rows.
                act_rows = lhs_vm[parity].reshape(tile_m, hidden)
                act_scales = (_tile_row_scales(ls_vm[parity], row_base, tile_m)
                              if has_act_scale else None)
                w1b_row = w1b_vm[pl.ds(e, 1), :] if has_w1_bias else None
                if rhs_packed4:
                    # The block scales apply inside expert_ffn_blockscale,
                    # so the epilogues below must not apply w2s again.
                    w1s_blocks = w1s_vm[e]  # [nb1, 2*inter] f32
                    w2s_blocks = w2s_vm[e]  # [nb2, hidden] f32
                    w1_block, w2_block = _fp4_block_readers(slot)
                    acc2, mid_scale = expert_ffn_blockscale(act_rows,
                                                            act_scales,
                                                            w1_block,
                                                            w2_block,
                                                            w1s_blocks,
                                                            w2s_blocks,
                                                            qb=rhs_qb,
                                                            act_fn=act_fn,
                                                            w1b=w1b_row)
                elif weight_format == host.WeightFormat.INT8:
                    w1_chunk, w2_chunk = _int8_chunk_readers(slot)
                    acc2, mid_scale = expert_ffn_int8(
                        act_rows,
                        w1_chunk,
                        w2_chunk,
                        w1s_vm[pl.ds(e, 1), :],
                        n_chunks1=hidden // host.WIDEN_KCHUNK,
                        n_chunks2=inter // host.WIDEN_KCHUNK,
                        act_fn=act_fn,
                        w1b=w1b_row)
                elif weight_format == host.WeightFormat.BF16:
                    acc2, mid_scale = expert_ffn_bf16(act_rows,
                                                      w1_vm[slot],
                                                      w2_vm[slot],
                                                      act_fn=act_fn,
                                                      w1b=w1b_row)
                else:
                    acc2, mid_scale = expert_ffn_fp8(act_rows,
                                                     act_scales,
                                                     w1_vm[slot],
                                                     w2_vm[slot],
                                                     w1s_vm[pl.ds(e, 1), :],
                                                     act_fn=act_fn,
                                                     w1b=w1b_row)
                # The destination applies the router weight, not this. Four-bit
                # weights carry w2s inside the block sums; the formats whose
                # second matmul took bf16 rows return no row scale, so the only
                # thing left to apply is the per-channel weight scale where the
                # format has one. The epilogue itself is a module-level
                # function, so the placement of each of these -- and the down
                # bias's add-once property in particular -- has a host-side
                # witness rather than only a device-marked one.
                wire_rows, wire_scales = wire_row_and_scale(
                    acc2,
                    mid_scale,
                    w2s=(w2s_vm[pl.ds(e, 1), :] if
                         (has_scales and not rhs_packed4) else None),
                    w2b=w2b_vm[pl.ds(e, 1), :] if has_w2_bias else None,
                    wire_dtype=form.wire_dtype,
                    tile_m=tile_m)
                wire_rows_staged = wire_rows.reshape(tile_m, lane_blocks,
                                                     HIDDEN_LANE_BLOCK)
                wire_scales_staged = jnp.broadcast_to(
                    wire_scales,
                    (tile_m, scale_lanes)).reshape(tile_blocks, ROWBLK,
                                                   scale_lanes)

                def store_parity(static_parity):
                    """Stage the tile's rows and scales in one out_vm slot."""
                    out_vm[static_parity] = wire_rows_staged
                    oscl_vm[static_parity] = wire_scales_staged

                # Store slots must be static, so these branches stay static.
                @pl.when(parity == 0)
                def _():
                    store_parity(0)

                @pl.when(parity == 1)
                def _():
                    store_parity(1)

                commit_tile(parity, tile_block_base, live_blocks)

                return (tile_count + 1,
                        jnp.where(parity == 0, live_blocks, pending0),
                        jnp.where(parity == 1, live_blocks, pending1))

            carry = lax.fori_loop(0, n_tiles, tile_body, carry)
            # Cross-expert lookahead: issue the next expert's tile 0 here, so
            # its head wait finds the stream in flight. An empty next expert
            # issues nothing and its own tail issues the one after.
            tiles_done = carry[0]

            @pl.when(visit_i + 1 < n_visit)
            def _():
                nxt = visit_sm[visit_i + 1]
                lhs_issue_tile(
                    expert_base_sm[nxt],
                    jnp.minimum(expert_rows_sm[nxt], tile_m) // ROWBLK,
                    lax.rem(tiles_done, jnp.int32(OUT_PARITIES)))

            return carry

        def drain_commits(carried):
            """Drain both parities' pending commits, then zero the counts."""
            # The tile reuse guards test pending > 0, so nothing double-waits.
            tile_count, pending0, pending1 = carried

            @pl.when(pending0 > 0)
            def _():
                wait_commits(0, pending0)

            @pl.when(pending1 > 0)
            def _():
                wait_commits(1, pending1)

            return (tile_count, jnp.int32(0), jnp.int32(0))

        @pl.when(n_visit > 0)
        def _():
            first_expert = visit_sm[0]
            lhs_issue_tile(
                expert_base_sm[first_expert],
                jnp.minimum(expert_rows_sm[first_expert], tile_m) // ROWBLK,
                jnp.int32(0))

        def push_expert(e):
            """Push expert e's remote regions and hop its own-dest region."""
            # Remote pushes go per (expert, dest) at true length; run starts
            # stay aligned, so recv positions are unchanged. Each is predicated
            # on its own length: a zero-length REMOTE DMA must never issue.
            for d in range(ep):
                region_blocks = push_len_sm[e, d]
                src_block = contrib_off_sm[e, d]
                dst_block = push_dst_sm[e, d]
                is_me = jnp.int32(d) == me

                @pl.when(_and_nonempty(jnp.logical_not(is_me), region_blocks))
                def _():
                    # One expert per push, so the region IS expert e's run.
                    true_rows = true_rows_sm[e, d]

                    def push_run():
                        """The rows themselves, at their true length."""
                        pltpu.make_async_remote_copy(
                            src_ref=contrib_hbm.at[pl.ds(
                                contrib_off_sm[e, d] * ROWBLK, true_rows)],
                            dst_ref=recv_hbm.at[pl.ds(recv_row_off_sm[e, d],
                                                      true_rows)],
                            send_sem=send_sem,
                            recv_sem=recv_sem,
                            device_id=(jnp.int32(d), ),
                            device_id_type=pl.DeviceIdType.MESH).start()

                    # A nonempty region can hold an empty (e, d) run.
                    pl.when(true_rows > 0)(push_run)
                    # Scale mirror, same block offsets.
                    pltpu.make_async_remote_copy(
                        src_ref=cscl_hbm.at[pl.ds(src_block, region_blocks)],
                        dst_ref=rscl_hbm.at[pl.ds(dst_block, region_blocks)],
                        send_sem=send_scl_sem,
                        recv_sem=recv_scl_sem,
                        device_id=(jnp.int32(d), ),
                        device_id_type=pl.DeviceIdType.MESH).start()

                # Safe to defer: recv is read only after the drain.
                @pl.when(_and_nonempty(is_me, region_blocks))
                def _():
                    pltpu.make_async_copy(
                        contrib_hbm.at[pl.ds(src_block * ROWBLK,
                                             region_blocks * ROWBLK)],
                        recv_hbm.at[pl.ds(dst_block * ROWBLK,
                                          region_blocks * ROWBLK)],
                        mehop_sem).start()
                    pltpu.make_async_copy(
                        cscl_hbm.at[pl.ds(src_block, region_blocks)],
                        rscl_hbm.at[pl.ds(dst_block, region_blocks)],
                        mehop_scl_sem).start()

        def visit_step(visit_i, carry):
            """One step of the visit list: compute, drain, push."""
            e = visit_sm[visit_i]
            carry = expert_tiles(e, visit_i, carry)
            carry = drain_commits(carry)
            push_expert(e)
            return carry

        carry0 = (jnp.int32(0), jnp.int32(0), jnp.int32(0))
        lax.fori_loop(0, n_visit, visit_step, carry0)

        def drain_transport():
            """Consume the pushes and the commits the per-tile waits left."""
            # The per-tile head waits consume every commit but the last one
            # per parity; the drains below consume those.
            _rows_wait(send_sem, contrib_hbm, count(host.COUNT_SEND_ROWS))
            _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
            # These two arrive in rows; the scale mirror moves one entry
            # per block, so they are divided here, on one reduced scalar
            # each, by a build-time constant.
            _rows_wait(send_scl_sem, cscl_hbm,
                       count(host.COUNT_SEND_ALIGNED_ROWS) // ROWBLK)
            _rows_wait(recv_scl_sem, rscl_hbm,
                       count(host.COUNT_RECV_ALIGNED_ROWS) // ROWBLK)
            # Deferred own-destination drain: the total sums my own-dest
            # region lengths, and a skipped pair owes exactly zero rows.
            self_blocks = push_len_sm[0, me]
            for g in range(1, g_local):
                self_blocks = self_blocks + push_len_sm[g, me]
            _rows_wait(mehop_sem, recv_hbm, self_blocks * ROWBLK)
            _rows_wait(mehop_scl_sem, rscl_hbm, self_blocks)

        drain_transport()
        _all_pairs_barrier(ep)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    # The VMEM buffers come from the same list the VMEM accounting sums, in
    # the order the kernel body unpacks them: lhs_vm, w1_vm, w2_vm, the scale
    # tables where the format has them, the bias tables each build asked for,
    # ls_vm where the activations are quantized, out_vm, oscl_vm.
    scratch = [
        pltpu.VMEM(shape, dtype) for _, shape, dtype in
        host.vmem_scratch_arrays(g_local,
                                 capacity,
                                 hidden,
                                 inter,
                                 nbuf=NBUF,
                                 weight_format=weight_format,
                                 rhs_qb=rhs_qb,
                                 has_w1_bias=has_w1_bias,
                                 has_w2_bias=has_w2_bias)
    ] + [
        # lhs_sems. Parity-deep, not NBUF-deep: every index into it, and into
        # the lhs_vm and ls_vm buffers it guards, is an out-parity.
        pltpu.SemaphoreType.DMA((OUT_PARITIES, )),
        pltpu.SemaphoreType.DMA((NBUF, )),  # w1_sems
        pltpu.SemaphoreType.DMA((NBUF, )),  # w2_sems
        pltpu.SemaphoreType.DMA,  # cp_sem
        pltpu.SemaphoreType.DMA((OUT_PARITIES, )),  # commit_sems
        pltpu.SemaphoreType.DMA,  # send_sem
        pltpu.SemaphoreType.DMA,  # recv_sem
        pltpu.SemaphoreType.DMA((OUT_PARITIES, )),  # commit_scl_sems
        pltpu.SemaphoreType.DMA,  # send_scl_sem
        pltpu.SemaphoreType.DMA,  # recv_scl_sem
        pltpu.SemaphoreType.DMA,  # mehop_sem
        pltpu.SemaphoreType.DMA,  # mehop_scl_sem
    ]
    # The staging buffer and the two weight slabs are always operands; the
    # activation row scale, the two weight scale tables and the two bias
    # tables are there only where the build has them.
    n_inputs = (N_STAGING_INPUTS + N_WEIGHT_SLABS + int(has_act_scale) +
                N_SCALE_TABLES * int(has_scales) + int(has_w1_bias) +
                int(has_w2_bias))

    def make_call(recv_rows):
        """The pallas_call for this build, at this arrival-buffer height."""
        # A caller value, so it refuses rather than asserts: under `python
        # -O` a non-multiple truncates in the two divisions below and the
        # arrival buffer comes out short by up to ROWBLK - 1 rows, while the
        # combine still gathers at positions reaching recv_rows - 1.
        if recv_rows % ROWBLK:
            raise ValueError(
                f"the arrival buffer height {recv_rows} is not a whole "
                f"number of {ROWBLK}-row blocks, which is the unit every "
                "transport into it moves")
        # contrib and cscl are sized for the no-drop worst case.
        contrib_rows = ragged_rows_alloc
        wire = form.wire_dtype
        out_shape = [
            jax.ShapeDtypeStruct((recv_rows, lane_blocks, HIDDEN_LANE_BLOCK),
                                 wire),
            jax.ShapeDtypeStruct((recv_rows // ROWBLK, ROWBLK, scale_lanes),
                                 jnp.float32),
            jax.ShapeDtypeStruct(
                (contrib_rows, lane_blocks, HIDDEN_LANE_BLOCK), wire),
            jax.ShapeDtypeStruct((contrib_rows // ROWBLK, ROWBLK, scale_lanes),
                                 jnp.float32),
        ]
        # Each suffix appears only when its feature is on, so a kernel that
        # fuses silu on eight-bit float weights with no biases keeps the name
        # it had before any of those operands existed, and every fingerprint
        # taken against it still matches.
        format_tag = {
            host.WeightFormat.FP8: "",
            host.WeightFormat.FP4: "_fp4w",
            host.WeightFormat.INT8: "_int8w",
            host.WeightFormat.BF16: "_bf16w",
        }[weight_format]
        bias_tag = (("_w13bias" if has_w1_bias else "") +
                    ("_w2bias" if has_w2_bias else ""))
        name = (f"fused_ep_moe_v2"
                f"{format_tag}"
                f"{'' if act_fn == 'silu' else '_' + act_fn}"
                f"{bias_tag}"
                f"_g{g_local}_c{capacity}_nb{NBUF}")
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=N_PREFETCH,
                in_specs=[hbm] * n_inputs,
                out_specs=[hbm] * len(out_shape),
                scratch_shapes=tuple(scratch),
                grid=()),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(
                collective_id=COLLECTIVE_ID,
                vmem_limit_bytes=host.vmem_limit(),
                # Bounds checks cost time and change no computed value.
                disable_bounds_checks=True),
            name=name,
        )

    def fn(tables,
           token_gather,
           expert_rows,
           expert_base,
           act_q_gathered,
           act_scales,
           w1,
           w2,
           w1s,
           w2s,
           w1b=None,
           w2b=None,
           *,
           recv_rows,
           visit,
           counts):
        """The served ragged transport form; returns arrivals and scales."""
        # tables = the block-unit transport tuple and the row-unit push
        # tuple, (rows, base) from shard_expert_slabs, act_scales the
        # activation scale slab in the dense lane-block view
        # host.act_scale_slab_rows names, one f32 per slab row.
        # act_scales, w1s and w2s are the operands the weight format decides
        # on: the activation row scale exists where the rows were quantized
        # and the scale tables where the weights carry scales. w1b
        # [G, 2 * inter] and w2b [G, hidden] are the optional expert biases.
        # Whether each is present is a build-time property, so a mismatch is
        # a caller bug rather than a shape error deeper in. These are caller
        # operands, so they refuse rather than assert, for the reason stated
        # at the top of the builder -- and the failure here is not symmetric.
        # Supplying weight scales to a build whose format has none leaves
        # scale_args empty below and the operand is silently DROPPED, so the
        # kernel runs its matmuls with no weight scale applied and returns a
        # plausible, uniformly wrong result; this is the only guard for that
        # direction. (A build that wants scales and gets None fails loudly on
        # None.astype, and a missing bias fails on the pallas_call operand
        # count.) This is also the check that catches "you named bf16 and
        # supplied scales".
        for operand, present, what in ((act_scales, has_act_scale,
                                        "activation row scale"),
                                       (w1s, has_scales, "gate/up weight "
                                        "scale table"),
                                       (w2s, has_scales, "down weight scale "
                                        "table"), (w1b, has_w1_bias,
                                                   "gate/up bias"),
                                       (w2b, has_w2_bias, "down bias")):
            if (operand is not None) != present:
                raise ValueError(
                    f"the kernel was built for weight format "
                    f"{weight_format!r} with has_w1_bias={has_w1_bias} "
                    f"has_w2_bias={has_w2_bias}, which "
                    f"{'wants' if present else 'has no operand for'} the "
                    f"{what}, and it is "
                    f"{'present' if operand is not None else 'absent'}")
        act_scale_arg = ((act_scales.reshape(
            host.act_scale_slab_rows(ragged_rows_alloc), HIDDEN_LANE_BLOCK), )
                         if has_act_scale else ())
        scale_args = ((w1s.astype(jnp.float32),
                       w2s.astype(jnp.float32)) if has_scales else ())
        biases = tuple(
            b.astype(jnp.float32) for b in (w1b, w2b) if b is not None)
        call = make_call(recv_rows)
        res = call(*tables, token_gather.astype(jnp.int32),
                   expert_rows.astype(jnp.int32),
                   expert_base.astype(jnp.int32), visit.astype(jnp.int32),
                   counts.astype(jnp.int32),
                   act_q_gathered.reshape(-1, lane_blocks, HIDDEN_LANE_BLOCK),
                   *act_scale_arg, w1, w2, *scale_args, *biases)
        recv, rscl, _contrib, _contrib_scl = res
        return recv, rscl.reshape(recv_rows, scale_lanes)

    return fn


# Pallas keys its kernel-to-jaxpr cache on the `kernel` object itself, so
# memoizing the build is what makes pallas hit across builds.
_BUILD_CACHE = {}
# Two threads missing on the same key would each build a kernel, destroying
# the object identity this cache holds, so the miss path is serialized.
_BUILD_CACHE_LOCK = threading.Lock()


def build_fused_ep_moe_kernel(**kwargs):
    """Memoizing front door for _build_fused_ep_moe_kernel.

    The cache is unbounded. host.ragged_stride_bound's docstring states the
    growth law that decides how many entries a deployment ends up holding;
    the size is logged on every insert so a boot that is accumulating
    programs says so, rather than being read off host memory afterwards.
    """
    key = tuple(sorted(kwargs.items()))
    fn = _BUILD_CACHE.get(key)
    if fn is None:
        with _BUILD_CACHE_LOCK:
            fn = _BUILD_CACHE.get(key)
            if fn is None:
                fn = _build_fused_ep_moe_kernel(**kwargs)
                _BUILD_CACHE[key] = fn
                logger.info("fused EP MoE: built program %d for %s",
                            len(_BUILD_CACHE),
                            ", ".join(f"{k}={v}" for k, v in key))
    return fn
