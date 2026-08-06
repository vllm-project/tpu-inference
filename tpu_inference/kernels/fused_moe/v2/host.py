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
"""Host-side halves of the fused expert-parallel MoE kernel: the layout
constants the two halves share, the routing tables a call builds before it
enters the kernel, and the VMEM accounting that answers whether a build
fits. Nothing here runs inside the Pallas body."""
import enum
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

FP8_MAX = 448.0
FP8 = jnp.float8_e4m3fn
# Storage only: there is no four-bit MXU, so each block is widened to e4m3.
FP4 = jnp.float4_e2m1fn
# Default scale block along the contraction axis for a block-scaled weight.
QB4 = 512
# Four-bit values packed per uint32 word along K. The packing is a property
# of the width, not of the element type: any four-bit form packs eight to a
# word the same way.
PACK4 = 8
BF16 = jnp.bfloat16
# Storage only, the same way four-bit weights are: TPU v7, the generation
# this kernel is validated on, takes no integer input on the matrix unit
# (earlier generations do), so an integer weight is widened to
# bf16 one contraction chunk at a time before it is contracted. The format
# buys weight bytes and weight bandwidth, not matmul work.
INT8 = jnp.int8
# The contraction chunk integer weights widen in. It has to divide both
# matmuls' contractions, and the acceptance check says so.
WIDEN_KCHUNK = 512
# How far ahead of the expert being computed the next weight refill goes
# out: at expert e's head the kernel issues the refill for expert e + 2.
# Two is a schedule choice -- one refill in flight per weight buffer, one
# expert of slack -- not a hardware fact.
WEIGHT_PREFETCH_DISTANCE = 2
# Weight buffer slots. A refill writes its slot while the readers of the
# prefetch distance's worth of earlier experts are still live, so the slot
# count has to exceed the distance for all of them to stay distinct.
NBUF = WEIGHT_PREFETCH_DISTANCE + 1
# Every transport moves whole 8-row blocks; dynamic DMA offsets are
# block-aligned.
ROWBLK = 8
# A token row is a whole number of 128-lane blocks: the lane count of the
# vector unit this kernel is built for.
HIDDEN_LANE_BLOCK = 128
# The most of those blocks the kernel's row staging holds.
HIDDEN_MAX_BLOCKS = 32
# The share of the chip's VMEM the kernel may claim.
VMEM_FRACTION = 0.98
# The routing tables pack an arrival position and an alignment slot into
# one word as position * this + slot, so the slot field is this wide and
# the alignment slots a mesh needs have to fit under it.
ALIGNMENT_SLOT_FIELD = 64
# The widest block the routing tables bin routed pairs over. The rank pass
# inside a block is quadratic in it, so this is a chosen ceiling.
MAX_ROUTING_BLOCK = 256


def _pow2_shift(m, what):
    """The shift a division by `m` is, refusing anything else by name.

    The plan divides and takes remainders by these widths often enough
    that the arithmetic is worth writing as what it is. That rewrite is
    only an identity while the width is a power of two, so the condition
    is checked here, once, at import: a width that stopped being one
    would otherwise turn every rewritten site into silent wrong
    arithmetic rather than an error.
    """
    if m <= 0 or (m & (m - 1)):
        raise ValueError(
            f"{what} is {m}, which is not a power of two. The routing "
            "plan writes its divisions and remainders by this width as a "
            "shift and a mask, which is an identity only for a power of "
            "two, so this width cannot be changed without restoring the "
            "division at every site that reads these constants.")
    return m.bit_length() - 1


# A floor division by a power of two IS an arithmetic right shift and a
# remainder by one IS a bitwise and, for every int32 including negatives:
# both round toward negative infinity and both give the non-negative
# residue. So these are spellings of the same integer, not an assumption
# about the sign of anything the plan computes. Written out, they cost a
# shift or a mask; left as a division, they cost that plus the fixup a
# signed division owes, which this compiler emits and cannot fold away.
ROWBLK_SHIFT = _pow2_shift(ROWBLK, "ROWBLK")
SLOT_FIELD_SHIFT = _pow2_shift(ALIGNMENT_SLOT_FIELD, "ALIGNMENT_SLOT_FIELD")
SLOT_FIELD_MASK = ALIGNMENT_SLOT_FIELD - 1
# out_vm and its scale mirror are double-buffered: one tile computes into
# one parity while the previous tile's commit drains out of the other.
OUT_PARITIES = 2
# The widened view of a weight chunk a format has to widen -- a four-bit
# k-block to fp8, an integer contraction chunk to bf16 -- is double-
# buffered, so the next chunk widens while the current one feeds the MXU.
WIDENED_BLOCK_BUFFERS = 2
# The wire's scale mirror carries one f32 per row, but a mirror one lane
# wide would make every scale DMA a few bytes. Each row is widened to
# hidden // this many lanes, of which only lane zero is ever read; the
# ratio is a chosen one rather than a hardware fact.
SCALE_MIRROR_LANE_RATIO = 1024


def align_up(v, m):
    return -(-v // m) * m


def row_lane_blocks(hidden):
    """Lane blocks one token row is staged as: a row is (this many, 128)."""
    return hidden // HIDDEN_LANE_BLOCK


def scale_mirror_lanes(hidden):
    """f32 lanes the wire's scale mirror holds for one row."""
    return max(1, hidden // SCALE_MIRROR_LANE_RATIO)


def act_scale_slab_rows(rows_alloc):
    """Sublanes the activation row-scale slab is handed to the kernel as.

    The slab is one f32 per slab row, and a column of them is the wrong
    thing to ship: a [rows, 1] array is padded out to a full lane block on
    the way to the kernel, so building it costs a copy of the padded array
    and that copy grows with the row count. The dense [rows / lanes, lanes]
    view holds the same bytes in the same order -- a lane block of f32 is
    exactly the tile the flat array is already laid out in -- so the layer
    hands the kernel that view and pays nothing to build it.

    ONE ROW MORE than the slab needs. The kernel reads a tile's scales as
    the sublanes its first row falls in, and a tile whose first row is not
    on a lane-block boundary reaches into the next sublane; at the last
    tile that sublane is past the slab. Bounds checks are off in this
    kernel, so the row exists rather than being trusted not to be read.
    Nothing selects from it: the rows it carries are past the slab's last.
    """
    return align_up(rows_alloc, HIDDEN_LANE_BLOCK) // HIDDEN_LANE_BLOCK + 1


def act_scale_window_rows(tile_m):
    """Sublanes of the scale slab one tile of `tile_m` rows can touch."""
    return align_up(tile_m, HIDDEN_LANE_BLOCK) // HIDDEN_LANE_BLOCK + 1


class WeightFormat(str, enum.Enum):
    """The weight formats the kernel takes, one member per accepted form.

    A member IS its own spelling: the value is the word the format has
    always been called by, and __str__ and __repr__ render that word, so a
    kernel name, a log line, a cache key and a refusal read exactly what
    they read when the format was a bare string. The point of the type is
    that a format now has one declaration to spell it, and a caller naming
    one that does not exist is refused by the table below rather than
    carried as far as the first comparison that quietly fails.
    """
    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    BF16 = "bf16"

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


class WeightForm(NamedTuple):
    """One weight format the kernel takes, and everything that follows it.

    A caller names a format; every other element type in the layer is read
    off this record rather than written down where it is used, so there is
    one place that says what a format implies.

    scale_layout is the layout of the weight scales the caller supplies:
    "per_channel" is one f32 per output channel of each matmul,
    "per_contraction_block" one per (contraction block, output channel), and
    "none" is an unquantized weight that carries no scales at all.

    act_dtype is what token rows are staged and contracted in and act_max is
    the peak they are row-quantized against, or None where they are carried
    as they arrive. wire_dtype is what a computed result row crosses the
    transport in; it follows the activation format, because an arrival row
    re-enters its token owner's sum as one more term of the same kind.
    """
    name: WeightFormat
    weight_dtype: object
    act_dtype: object
    act_max: object
    scale_layout: str
    wire_dtype: object

    @property
    def has_scales(self):
        """Whether the caller supplies weight scales at all."""
        return self.scale_layout != "none"

    @property
    def quantized_activations(self):
        """Whether token rows are quantized, and so carry a row scale."""
        return self.act_max is not None


# The accepted (weight dtype, weight-scale layout) pairs. A weight element
# type outside this table is refused by name rather than reinterpreted, and
# a refusal lists these keys, so a caller is told what IS accepted.
#
# Two four-bit forms are deliberately absent. Integer-four has no scale
# layout here and no widening; and a four-bit block of 32, which the common
# mixed-format layout uses, is refused by the packed-weight row tile rather
# than by this table: eight values pack into a 32-bit word and those words
# tile to eight sublanes, so a block is a whole number of 64 rows or it is
# not addressable at all.
#
# Only the ELEMENT TYPE is fp4-specific. The packing, the block geometry,
# the staging and the transport are four-bit generic, and the block-scaled
# FFN body contracts whatever the reader hands it, so a block-scaled
# integer-four form would be one more row here plus the bitcast and widening
# target in the block reader -- the body itself would be reused verbatim.
WEIGHT_FORMS = {
    WeightFormat.FP8:
    WeightForm(WeightFormat.FP8, FP8, FP8, FP8_MAX, "per_channel", FP8),
    WeightFormat.FP4:
    WeightForm(WeightFormat.FP4, FP4, FP8, FP8_MAX, "per_contraction_block",
               FP8),
    WeightFormat.INT8:
    WeightForm(WeightFormat.INT8, INT8, BF16, None, "per_channel", BF16),
    WeightFormat.BF16:
    WeightForm(WeightFormat.BF16, BF16, BF16, None, "none", BF16),
}

WEIGHT_FORMAT_NAMES = tuple(WEIGHT_FORMS)


def weight_form(weight_format):
    """The record for a format; raises naming the accepted set."""
    try:
        return WEIGHT_FORMS[WeightFormat(weight_format)]
    except ValueError:
        raise NotImplementedError(
            f"the fused EP MoE kernel takes weight formats "
            f"{WEIGHT_FORMAT_NAMES}; got {weight_format!r}") from None


def weight_format_of_dtype(dtype):
    """The format carrying this weight element type, or None.

    This is the one place a format is DERIVED rather than named, so it is
    also the one place the enum is constructed from something outside it;
    everything downstream carries the member this returns.
    """
    for weight_format, form in WEIGHT_FORMS.items():
        if jnp.dtype(form.weight_dtype) == jnp.dtype(dtype):
            return WeightFormat(weight_format)
    return None


def ragged_stride_bound(num_tokens, topk, e_total, capacity):
    """Per-shard slab rows the no-drop worst case needs: every routed row
    landing on one shard, the row-block alignment padding and one
    tile-height tail-read window.

    This value is part of the built kernel's cache key, and it is a function
    of the TOKEN BUCKET, so the growth law of that cache is: one Pallas
    program per compiled token bucket, times the weight formats a process
    serves, times the four combinations of the two optional expert biases,
    times the fused activations. A deployment on exponential buckets to
    32768 compiles a handful; one that sets a fixed bucket padding gap
    compiles roughly the span divided by that gap. The cache is unbounded
    and holds every one for the life of the process, so a deployment that
    widens its bucket set pays in host memory and in boot time, and the
    build logs its size on every insert.

    The bound is a MINIMUM, so rounding it up to a coarser bucket spacing
    would collapse that cardinality without changing what the kernel
    computes. That is a design change rather than a bound, and it is not
    made here.
    """
    return align_up(num_tokens * topk + (ROWBLK - 1) * e_total + capacity,
                    capacity)


def gather_clamp_bounds(T):
    """The (low, high) token numbers the routing gather table is clipped to.

    A function rather than an expression inside build_routing_tables so the
    bound has a witness: the interesting value is T = 0, which the table
    builder refuses before it gets here, so nothing that goes through the
    builder can check what this returns there.

    Both bounds matter, and the upper one has to be floored at zero to say
    anything at all. At T = 0 the natural bound is T - 1 = -1, and numpy --
    which jax follows -- makes a lower bound above the upper bound yield the
    UPPER bound, so clipping to (0, -1) returns -1 for every entry: the one
    line that exists to keep the gather in bounds is what puts it out of
    them, and bounds checks are off by construction so nothing downstream
    re-checks. At every T the builder accepts this is the identical pair it
    always was.
    """
    return 0, max(T - 1, 0)


def routing_block(t_local, topk):
    """The widest power-of-two block dividing this shard's routed pairs."""
    b = MAX_ROUTING_BLOCK
    while (t_local * topk) % b:
        b //= 2
    return b


class RoutingTables(NamedTuple):
    """Where every routed (token, expert) pair computes, and where it lands.

    One shard's kernel operands are cut from these by the three shard_
    functions below, which is why the whole set is replicated rather than
    sharded.
    """
    # [T, K]: the arrival row each token's k-th selection comes back on.
    arrival_row: jax.Array
    # [T * K]: the slab row each routed pair computes on.
    slab_row: jax.Array
    # [E, ep]: true rows of expert e whose tokens shard d owns.
    run_rows: jax.Array
    # [E, ep]: the same runs, each padded up to a whole ROWBLK.
    run_rows_aligned: jax.Array
    # [E, ep]: where each padded run starts inside its expert's slab.
    run_start_aligned: jax.Array
    # [source, g, dest]: rows one (expert, dest) push carries.
    region_rows: jax.Array
    # [dest, source, g]: the arrival row that push lands on.
    recv_base: jax.Array
    # [dest]: arrival rows a shard receives, its own rows included.
    recv_rows: jax.Array
    # [E]: padded rows per expert.
    expert_rows_aligned: jax.Array
    # [E]: where each expert's slab starts.
    expert_base: jax.Array
    # Slab rows one shard is allocated.
    rows_alloc: int


def build_routing_tables(topk_idx, *, e_total, ep, t_local, block, tile_m,
                         shard_stride):
    """Assign every routed (token, expert) pair the slab row it computes on."""
    # topk_idx [T, K] i32 expert ids, all-gathered. Rows order by expert then
    # by owning shard, each run padded to ROWBLK.
    T, K = topk_idx.shape
    n = T * K
    g_local = e_total // ep
    # Every check here tests an argument, not this function's own
    # arithmetic, so each refuses rather than asserts: under `python -O` an
    # assert is gone and the overflow it guards becomes silent corruption of
    # the routing tables. The serving path re-checks the field width
    # independently; a direct caller of this module has only these.
    if T < 1:
        raise ValueError(
            "the routing tables need at least one token; an empty batch "
            "builds a gather table with no in-bounds row to clamp to")
    if e_total % ep:
        raise ValueError(f"expert count {e_total} is not divisible by the "
                         f"expert-parallel width {ep}")
    # packed below carries the arrival position and the alignment slot in
    # one word. The slot runs to (ROWBLK - 1) * (ep - 1), so a wider ep
    # overflows its field and the two silently corrupt each other.
    if (ROWBLK - 1) * (ep - 1) >= ALIGNMENT_SLOT_FIELD:
        raise ValueError(
            f"ep={ep} needs up to {(ROWBLK - 1) * (ep - 1)} alignment slots "
            f"and the routing tables pack them into {ALIGNMENT_SLOT_FIELD}; "
            f"widths up to {1 + (ALIGNMENT_SLOT_FIELD - 1) // (ROWBLK - 1)} "
            "are representable")
    # The other factor of that same packed word. The position field holds
    # what is left of the 32 bits, and past it the arrival position and the
    # alignment slot corrupt each other with no error anywhere near them.
    max_pos = n + (ROWBLK - 1) * e_total
    if max_pos * ALIGNMENT_SLOT_FIELD >= 2**31:
        raise ValueError(
            f"the routing tables would carry up to {max_pos} arrival rows "
            f"per shard, and packing one alongside a {ALIGNMENT_SLOT_FIELD}"
            f"-wide alignment slot holds {2**31 // ALIGNMENT_SLOT_FIELD}: "
            f"past that the two fields corrupt each other silently")
    if n % block:
        raise ValueError(f"the {n} routed pairs are not a whole number of "
                         f"{block}-pair routing blocks")
    if (t_local * K) % block:
        raise ValueError(f"this shard's {t_local * K} routed pairs are not a "
                         f"whole number of {block}-pair routing blocks")
    if tile_m % ROWBLK:
        raise ValueError(f"tile height {tile_m} is not a whole number of "
                         f"{ROWBLK}-row blocks")
    expert_of_pair = topk_idx.reshape(-1).astype(jnp.int32)
    n_blocks = n // block
    expert_blocks = expert_of_pair.reshape(n_blocks, block)
    bins = jnp.arange(e_total, dtype=jnp.int32)

    block_hist = jnp.sum(
        (expert_blocks[:, :, None] == bins[None, None, :]).astype(jnp.int32),
        axis=1)  # [n_blocks, E]
    block_off = jnp.cumsum(block_hist, axis=0) - block_hist  # excl over blocks
    base_per_slot = jnp.sum(jnp.where(
        expert_blocks[:, :, None] == bins[None, None, :],
        block_off[:, None, :], 0),
                            axis=2)  # [n_blocks, block]
    eq = expert_blocks[:, :, None] == expert_blocks[:, None, :]
    tri = jnp.tril(jnp.ones((block, block), dtype=jnp.bool_), k=-1)
    rank = jnp.sum((eq & tri[None]).astype(jnp.int32), axis=2)
    pair_rank = (base_per_slot + rank).reshape(-1)  # [n] rank in its expert

    blocks_per_dest = (t_local * K) // block
    rows_by_dest = block_hist.reshape(ep, blocks_per_dest, e_total).sum(axis=1)
    run_rows = rows_by_dest.T  # [E, ep]
    run_start = jnp.cumsum(run_rows, axis=1) - run_rows  # excl over d

    # Rounding up to a whole block: one add and one mask, where the
    # negate-divide-negate spelling owed a signed division's fixup.
    run_rows_aligned = (run_rows + (ROWBLK - 1)) & jnp.int32(-ROWBLK)
    run_start_aligned = (jnp.cumsum(run_rows_aligned, axis=1) -
                         run_rows_aligned)
    slot_shift = run_start_aligned - run_start  # slot = pair_rank + shift

    # Ragged expert slabs: shard s owns rows [s*stride, (s+1)*stride).
    expert_rows_aligned = run_rows_aligned.sum(axis=1)  # [E] 8-aligned
    if shard_stride % tile_m:
        raise ValueError(f"the per-shard slab stride {shard_stride} is not a "
                         f"whole number of {tile_m}-row tiles")
    rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
    local_base = jnp.cumsum(rows_by_shard, axis=1) - rows_by_shard  # [s, G]
    expert_base = (local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] *
                                 shard_stride)).reshape(e_total)
    rows_alloc = ep * shard_stride

    # Every push is per expert, so a receive region is one (source, expert)
    # run: region_rows is run_rows_aligned read as [source, expert, dest].
    region_rows = run_rows_aligned.reshape(ep, g_local, ep)  # [s, g, d]
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(ep, ep * g_local)
    recv_base = (jnp.cumsum(rows_per_dest, axis=1) - rows_per_dest).reshape(
        ep, ep, g_local)  # [d, s, g]
    recv_rows = rows_per_dest.sum(axis=1)  # [d] incl. self

    pos_shift = recv_base.transpose(1, 2, 0).reshape(e_total, ep) - run_start
    dest_of_block = (jnp.arange(n_blocks, dtype=jnp.int32) *
                     block) // (t_local * K)

    # One packed pass over both shifts; the ragged expert base needs its
    # own select-sum because its value range is too wide for the word.
    packed = pos_shift * ALIGNMENT_SLOT_FIELD + slot_shift
    packed_blocks = jnp.take(packed.T, dest_of_block, axis=0)  # [n_blocks, E]
    packed_sel = jnp.sum(jnp.where(
        expert_blocks[:, :, None] == bins[None, None, :],
        packed_blocks[:, None, :], 0),
                         axis=2).reshape(-1)
    base_of_pair = jnp.sum(jnp.where(
        expert_blocks[:, :, None] == bins[None, None, :],
        expert_base[None, None, :], 0),
                           axis=2).reshape(-1)
    # Unpacking the two fields of the packed word: the position is the
    # high part and the slot is the low part, which is a shift and a mask.
    pos = pair_rank + jnp.right_shift(packed_sel, SLOT_FIELD_SHIFT)
    slot = pair_rank + (packed_sel & SLOT_FIELD_MASK)

    slab_row = base_of_pair + slot  # always < total

    return RoutingTables(arrival_row=pos.reshape(T, K),
                         slab_row=slab_row,
                         run_rows=run_rows,
                         run_rows_aligned=run_rows_aligned,
                         run_start_aligned=run_start_aligned,
                         region_rows=region_rows,
                         recv_base=recv_base,
                         recv_rows=recv_rows,
                         expert_rows_aligned=expert_rows_aligned,
                         expert_base=expert_base,
                         rows_alloc=rows_alloc)


def shard_expert_slabs(routing, me, *, e_total, ep):
    """Shard `me`'s local experts: (rows, slab start) i32 [G], in row units."""
    g_local = e_total // ep
    rows = lax.dynamic_slice(routing.expert_rows_aligned, (me * g_local, ),
                             (g_local, ))
    base_g = lax.dynamic_slice(routing.expert_base, (me * g_local, ),
                               (g_local, ))
    base = base_g - base_g[0]
    return rows.astype(jnp.int32), base.astype(jnp.int32)


def local_slab_rows(routing, me, *, shard_stride):
    """`routing.slab_row` rebased onto shard `me`'s own slab.

    A pair that computes on another shard is sent past the end of this
    shard's slab, so a scatter through this index carrying mode="drop"
    writes only the rows this shard will read. The off-shard pairs are
    dropped rather than accumulated onto a sink row, because every shard but
    one owns most of the pairs and a sink would take (ep - 1) / ep of them
    as write conflicts on a single row.

    The index is never negative: a row below this shard's slab is mapped to
    shard_stride, not to a negative offset, because jax indexing wraps a
    negative index around the destination instead of dropping it.
    """
    base = me * shard_stride
    return jnp.where(routing.slab_row < base, jnp.int32(shard_stride),
                     routing.slab_row - base)


def shard_token_gather(routing, me, *, shard_stride):
    """The token number each of shard `me`'s slab rows computes, [stride].

    The scatter runs against this shard's slab alone rather than against the
    replicated ep-wide slab followed by a slice, which built seven eighths
    of its result to discard it. What the kernel receives is unchanged.

    The kernel fetches an input row by this table value with bounds checks
    off, so the one index that reaches memory is clamped here rather than
    trusted. An expert id outside [0, e_total) would land two pairs on one
    row and sum their token numbers; the acceptance check refuses a gating
    width that could produce one, and this is the backstop for it. Clipped
    on BOTH sides, to the bounds gather_clamp_bounds names -- see there for
    why the upper one is floored at zero. build_routing_tables refuses T = 0
    before any of this; the clip is written so that it would hold on its own.
    """
    T, K = routing.arrival_row.shape
    # The token number of each routed pair is a counted repeat, not a
    # division: pair i belongs to token i // K, which is the token index
    # broadcast K times. K is the selection width and need not be a power
    # of two, so this is the spelling that removes the division rather
    # than the one that shifts it.
    token_of_pair = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)).reshape(-1)
    row = local_slab_rows(routing, me, shard_stride=shard_stride)
    gather_lo, gather_hi = gather_clamp_bounds(T)
    scattered = jnp.zeros((shard_stride, ),
                          jnp.int32).at[row].add(token_of_pair, mode="drop")
    return jnp.clip(scattered, jnp.int32(gather_lo), jnp.int32(gather_hi))


def expert_visit_list(rows, g_local):
    """The local experts to visit, and how many, from rows [g_local] i32."""
    # visit[:n_visit] = the local expert indices with rows > 0, most rows
    # first; the tail is never visited. Ties break by ascending index. The
    # visit order does not change the output, only the weight refill order.
    mask = rows > 0
    n_visit = mask.sum().astype(jnp.int32).reshape(1)
    order = jnp.arange(g_local, dtype=jnp.int32)
    rows_i = rows.astype(jnp.int32)
    # One key, sorted once. The negated row count is the high factor of the
    # word and the index the low one, so the packed integer compares exactly
    # as the (rows descending, index ascending) pair does: every index is
    # below g_local, so the low factor is exact and no two experts can tie.
    # An empty expert's negated count is zero, the largest value the high
    # factor takes, so the empty experts land last without an activity key
    # of their own. The word is int32, and an expert's rows never exceed the
    # shard's slab row allocation, so the pack is exact while that
    # allocation stays under 2**31 // g_local.
    perm = jnp.argsort(order - rows_i * g_local).astype(jnp.int32)
    visit = jnp.minimum(perm, jnp.int32(g_local - 1)).astype(jnp.int32)
    return visit, n_visit


def shard_push_tables_in_rows(routing, me, *, e_total, ep):
    """Shard `me`'s push tables in ROW units, for true-length pushes."""
    # true_rows [G, ep] rows per (e, d), recv_row_off [G, ep] the arrival row
    # that run starts at, totals [2] send and remote recv rows. Only the
    # pushed lengths shrink: the recv and contrib layouts stay aligned, so the
    # arrival tables do not move.
    g_local = e_total // ep
    all_run_rows = routing.run_rows  # [E, ep]
    true_rows = lax.dynamic_slice(all_run_rows, (me * g_local, 0),
                                  (g_local, ep))
    recv_base = routing.recv_base  # [d, s, g]
    my_recv_base = lax.dynamic_slice(recv_base, (0, me, 0),
                                     (ep, 1, g_local))[:, 0]
    recv_row_off = my_recv_base.T  # [G, d]
    not_me = (jnp.arange(ep) != me).astype(true_rows.dtype)
    send_true = (true_rows * not_me[None, :]).sum()
    self_true = (true_rows * (1 - not_me)[None, :]).sum()
    recv_true = lax.dynamic_slice(
        all_run_rows.sum(axis=0).astype(jnp.int32), (me,), (1,))[0] \
        - self_true
    return (true_rows.astype(jnp.int32), recv_row_off.astype(jnp.int32),
            jnp.stack([send_true, recv_true]).astype(jnp.int32))


def shard_transport_tables_in_blocks(routing, me, *, e_total, ep):
    """Shard `me`'s transport tables in 8-ROW BLOCK units."""
    # All i32. contrib: regions per dest d, packed in d order, each region =
    # groups asc, experts asc. recv: regions per (src asc, group asc).
    g_local = e_total // ep
    # Every table value below is in 8-row BLOCK units, which the
    # conversions at the bottom reach by a shift.
    aligned_rows = lax.dynamic_slice(routing.run_rows_aligned,
                                     (me * g_local, 0), (g_local, ep))
    run_start = lax.dynamic_slice(routing.run_start_aligned, (me * g_local, 0),
                                  (g_local, ep))
    my_region_rows = lax.dynamic_slice(routing.region_rows, (me, 0, 0),
                                       (1, g_local, ep))[0]
    recv_base = routing.recv_base  # [d, s, g]

    not_me = (jnp.arange(ep) != me)
    # contrib includes the own-dest region, which hops to recvbuf later.
    out_total = my_region_rows.sum(axis=0)  # [d] incl. me
    contrib_base = jnp.cumsum(out_total) - out_total  # [d]
    grp_off = jnp.cumsum(my_region_rows, axis=0) - my_region_rows  # [g, d]

    # One expert per push, so an expert's contrib offset is its region's.
    contrib_off = contrib_base[None, :] + grp_off  # [G, d]

    push_src = contrib_base[None, :] + grp_off  # [g, d]
    push_len = my_region_rows
    # recv_base[d (receiver), me (src), g] for every d: [d, g] -> [g, d].
    my_recv_base = lax.dynamic_slice(recv_base, (0, me, 0),
                                     (ep, 1, g_local))[:, 0]
    push_dst = my_recv_base.T
    not_me_i = not_me.astype(my_region_rows.dtype)
    send_rows = (my_region_rows * not_me_i[None, :]).sum()
    self_rows = jnp.sum(aligned_rows * (1 - not_me_i)[None, :])
    recv_remote = lax.dynamic_slice(routing.recv_rows, (me,), (1,))[0] \
        - self_rows
    totals = jnp.stack([
        jnp.right_shift(send_rows, ROWBLK_SHIFT),
        jnp.right_shift(recv_remote, ROWBLK_SHIFT),
    ]).astype(jnp.int32)

    def i32(a):
        """One table in block units: the row count's high bits."""
        return jnp.right_shift(a, ROWBLK_SHIFT).astype(jnp.int32)

    return (i32(run_start), i32(aligned_rows), i32(contrib_off), i32(push_src),
            i32(push_len), i32(push_dst), totals)


# Rows of the count table, in the order the kernel reads them. The two
# transport rows are in ROWS; the kernel divides them into blocks, which
# is where that division is cheapest (see shard_count_vector).
COUNT_SEND_ROWS = 0
COUNT_RECV_ROWS = 1
COUNT_SEND_ALIGNED_ROWS = 2
COUNT_RECV_ALIGNED_ROWS = 3
COUNT_VISITS = 4
N_COUNTS = 5


def shard_count_vector(routing, expert_rows, me, *, e_total, ep):
    """Shard `me`'s five kernel counts, one per row, [N_COUNTS, ep] i32.

    The kernel needs five integers: the rows it sends and receives, the
    same two in block units, and how many local experts it visits. Built
    one at a time they are five separate reductions to a scalar, and a
    reduction that lands on a scalar is the expensive shape in this stage:
    the neighbouring reduction that keeps a vector reads a far larger
    array for a fraction of the time.

    So all five are built in one pass that stops one step early, keeping
    the expert-parallel axis, and the kernel closes them. Summing the last
    axis is free where it is finished: those tables live in scalar memory,
    `ep` is a build-time constant, and the sum is a fixed run of loads and
    adds with no loop.

    Two of the five used a dynamic index to pick this shard's own entry
    out of a per-destination vector. Here that is a mask instead, so the
    pick joins the same pass rather than standing as its own operation.
    `first` is what keeps such a per-destination total from being added
    once per local expert: it survives on one row and is zero on the rest.

    Two spellings of the same algebra are NOT the same cost, and the
    cheap one is not the obvious one.

    The rows are reduced first and joined afterwards. Stacking the five
    terms and reducing the stack materializes a [N_COUNTS, G, ep] array to
    read it once; reducing each term and joining the results concatenates
    five [1, ep] rows instead, which is a few dozen values. The compiler's
    own cost note prefers the second by more than the whole operation this
    is trying to remove.

    The block counts stay in ROW units here and the kernel divides. Every
    aligned run is a whole number of ROWBLK rows, so the quotient is the
    same taken before or after the sum, but taken here it is a floor
    division on a signed array, which carries its sign fixup on every
    element; taken in the kernel it is one scalar division by a build-time
    constant, on values that are already reduced.
    """
    g_local = e_total // ep
    all_run_rows = routing.run_rows  # [E, ep]
    true_rows = lax.dynamic_slice(all_run_rows, (me * g_local, 0),
                                  (g_local, ep)).astype(jnp.int32)
    aligned = lax.dynamic_slice(routing.run_rows_aligned, (me * g_local, 0),
                                (g_local, ep)).astype(jnp.int32)
    mine = (jnp.arange(ep, dtype=jnp.int32) == me).astype(jnp.int32)  # [ep]
    other = 1 - mine
    first = (jnp.arange(g_local, dtype=jnp.int32) == 0).astype(
        jnp.int32)[:, None]  # [G, 1]
    recv_rows_all = all_run_rows.sum(axis=0).astype(jnp.int32)  # [ep]
    recv_total = routing.recv_rows.astype(jnp.int32)  # [ep]
    active = (expert_rows > 0).astype(jnp.int32)[:, None]  # [G, 1]

    def over_experts(term):
        """One count's reduction, keeping the expert-parallel axis."""
        return term.sum(axis=0, keepdims=True)  # [1, ep]

    return jnp.concatenate([
        over_experts(true_rows * other[None, :]),
        over_experts((recv_rows_all * mine)[None, :] * first -
                     true_rows * mine[None, :]),
        over_experts(aligned * other[None, :]),
        over_experts((recv_total * mine)[None, :] * first -
                     aligned * mine[None, :]),
        over_experts(active * mine[None, :]),
    ],
                           axis=0).astype(jnp.int32)  # [N_COUNTS, ep]


def vmem_limit():
    """VMEM budget for the kernel, read from this generation's capacity."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_FRACTION)


# The oldest chip generation this kernel has been built and measured on.
# Everything below reads its geometry off the device record, so an earlier
# generation produces a kernel that compiles and runs and has never been
# correctness-checked or timed anywhere. The number is written down here
# rather than at the serving adapter, because it is a statement about what
# this kernel has been run on.
MIN_GENERATION = 7


def chip_generation(info=None):
    """The generation of the chip this build is for.

    Read off the same device record the VMEM accounting reads, so a host
    with no chip to name raises here the way it raises there, and one
    guard in the caller answers for both.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    return info.generation


def array_vmem_bytes(shape, dtype, info):
    """Bytes one VMEM array of this shape and dtype occupies.

    The minor dimension pads to the chip's lane count and the second-minor
    to this dtype's sublane tiling; both numbers come off the device record
    rather than being written down. A one-dimensional array is laid out as
    a single row of lanes. Padding is most of the difference between an
    array's element count and the memory it costs: a scale mirror four
    lanes wide still pays for a full row of lanes.
    """
    itemsize = jnp.dtype(dtype).itemsize
    if len(shape) == 1:
        return align_up(shape[0], info.num_lanes) * itemsize
    lanes = align_up(shape[-1], info.num_lanes)
    sublanes = align_up(shape[-2], info.get_sublane_tiling(dtype))
    return math.prod(shape[:-2]) * sublanes * lanes * itemsize


# A 32-bit array's second-minor dimension tiles to this many sublanes, so a
# packed four-bit k-block has to be a whole number of them deep. Written
# down rather than read off the device record, unlike everything else in
# this file's layout arithmetic, because the acceptance check that uses it
# answers before any device read and the serving adapter builds the same
# refusal string at import on a host with no chip attached.
# check_u32_sublane_tile is what keeps the literal honest.
U32_SUBLANE_TILE = 8


def check_u32_sublane_tile(info=None):
    """Refuse a chip whose 32-bit sublane tiling is not the one the packed
    four-bit weight layout is written for.

    array_vmem_bytes sizes the packed w1_vm and w2_vm buffers off the device
    record and the four-bit acceptance check decides whether the block size
    that indexes them is addressable from the constant above. The two are
    load-bearing for one layout, so on a generation where they disagree the
    accounting adapts and the acceptance check does not, and the
    disagreement surfaces as a Mosaic slice error inside the kernel body
    with nothing naming the constant. This names it.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    queried = info.get_sublane_tiling(jnp.uint32)
    if queried != U32_SUBLANE_TILE:
        raise ValueError(
            f"the four-bit weight stream is packed as 32-bit words whose "
            f"second-minor dimension tiles to {U32_SUBLANE_TILE} sublanes, "
            f"and this chip tiles them to {queried}; the packed-weight row "
            f"tile the block size is checked against is written for "
            f"{U32_SUBLANE_TILE} and the buffers beside it are sized from "
            "the device record, so the two no longer describe one layout")


def vmem_scratch_arrays(g_local,
                        capacity,
                        hidden,
                        inter,
                        *,
                        nbuf=NBUF,
                        weight_format=WeightFormat.FP8,
                        rhs_qb=QB4,
                        has_w1_bias=False,
                        has_w2_bias=False):
    """The kernel's VMEM scratch buffers, as (name, shape, dtype).

    _build_fused_ep_moe_kernel declares its scratch from this list, in this
    order, and the VMEM accounting below sums this same list, so a buffer
    cannot be resized on one side only.

    Every element type here comes off the weight format's record: the
    weight slabs, the row staging the activations arrive in, the wire
    buffer results leave in, and whether the scale tables and the
    activation row scale exist at all.

    The staging buffers are three-dimensional on purpose. An eight-bit
    array's second-minor dimension tiles to 32 sublanes, so a flat
    [rows, hidden] buffer could only be sliced at row offsets that are
    multiples of 32; splitting a token row across the two minor dimensions
    leaves the row axis untiled, which is what makes an 8-row transport
    offset legal. It does that for every element type, which is why a
    sixteen-bit wire takes the same geometry at twice the bytes.
    """
    form = weight_form(weight_format)
    lane_blocks = row_lane_blocks(hidden)
    scale_lanes = scale_mirror_lanes(hidden)
    # The tile height IS the capacity, and the out pair is in tile units.
    tile_m = capacity
    tile_blocks = tile_m // ROWBLK
    # Every local expert's scale table is resident. At four-bit block scales
    # that makes the two scale tables some of the largest buffers here.
    if weight_format == WeightFormat.FP4:
        # Four-bit weights stream as PACKED u32 words ([K/8, N]), so the
        # transfer moves half the bytes an eight-bit slab would.
        weights = [
            ("w1_vm", (nbuf, hidden // PACK4, 2 * inter), jnp.uint32),
            ("w2_vm", (nbuf, inter // PACK4, hidden), jnp.uint32),
            ("w1s_vm", (g_local, hidden // rhs_qb, 2 * inter), jnp.float32),
            ("w2s_vm", (g_local, inter // rhs_qb, hidden), jnp.float32),
        ]
    else:
        weights = [
            ("w1_vm", (nbuf, hidden, 2 * inter), form.weight_dtype),
            ("w2_vm", (nbuf, inter, hidden), form.weight_dtype),
        ]
        if form.has_scales:
            weights += [
                ("w1s_vm", (g_local, 2 * inter), jnp.float32),
                ("w2s_vm", (g_local, hidden), jnp.float32),
            ]
    # The optional expert biases are resident for every local expert, one
    # row per expert on each matmul's output channels. They carry no block
    # dimension, so they take the same shape on every weight format -- the
    # shape and the cost of the per-channel scale tables beside them.
    biases = []
    if has_w1_bias:
        biases.append(("w1b_vm", (g_local, 2 * inter), jnp.float32))
    if has_w2_bias:
        biases.append(("w2b_vm", (g_local, hidden), jnp.float32))
    # The activation row scale exists only where the rows were quantized.
    # It rides per tile, so it is sized by the OUT-PARITY count: a tile's
    # stream is issued into, waited on and read at the parity its results
    # will be staged in, and never at any other index. nbuf is the depth the
    # WEIGHT slabs need so a refill and the live readers of the prefetch
    # distance's worth of earlier experts occupy distinct slots, which is a
    # different question and a larger answer.
    act_scale = ([("ls_vm", (OUT_PARITIES, act_scale_window_rows(capacity),
                             HIDDEN_LANE_BLOCK),
                   jnp.float32)] if form.quantized_activations else [])
    return [
        # The indirect form keeps each token row as one lane-block row.
        # Parity-deep for the same reason ls_vm is.
        ("lhs_vm", (OUT_PARITIES, capacity, lane_blocks, HIDDEN_LANE_BLOCK),
         form.act_dtype),
        *weights,
        *biases,
        *act_scale,
        ("out_vm", (OUT_PARITIES, tile_m, lane_blocks, HIDDEN_LANE_BLOCK),
         form.wire_dtype),
        ("oscl_vm", (OUT_PARITIES, tile_blocks, ROWBLK, scale_lanes),
         jnp.float32),
    ]


def vmem_tile_body_arrays(capacity,
                          hidden,
                          inter,
                          *,
                          weight_format=WeightFormat.FP8,
                          rhs_qb=QB4):
    """What one tile body keeps live, as (name, shape, dtype).

    These are not declared scratch: the compiler places them, in the same
    memory as the buffers above. In order, the first matmul's accumulator,
    the bf16 intermediate the activation chunk loop builds up before it is
    concatenated, that concatenation -- as an fp8 requantization where the
    format has one and as a second bf16 array where it does not -- the
    second matmul's accumulator, its bf16 result, and the wire row that
    result is quantized into where the wire is eight-bit.

    A format whose weights reach the matrix unit through a widening adds one
    widened contraction chunk, double-buffered; the two matmuls widen in
    turn and never overlap, so the peak is the wider of the two. Four-bit
    weights widen a scale block to fp8, integer weights a fixed contraction
    chunk to bf16.

    The row staging a tile reads is a view of lhs_vm, not a second array,
    so it is not counted again here.
    """
    form = weight_form(weight_format)
    tile_m = capacity
    arrays = [
        ("acc1", (tile_m, 2 * inter), jnp.float32),
        ("mid_chunks", (tile_m, inter), jnp.bfloat16),
    ]
    # The second matmul takes the intermediate quantized only where its
    # weights are. The other formats contract the bf16 rows directly -- but
    # they still materialize a second copy: the non-requantizing path
    # concatenates the chunk list into one [tile_m, inter] bf16 array while
    # the chunks that feed it are still live. The requantizing path does not,
    # because it quantizes straight out of the chunk list into mid_q.
    if form.quantized_activations:
        arrays.append(("mid_q", (tile_m, inter), FP8))
    else:
        arrays.append(("mid_concat", (tile_m, inter), jnp.bfloat16))
    arrays += [
        ("acc2", (tile_m, hidden), jnp.float32),
        ("down_bf16", (tile_m, hidden), jnp.bfloat16),
    ]
    # An eight-bit wire quantizes the down projection into a fresh array
    # while down_bf16, the bf16 source it reduces over, is still live. A
    # sixteen-bit wire ships down_bf16 itself and materializes nothing.
    if form.wire_dtype is FP8:
        arrays.append(("wire_rows", (tile_m, hidden), FP8))
    if weight_format == WeightFormat.FP4:
        arrays.append(("widened_weight_block", (WIDENED_BLOCK_BUFFERS, rhs_qb,
                                                max(2 * inter, hidden)), FP8))
    elif weight_format == WeightFormat.INT8:
        arrays.append(
            ("widened_weight_block", (WIDENED_BLOCK_BUFFERS, WIDEN_KCHUNK,
                                      max(2 * inter, hidden)), BF16))
    return arrays


def vmem_estimate_bytes(g_local,
                        capacity,
                        hidden,
                        inter,
                        nbuf=NBUF,
                        weight_format=WeightFormat.FP8,
                        rhs_qb=QB4,
                        has_w1_bias=False,
                        has_w2_bias=False,
                        info=None):
    """VMEM one built kernel occupies, from the arrays it declares.

    An upper bound, and deliberately so: the declared scratch is live for
    the whole call, while the tile body's values are counted as though all
    of them were live at once even though several die before the next is
    born. Nothing is left out, which is what a caller asking "will this
    fit" needs; a caller wanting the exact high-water mark should read the
    figure the compiler reports for a built kernel.

    Reads the chip's lane count and per-dtype sublane tiling off the device
    record, so a host with no chip to name raises rather than answering.
    """
    if info is None:
        info = pltpu.get_tpu_info()
    arrays = vmem_scratch_arrays(g_local,
                                 capacity,
                                 hidden,
                                 inter,
                                 nbuf=nbuf,
                                 weight_format=weight_format,
                                 rhs_qb=rhs_qb,
                                 has_w1_bias=has_w1_bias,
                                 has_w2_bias=has_w2_bias)
    arrays += vmem_tile_body_arrays(capacity,
                                    hidden,
                                    inter,
                                    weight_format=weight_format,
                                    rhs_qb=rhs_qb)
    return sum(
        array_vmem_bytes(shape, dtype, info) for _, shape, dtype in arrays)
