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

import dataclasses
import enum

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import utils


def accum_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Return the internal attention accumulator dtype for an output dtype."""
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1:
        return jnp.bfloat16
    return dtype


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tuning parameters for the RPA kernel."""

    bq_sz: int
    bq_c_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int

    def floor_bq_to_decode_q_len(self, decode_q_len: int) -> "BlockSizes":
        """Return a copy with ``bq_sz`` raised to at least ``decode_q_len``.

        The DECODE region processes each sequence's ``decode_q_len`` (=
        ``num_speculative_tokens + 1``) query tokens in a single q-block
        (``num_q == cdiv(decode_q_len, bq_sz) == 1``), which requires ``bq_sz >=
        decode_q_len``. Tuned/default decode entries pin ``bq_sz=1`` for
        single-token decode, so spec-decode must floor it up -- to
        ``decode_q_len`` exactly, with no sublane(8) rounding. ``bq_sz`` is not
        itself a tiled dimension (the flash row dim is
        ``bq_sz * num_q_heads_per_kv_head``), and single-token decode already
        runs ``bq_sz=1``, so a non-8-aligned ``bq_sz`` is valid. The flash
        compute iterates the full static ``bq_sz`` with no per-row clamp, so any
        ``bq_sz`` above ``decode_q_len`` is wasted attention FLOPs (e.g. rounding
        ``q_len=4`` up to 8 doubles the per-seq QK/PV work). Single-token decode
        (``decode_q_len == 1``) is a strict no-op.
        """
        floor = max(int(decode_q_len), 1)
        if self.bq_sz >= floor:
            return self
        # Process the whole draft in ONE flash compute chunk: bq_c_sz == bq_sz.
        # The flash body loops ``for bq_start in range(0, bq_sz, bq_c_sz)``, so a
        # smaller bq_c_sz (e.g. the single-token default of 1) splits the bq_sz
        # q-rows into bq_sz separate QK+softmax+PV matmuls (M = qpk each), paying
        # ~bq_sz x the MXU fill/launch overhead. Measured ~35% slower for spec
        # decode at mid context -- enough to regress bq=4 below the old padded
        # bq=8. One chunk (M = bq_sz*qpk) is the efficient tiling here.
        return dataclasses.replace(self, bq_sz=floor, bq_c_sz=floor)

    def cap_bq_to_total_q(self, total_q_tokens: int) -> "BlockSizes":
        """Return a copy with ``bq_sz`` capped at the (sublane-aligned) q count.

        The MIXED kernel processes queries in blocks of ``bq_sz``; a ``bq_sz``
        larger than ``total_q_tokens`` only pads (and still fully computes) empty
        q rows. The prefill-tuned ``bq_sz`` (e.g. 768) is therefore very wasteful
        for small-query MIXED workloads where total_q_tokens may be only a few
        hundred.
        small-q MIXED to a decode-like (small bq, large bkv) tile;
        ``bkv_sz``/``batch_size``/``n_buffer`` are unchanged and a smaller
        ``bq_sz`` only lowers VMEM, so there is no OOM risk. A no-op when
        ``bq_sz`` already fits the q count (e.g. real prefill).
        """
        cap = utils.align_to(max(int(total_q_tokens), 1), 8)
        if cap >= self.bq_sz:
            return self
        bq_c_sz = min(self.bq_c_sz, cap)
        while cap % bq_c_sz != 0:
            bq_c_sz -= 1
        return dataclasses.replace(self, bq_sz=cap, bq_c_sz=bq_c_sz)


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
    """Model config that will always stay constant."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    mask_value: float
    sm_scale: float = 1.0
    soft_cap: float | None = None
    sliding_window: int | None = None

    @property
    def num_q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads


class KVLayout(enum.StrEnum):
    """Represents the different layouts for KV cache.

    - HEAD_ALONG_SUBLANE: Number of heads on sublane, head_dim on lane.
    - SEQ_ALONG_LANE: Sequence is packed along the lane, head_dim on sublane.
    """

    HEAD_ALONG_SUBLANE = enum.auto()
    SEQ_ALONG_LANE = enum.auto()


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
    """Serving config that can change depending on use cases."""

    num_seqs: int
    page_size: int
    total_q_tokens: int
    num_page_indices: int
    dtype_q: jnp.dtype
    dtype_kv: jnp.dtype
    dtype_out: jnp.dtype
    scale_q: int | None = None
    scale_k: int | None = None
    scale_v: int | None = None
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE

    @property
    def pages_per_seq(self) -> int:
        return self.num_page_indices // self.num_seqs

    @property
    def page_size_log2(self) -> int:
        return (self.page_size - 1).bit_length()

    @property
    def page_size_mask(self) -> int:
        return self.page_size - 1

    @property
    def int_ty(self) -> jnp.dtype:
        if utils.get_dtype_packing(self.dtype_q) == 1:
            return jnp.int32

        # Absolute position/length arithmetic (kv_len, processed_kv_len, the
        # position iotas in the flash mask) is performed in int_ty. Signed int16
        # only represents positions in [-32768, 32767], so a context that can
        # reach >= 32768 tokens (max_model_len) overflows int16 and corrupts the
        # attention mask -> zero/garbage output (catastrophic at exactly 32768).
        # Fall back to int32 whenever the context can exceed the int16 range.
        max_model_len = self.pages_per_seq * self.page_size
        if max_model_len > 32767:
            return jnp.int32

        match pltpu.get_tpu_info().generation:
            case 6 | 7:
                return jnp.int16
            case _:
                return jnp.int32

    @property
    def packing_q(self) -> int:
        return utils.get_dtype_packing(self.dtype_q)

    @property
    def packing_kv(self) -> int:
        return utils.get_dtype_packing(self.dtype_kv)


class RpaCase(enum.StrEnum):
    """Represents the different cases for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = enum.auto()
    PREFILL = enum.auto()
    MIXED = enum.auto()

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(
        self, distribution: jax.Array
    ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
        assert distribution.shape == (3, )
        match self:
            case RpaCase.DECODE:
                return 0, distribution[0]
            case RpaCase.PREFILL:
                return distribution[0], distribution[1]
            case RpaCase.MIXED:
                return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True, eq=True)
class RpaConfigs:
    block: BlockSizes
    model: ModelConfigs
    serve: ServingConfigs
    mode: RpaCase
    vmem_limit_bytes: int
    has_visibility: bool = False
    update_kv_cache: bool = True
    disable_skip_mask: bool = False
    # DECODE-region static q-block length (= num_speculative_tokens + 1). Only
    # meaningful when ``mode == DECODE``; MIXED/PREFILL always keep 1. Default 1
    # reproduces the single-token decode path byte-for-byte.
    decode_q_len: int = 1
    # Dense cross-step packing for the stacked schedule. When True, requests are
    # packed contiguously across the n cells (wrapping over step boundaries,
    # ~100% cell utilization) with a single cross-step carry slot and a span==1
    # combine-free fast path. Enabled for the DECODE region (opt-in) and for the
    # MIXED region when the MIXED-stacked opt-in is set (see wrapper._make_cfgs);
    # drives ``is_stacked`` for MIXED.
    dense_pack: bool = False
    # Window-anchored sliding-window DECODE. When True (and mode==DECODE with a
    # sliding_window), the schedule anchors a single KV block at the page-aligned
    # window start and the consume mask uses that same anchor, instead of the
    # global-grid ``k_idx*bkv_sz`` skip. Gated by STACKED_RPA_SW_BOUND (default on).
    # When False, behavior is byte-for-byte the current global-grid path.
    sw_bound: bool = True

    @property
    def use_window_anchor(self) -> bool:
        """True iff window-anchored sliding-window decode is active."""
        return (self.mode == RpaCase.DECODE
                and self.model.sliding_window is not None and self.sw_bound)

    # Expose block sizes for ease of use.
    @property
    def bq_sz(self) -> int:
        return self.block.bq_sz

    @property
    def bq_c_sz(self) -> int:
        return self.block.bq_c_sz

    @property
    def bkv_sz(self) -> int:
        return self.block.bkv_sz

    @property
    def batch_size(self) -> int:
        return self.block.batch_size

    @property
    def n_buffer(self) -> int:
        return self.block.n_buffer

    # Define derived values.

    @property
    def max_steps_ub(self) -> int:
        """Get maximum upper bound of kernel steps based on SMEM limit."""

        fixed_bytes = 0
        fixed_bytes += self.serve.num_seqs  # kv_lens
        fixed_bytes += self.serve.num_seqs + 1  # cu_q_lens
        # page_indices is no longer resident in the consume kernel (its lookup is
        # folded into the build, which streams one sequence's slice at a time). The
        # build holds a single per-seq page-table slice in SMEM instead.
        fixed_bytes += self.seq_page_table_size  # seq_page_table (build, 128-padded)
        fixed_bytes += 3  # distribution
        fixed_bytes += self.block.batch_size  # lane_lengths
        fixed_bytes += 1  # actual_steps
        if self.is_stacked:
            fixed_bytes += self.serve.num_seqs  # sorted_seq_idx (scalar prefetch)

        word_size_bytes = 4
        fixed_bytes *= word_size_bytes

        # Reserve for SMEM the schedule window does NOT account for: the pipeline
        # / BufferedRef semaphores + bookkeeping, kv_cache_init, and (on the
        # long-context double-buffered path) the SECOND streamed schedule window.
        # The old 32K reserve only covered the single-window case, so at >=256k
        # (where the double-buffered path kicks in) the window it sized overflowed
        # SMEM by ~9K. Measured non-schedule SMEM is ~214K; reserve 256K so the
        # window always fits and long contexts (up to 1M+) stream in more windows.
        smem_reserve_bytes = 256 * 1024
        smem_limit_bytes = pltpu.get_tpu_info(
        ).smem_capacity_bytes - smem_reserve_bytes
        available_bytes = smem_limit_bytes - fixed_bytes

        # Per step per batch item:
        # s_idx, q_idx, k_idx, is_last_k, do_writeback, skip_mask, combine_span,
        #   is_final: 8 * 4 = 32
        # dma_q: 2 * 4 = 8
        # dma_kv_cache: bkv_p_cache * 3 * 4 = 12 * bkv_p_cache
        # dma_kv_new: bkv_p_new * self.dma_kv_new_size * 4
        bytes_per_step = (40 + 12 * self.bkv_p_cache +
                          4 * self.dma_kv_new_size * self.bkv_p_new)
        bytes_per_step *= self.block.batch_size

        max_steps_ub = available_bytes // bytes_per_step

        num_lanes = pltpu.get_tpu_info().num_lanes
        max_steps_ub = max(1, max_steps_ub // num_lanes) * num_lanes
        return max_steps_ub

    @property
    def _worst_steps(self) -> int:
        """Raw worst-case per-lane step count (independent of window size).

        Total (k-block) work items across all lanes: decode ~ num_seqs*num_k;
        prefill/mixed ~ num_q*num_k. The grid runs PER-LANE steps, and the
        scheduler load-balances across batch_size lanes, so the per-lane step
        count is ~ total / batch_size. Add num_k slack for load-imbalance
        granularity (a single q-block's k-range can land wholly on one lane).

        Stacked/dense packing is contiguous (steps ~ ceil(total_work / n), only
        the final step partial), so it needs less slack than the per-unit
        rounding term below; the +num_seqs term is therefore a safe over-estimate
        for the MIXED-stacked path too (whose units are the (seq, q-block)
        blocks already counted by num_q*num_k).
        """
        max_model_len = self.serve.pages_per_seq * self.serve.page_size
        num_k = -(-max_model_len // self.block.bkv_sz)  # ceil
        num_q = -(-self.serve.total_q_tokens // self.block.bq_sz)  # ceil
        total_work = max(self.serve.num_seqs * num_k, num_q * num_k)
        worst = -(-total_work // self.block.batch_size)  # ceil per-lane
        if self.is_stacked:
            # Stacked steps = sum_i ceil(num_k_i / n) <= total_work/n + num_seqs
            # (each seq's tail step is not shared), so add per-seq rounding slack.
            worst += self.serve.num_seqs
        worst += num_k + self.block.n_buffer + 1
        return worst

    @property
    def fits_one_window(self) -> bool:
        """Whether the whole schedule fits in one SMEM window.

        When True the consume kernel uses the single-window path (one clamped
        schedule copy overlapped with kv_cache_init). When False it uses the
        double-buffered multi-window path. Static (compile-time).
        """
        return self._worst_steps <= self.max_steps_ub

    @property
    def sched_window(self) -> int:
        """Step count per HBM<->SMEM schedule-streaming window.

        Fits: one full SMEM window (== max_steps_ub). Otherwise the window is
        HALVED so TWO windows fit in SMEM, enabling double-buffered streaming
        (prefetch window w+1 while computing window w). Kept a multiple of
        num_lanes (the 1-D DMA leaf tile).
        """
        if self.fits_one_window:
            return self.max_steps_ub
        return max(1, self.max_steps_ub // 2)

    @property
    def num_sched_windows(self) -> int:
        """Number of sched_window-sized windows covering the worst-case schedule.

        1 in the fits case; >= 3 otherwise (since sched_window ~= max_steps_ub/2
        and the schedule exceeds max_steps_ub).
        """
        return max(1, -(-self._worst_steps // self.sched_window))

    @property
    def total_steps_ub(self) -> int:
        """Upper bound on TOTAL kernel steps, used to size the HBM schedule.

        The per-step DMA schedule is streamed window-by-window (window size =
        ``sched_window``) from HBM into SMEM, so the SMEM scratch stays bounded
        by one (or two, double-buffered) window while the HBM-resident schedule
        must hold the worst-case step count. Rounded up to a whole number of
        windows. HBM is cheap, so this is intentionally a generous over-estimate.
        """
        return self.num_sched_windows * self.sched_window

    @property
    def seq_page_table_size(self) -> int:
        """SMEM size of the per-seq page-table slice in the build kernel.

        page_indices is streamed one sequence's slice (pages_per_seq entries) at a
        time, laid out 1-D and sliced at offset ``s_idx * seq_page_table_size``.
        The 1-D int32 DMA tile is 1024, so this must be a multiple of 1024 for both
        the slice offset and size to be tile-aligned for any (dynamic) s_idx.
        """
        return -(-self.serve.pages_per_seq // 1024) * 1024

    @property
    def bkv_p(self) -> int:
        return self.block.bkv_sz // self.serve.page_size

    @property
    def bkv_p_cache(self) -> int:
        if self.mode == RpaCase.PREFILL:
            return 0
        if not self.update_kv_cache:
            return self.bkv_p
        return self.bkv_p

    @property
    def bkv_p_new(self) -> int:
        if not self.update_kv_cache:
            return 1
        if self.mode == RpaCase.DECODE:
            if self.decode_q_len == 1:
                # Single-token decode: the one new token lands in the boundary page.
                return 1
            # Multi-token decode (spec decode): the decode_q_len new tokens sit at
            # the cache boundary and span at most ceil(decode_q_len/page)+1 pages,
            # filled by a BOUNDED ANCHORED loop (see schedule.py). This must stay
            # small and independent of bkv: sizing it like MIXED (bkv_p+1) makes
            # the per-step ``dma_kv_new`` SMEM leaf scale with the (large) decode
            # bkv and can overflow SMEM for large tuned decode blocks.
            return -(-self.decode_q_len // self.serve.page_size) + 1
        return self.bkv_p + 1

    @property
    def bkv_stride(self) -> int:
        bkv_stride = pl.cdiv(self.model.num_kv_heads * 2,
                             self.serve.packing_kv)

        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        return bkv_stride

    @property
    def aligned_q_head_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return utils.align_to(self.model.head_dim, num_lanes)

    @property
    def aligned_kv_head_dim(self) -> int:
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
        return utils.align_to(self.model.head_dim, num_sublanes * kv_packing)

    @property
    def aligned_num_kv_heads_x2(self) -> int:
        packing_kv = self.serve.packing_kv
        return utils.align_to(self.model.num_kv_heads * 2, packing_kv)

    @property
    def aligned_num_q_heads_per_kv_head(self) -> int:
        packing_q = self.serve.packing_q
        return utils.align_to(self.model.num_q_heads_per_kv_head, packing_q)

    @property
    def kv_hbm_stride(self) -> int:
        return self.model.num_kv_heads * 2

    @property
    def fuse_accum(self) -> bool:
        return self.mode == RpaCase.DECODE

    @property
    def is_stacked(self) -> bool:
        """Whether the transposed stacked schedule is active.

        The DECODE region always uses the stacked schedule. The MIXED region
        uses it only when the dense-packing opt-in is set (``dense_pack``);
        otherwise MIXED/PREFILL use the original data-parallel schedule.
        """
        if self.mode == RpaCase.DECODE:
            return True
        if self.mode == RpaCase.MIXED and self.dense_pack:
            return True
        return False

    @property
    def q_vmem_shape(self):
        q_per_kv_packing = self.aligned_num_q_heads_per_kv_head // self.serve.packing_q
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz,
            q_per_kv_packing,
            self.serve.packing_q,
            self.aligned_q_head_dim,
        )

    @property
    def kv_vmem_shape(self):
        # 4D [.., head_dim, tokens] SEQ_ALONG_LANE layout: head_dim is contiguous
        # so the matmul reads K/V in its native packed layout (no relayout) and
        # the V-load can overlap the softmax.
        return (
            self.block.batch_size,
            self.model.num_kv_heads * 2,
            self.aligned_kv_head_dim,
            self.block.bkv_sz + 2 * self.serve.page_size,
        )

    @property
    def dma_kv_new_size(self) -> int:
        return 5

    @property
    def lm_scratch_shape(self):
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            num_lanes,
        )

    @property
    def acc_scratch_shape(self):
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            self.aligned_kv_head_dim,
        )

    def validate_inputs(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        visibility: jax.Array | None = None,
    ):
        """Validate inputs to the RPA kernel statically."""

        if not q.ndim == k.ndim == v.ndim == 3:
            raise ValueError(
                f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
        if k.shape != v.shape:
            raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
        if not (q.shape[0] == k.shape[0] == v.shape[0]):
            raise ValueError(
                "Expected number of sequences in Q, K, and V to be the same, but got"
                f" {q.shape[0]=}, {k.shape[0]=}, and {v.shape[0]=}")
        if not (q.shape[2] == k.shape[2] == v.shape[2]):
            raise ValueError(
                "Expected number of head dimensions in Q, K, and V to be the same,"
                f" but got {q.shape[2]=}, {k.shape[2]=}, and {v.shape[2]=}")

        # page_size must be a positive multiple of 128 (the lane count).
        # page_size>128 (e.g. 512) cuts the KV-DMA descriptor count by
        # page_size/128, improving DMA/compute overlap at long context; the
        # boundary stitch (_stitch_decode_lane) indexes the new-KV in
        # 128-lane-chunk units so it stays correct for any such page_size.
        if self.serve.page_size <= 0 or self.serve.page_size % 128 != 0:
            raise ValueError(
                "Expected page_size to be a positive multiple of 128 (the "
                "lane count) for SEQ_ALONG_LANE tile alignment, but got "
                f"{self.serve.page_size=}")
        expected_kv_cache_shape = (
            kv_cache.shape[0],
            self.model.num_kv_heads * 2,
            self.aligned_kv_head_dim,
            self.serve.page_size,
        )

        if kv_cache.shape != expected_kv_cache_shape:
            raise ValueError(f"Expected {kv_cache.shape=} to be equal to"
                             f" {expected_kv_cache_shape=}")

        # Integer kv quantization is currently not supported.
        if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
            raise ValueError(
                f"Expected {kv_cache.dtype=} to be a floating point.")
        if not (kv_cache.dtype == k.dtype == v.dtype):
            raise ValueError(
                "Expected KV cache dtype and K/V dtype to be the same, but got"
                f" {kv_cache.dtype=}, {k.dtype=}, and {v.dtype=}")

        if not (jnp.int32 == kv_lens.dtype == page_indices.dtype ==
                cu_q_lens.dtype == distribution.dtype):
            raise ValueError(
                f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
                f" {cu_q_lens.dtype=}, {distribution.dtype=}")

        if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
            raise ValueError(
                f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
                f" {cu_q_lens.shape=}")

        max_num_seqs = kv_lens.shape[0]
        num_page_indices = page_indices.shape[0]
        if num_page_indices % max_num_seqs != 0:
            raise ValueError(
                f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
            )
        if cu_q_lens.shape != (max_num_seqs + 1, ):
            raise ValueError(
                f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
        if distribution.shape != (3, ):
            raise ValueError(f"Expected {distribution.shape=} to be (3,).")

        if visibility is not None:
            if visibility.shape != (q.shape[0], 2):
                raise ValueError(
                    f"Expected {visibility.shape=} to be ({q.shape[0]}, 2).")
            if visibility.dtype != jnp.int32:
                raise ValueError(
                    f"Expected {visibility.dtype=} to be {jnp.int32=}.")
            if self.model.sliding_window is not None:
                raise ValueError(
                    "visibility and sliding_window are mutually exclusive.")
