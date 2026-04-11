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
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tuning parameters for the RPA kernel."""

    bq_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int


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
class RPAConfig:
    block: BlockSizes
    model: ModelConfigs
    serve: ServingConfigs
    mode: RpaCase
    vmem_limit_bytes: int

    # Expose block sizes for ease of use.

    @property
    def bq_sz(self) -> int:
        return self.block.bq_sz

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
        fixed_bytes += self.serve.num_seqs * self.serve.pages_per_seq  # page_indices
        fixed_bytes += 3  # distribution
        fixed_bytes += self.block.batch_size  # lane_lengths
        fixed_bytes += 1  # actual_steps

        word_size_bytes = 4
        fixed_bytes *= word_size_bytes

        smem_limit_bytes = pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024
        available_bytes = smem_limit_bytes - fixed_bytes

        # Per step per batch item:
        # s_idx, q_idx, k_idx, is_last_k, do_writeback: 5 * 4 = 20
        # dma_q: 2 * 4 = 8
        # dma_kv_cache: bkv_p_cache * 3 * 4 = 12 * bkv_p_cache
        # dma_kv_new: bkv_p_new * 4 * 4 = 16 * bkv_p_new
        bytes_per_step = 28 + 12 * self.bkv_p_cache + 16 * self.bkv_p_new
        bytes_per_step *= self.block.batch_size

        max_steps_ub = available_bytes // bytes_per_step

        num_lanes = pltpu.get_tpu_info().num_lanes
        max_steps_ub = max(1, max_steps_ub // num_lanes) * num_lanes
        return max_steps_ub

    @property
    def bkv_p(self) -> int:
        return self.block.bkv_sz // self.serve.page_size

    @property
    def bkv_p_cache(self) -> int:
        if self.mode == RpaCase.PREFILL:
            return 0
        return self.bkv_p

    @property
    def bkv_p_new(self) -> int:
        if self.mode == RpaCase.DECODE:
            return 1
        return self.bkv_p

    @property
    def bkv_stride(self) -> int:
        bkv_stride = (self.model.num_kv_heads * 2) // self.serve.packing_kv

        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        return bkv_stride

    @property
    def aligned_head_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return utils.align_to(self.model.head_dim, num_lanes)

    @property
    def aligned_num_kv_heads(self) -> int:
        packing_kv = self.serve.packing_kv
        return utils.align_to(self.model.num_kv_heads, packing_kv)

    @property
    def aligned_num_q_heads_per_kv_head(self) -> int:
        packing_q = self.serve.packing_q
        return utils.align_to(self.model.num_q_heads_per_kv_head, packing_q)

    @property
    def kv_hbm_stride(self) -> int:
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
        return utils.align_to(self.model.num_kv_heads * 2,
                              kv_packing) // kv_packing

    @property
    def fuse_accum(self) -> bool:
        return self.mode == RpaCase.DECODE

    @property
    def mask_v(self) -> bool:
        return self.mode != RpaCase.DECODE

    @property
    def q_vmem_shape(self):
        q_per_kv_packing = (self.model.num_q_heads_per_kv_head //
                            self.serve.packing_q)
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz,
            q_per_kv_packing,
            self.serve.packing_q,
            self.model.head_dim,
        )

    @property
    def kv_vmem_shape(self):
        return (
            self.block.batch_size,
            self.block.bkv_sz,
            self.bkv_stride,
            self.serve.packing_kv,
            self.model.head_dim,
        )

    @property
    def lm_scratch_shape(self):
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz * self.model.num_q_heads_per_kv_head,
            num_lanes,
        )

    @property
    def acc_scratch_shape(self):
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz * self.model.num_q_heads_per_kv_head,
            self.model.head_dim,
        )
