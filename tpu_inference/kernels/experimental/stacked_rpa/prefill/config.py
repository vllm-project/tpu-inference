# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Prefill and mixed-mode configuration for stacked RPA."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import utils
from tpu_inference.kernels.experimental.stacked_rpa.configs import (
    ModelConfigs, RpaCase, ServingConfigs)
from tpu_inference.kernels.experimental.stacked_rpa.prefill.block_sizes import \
    BlockSizes


@dataclasses.dataclass(frozen=True, eq=True)
class PrefillConfig:
    """Static configuration shared by the PREFILL and MIXED kernels."""

    block: BlockSizes
    model: ModelConfigs
    serve: ServingConfigs
    mode: RpaCase
    vmem_limit_bytes: int
    update_kv_cache: bool = True
    disable_skip_mask: bool = False
    dense_pack: bool = False

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

    @property
    def bkv_p(self) -> int:
        return self.bkv_sz // self.serve.page_size

    @property
    def bkv_p_cache(self) -> int:
        if self.mode == RpaCase.PREFILL:
            return 0
        return self.bkv_p

    @property
    def bkv_p_new(self) -> int:
        if not self.update_kv_cache:
            return 1
        return self.bkv_p + 1

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
    def aligned_num_q_heads_per_kv_head(self) -> int:
        return utils.align_to(self.model.num_q_heads_per_kv_head,
                              self.serve.packing_q)

    @property
    def kv_hbm_stride(self) -> int:
        return self.model.num_kv_heads * 2

    @property
    def fuse_accum(self) -> bool:
        return False

    @property
    def is_stacked(self) -> bool:
        return self.mode == RpaCase.MIXED and self.dense_pack

    @property
    def q_vmem_shape(self):
        q_per_kv_packing = self.aligned_num_q_heads_per_kv_head // self.serve.packing_q
        return (
            self.batch_size,
            self.model.num_kv_heads,
            self.bq_sz,
            q_per_kv_packing,
            self.serve.packing_q,
            self.aligned_q_head_dim,
        )

    @property
    def kv_vmem_shape(self):
        return (
            self.batch_size,
            self.model.num_kv_heads * 2,
            self.aligned_kv_head_dim,
            self.bkv_sz + 2 * self.serve.page_size,
        )

    @property
    def num_lanes(self) -> int:
        return pltpu.get_tpu_info().num_lanes

    @property
    def wb_lane_bits(self) -> tuple[int, int]:
        width = (self.serve.page_size // self.num_lanes).bit_length()
        return width, (1 << width) - 1

    @property
    def dma_kv_new_size(self) -> int:
        return 5

    @property
    def lm_scratch_shape(self):
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (
            self.batch_size,
            self.model.num_kv_heads,
            self.bq_sz * self.aligned_num_q_heads_per_kv_head,
            num_lanes,
        )

    @property
    def acc_scratch_shape(self):
        return (
            self.batch_size,
            self.model.num_kv_heads,
            self.bq_sz * self.aligned_num_q_heads_per_kv_head,
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
    ) -> None:
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


def make_config(
    mode: RpaCase,
    model: ModelConfigs,
    serve: ServingConfigs,
    block: BlockSizes,
    vmem_limit_bytes: int,
    *,
    update_kv_cache: bool = True,
    dense_pack: bool = False,
) -> PrefillConfig:
    """Build a static PREFILL or MIXED kernel configuration."""
    if mode not in (RpaCase.PREFILL, RpaCase.MIXED):
        raise ValueError(f"Prefill config builder received {mode=}")
    return PrefillConfig(
        block=block,
        model=model,
        serve=serve,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
        update_kv_cache=update_kv_cache,
        dense_pack=mode == RpaCase.MIXED and dense_pack,
    )
