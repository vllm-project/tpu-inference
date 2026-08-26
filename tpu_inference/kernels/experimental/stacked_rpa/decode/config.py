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
"""Configuration shared by the concrete stacked-RPA decode kernels."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import \
    configs as common_configs
from tpu_inference.kernels.experimental.stacked_rpa import utils
from tpu_inference.kernels.experimental.stacked_rpa.decode.block_sizes import \
    BlockSizes


@dataclasses.dataclass(frozen=True, eq=True)
class DecodeConfig:
    """Static configuration for the dense stacked decode kernel."""

    block: BlockSizes
    model: common_configs.ModelConfigs
    serve: common_configs.ServingConfigs
    mode: common_configs.RpaCase
    vmem_limit_bytes: int
    update_kv_cache: bool = True
    disable_skip_mask: bool = False
    decode_q_len: int = 1
    dense_pack: bool = True

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

    def validate_decode(self) -> None:
        """Validate invariants shared by every standalone decode kernel."""
        if self.mode != common_configs.RpaCase.DECODE:
            raise ValueError(f"Decode kernel received mode={self.mode!r}.")
        if not self.dense_pack:
            raise ValueError("Decode kernel requires dense_pack=True.")
        if self.decode_q_len < 1:
            raise ValueError(
                f"decode_q_len must be >= 1, got {self.decode_q_len}.")
        if self.bq_sz < self.decode_q_len:
            raise ValueError(
                "Decode requires one query block per sequence, got "
                f"bq_sz={self.bq_sz} < decode_q_len={self.decode_q_len}.")

    @property
    def bkv_p(self) -> int:
        return self.block.bkv_sz // self.serve.page_size

    @property
    def bkv_p_cache(self) -> int:
        return self.bkv_p

    @property
    def bkv_p_new(self) -> int:
        if not self.update_kv_cache or self.decode_q_len == 1:
            return 1
        return -(-self.decode_q_len // self.serve.page_size) + 1

    @property
    def aligned_q_head_dim(self) -> int:
        return utils.align_to(self.model.head_dim,
                              pltpu.get_tpu_info().num_lanes)

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
        return True

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
        return (
            self.block.batch_size,
            self.model.num_kv_heads * 2,
            self.aligned_kv_head_dim,
            self.block.bkv_sz + 2 * self.serve.page_size,
        )

    @property
    def num_lanes(self) -> int:
        return pltpu.get_tpu_info().num_lanes

    @property
    def new_kv_padded_lanes(self) -> int:
        return utils.align_to(self.serve.total_q_tokens, self.num_lanes)

    @property
    def new_kv_resident(self) -> bool:
        return (self.decode_q_len == 1
                and self.new_kv_padded_lanes <= self.serve.page_size)

    @property
    def new_kv_vmem_shape(self):
        lanes = (self.new_kv_padded_lanes
                 if self.new_kv_resident else self.num_lanes)
        return (
            self.model.num_kv_heads * 2,
            self.aligned_kv_head_dim,
            lanes,
        )

    @property
    def wb_lane_bits(self) -> tuple[int, int]:
        width = (self.serve.page_size // self.num_lanes).bit_length()
        return width, (1 << width) - 1

    @property
    def dma_kv_new_size(self) -> int:
        return 5

    @property
    def lm_scratch_shape(self):
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            pltpu.get_tpu_info().num_lanes,
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
    ) -> None:
        """Validate decode inputs statically."""
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
    model: common_configs.ModelConfigs,
    serve: common_configs.ServingConfigs,
    block: BlockSizes,
    vmem_limit_bytes: int,
    *,
    update_kv_cache: bool = True,
    decode_q_len: int = 1,
) -> DecodeConfig:
    """Construct the static configuration shared by both decode kernels."""
    return DecodeConfig(
        block=block,
        model=model,
        serve=serve,
        mode=common_configs.RpaCase.DECODE,
        vmem_limit_bytes=vmem_limit_bytes,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
        dense_pack=True,
    )
