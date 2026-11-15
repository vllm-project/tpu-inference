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
"""Prefill and MIXED block-size selection for stacked RPA."""

import dataclasses

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils
from tpu_inference.kernels.experimental.stacked_rpa.utils import align_to, cdiv

VMEM_BUDGET_FRACTION = 0.8
NUM_BUFFERS = 2
# Calibrated inner-loop budget in batch * Q rows * KV tiles * head tiles * heads.
_COMPUTE_BLOCK_WORK_BUDGET = 2048


def _largest_divisor_at_most(value: int, limit: int) -> int:
    return next(divisor for divisor in range(min(value, limit), 0, -1)
                if value % divisor == 0)


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Prefill and mixed-mode RPA tuning parameters."""

    bq_sz: int
    bq_c_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int

    def __post_init__(self):
        if self.n_buffer != NUM_BUFFERS:
            raise ValueError(
                "Stacked RPA prefill requires double buffering, got "
                f"n_buffer={self.n_buffer}.")

    def cap_bq_to_total_q(self, total_q_tokens: int) -> "BlockSizes":
        """Return a copy with ``bq_sz`` capped at the aligned query count."""
        cap = utils.align_to(max(int(total_q_tokens), 1), 8)
        if cap >= self.bq_sz:
            return self
        bq_c_sz = _largest_divisor_at_most(cap, self.bq_c_sz)
        return dataclasses.replace(self, bq_sz=cap, bq_c_sz=bq_c_sz)


def calculate_vmem_usage(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    block: BlockSizes,
) -> int:
    """Estimate the VMEM footprint of a prefill or mixed tile."""
    tpu_info = pltpu.get_tpu_info()
    aligned_q_head_dim = align_to(model.head_dim, tpu_info.num_lanes)
    aligned_kv_head_dim = align_to(
        model.head_dim,
        tpu_info.num_sublanes * serve.packing_kv,
    )
    aligned_q_per_kv = align_to(
        model.num_q_heads_per_kv_head,
        serve.packing_q,
    )
    aligned_q_heads = aligned_q_per_kv * model.num_kv_heads

    q_bytes = jnp.dtype(serve.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve.dtype_out).itemsize
    accum_bytes = jnp.dtype(configs.accum_dtype(serve.dtype_out)).itemsize

    q_elements = block.bq_sz * aligned_q_heads * aligned_q_head_dim
    kv_elements = ((block.bkv_sz + 2 * serve.page_size) *
                   (model.num_kv_heads * 2) * aligned_kv_head_dim)
    buffer_bytes = (q_elements * q_bytes * block.n_buffer +
                    kv_elements * kv_bytes * block.n_buffer +
                    q_elements * out_bytes * 2)

    loaded_q_bytes = block.bq_sz * model.num_q_heads * aligned_q_head_dim * q_bytes
    loaded_kv_bytes = block.bkv_sz * model.num_kv_heads * aligned_kv_head_dim * kv_bytes
    qk_bytes = block.bq_sz * block.bkv_sz * model.num_q_heads * accum_bytes
    compute_bytes = loaded_q_bytes + loaded_kv_bytes + qk_bytes
    return (buffer_bytes + compute_bytes) * block.batch_size


def _choose_bq_chunk_size(
    model: configs.ModelConfigs,
    block: BlockSizes,
    mxu_column_size: int,
) -> int:
    """Choose the largest divisor of BQ within the compute-work budget."""
    work_per_query = (block.batch_size * cdiv(block.bkv_sz, mxu_column_size) *
                      cdiv(model.head_dim, mxu_column_size) *
                      model.num_q_heads)
    max_chunk_size = max(_COMPUTE_BLOCK_WORK_BUDGET // work_per_query, 1)
    return _largest_divisor_at_most(block.bq_sz, max_chunk_size)


def _choose_heuristic_block_sizes(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> BlockSizes:
    """Grow a balanced, page-aligned tile that fits the VMEM budget."""
    mxu_column_size = pltpu.get_tpu_info().mxu_column_size
    budget = vmem_limit_bytes * VMEM_BUDGET_FRACTION
    max_bkv_sz = serve.pages_per_seq * serve.page_size
    if max_bkv_sz <= 0:
        raise ValueError("Stacked RPA prefill requires at least one KV page.")

    def make_block(bq_sz: int, batch_size: int) -> BlockSizes:
        bkv_sz = min(align_to(bq_sz, serve.page_size), max_bkv_sz)
        return BlockSizes(bq_sz, bq_sz, bkv_sz, batch_size, NUM_BUFFERS)

    def fits(block: BlockSizes) -> bool:
        return calculate_vmem_usage(model, serve, block) <= budget

    for batch_size in (2, 1):
        block = make_block(mxu_column_size, batch_size)
        if fits(block):
            break
    else:
        raise ValueError(
            "Cannot fit the minimum double-buffered prefill tile within the VMEM "
            "limit.")

    while fits(candidate := make_block(
            block.bq_sz + mxu_column_size,
            block.batch_size,
    )):
        block = candidate

    return dataclasses.replace(
        block,
        bq_c_sz=_choose_bq_chunk_size(model, block, mxu_column_size),
    )


def choose_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    *,
    override: BlockSizes | None = None,
) -> BlockSizes:
    """Choose an explicit or VMEM-heuristic prefill block size."""
    if override is not None:
        effective = override
    else:
        effective = _choose_heuristic_block_sizes(
            model_cfgs,
            serve_cfgs,
            vmem_limit_bytes,
        ).cap_bq_to_total_q(serve_cfgs.total_q_tokens)
    aligned_bkv = max(
        serve_cfgs.page_size,
        align_to(effective.bkv_sz, serve_cfgs.page_size),
    )
    if aligned_bkv == effective.bkv_sz:
        return effective
    return dataclasses.replace(effective, bkv_sz=aligned_bkv)
