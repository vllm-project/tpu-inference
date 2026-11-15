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
"""Block-size types and policies for stacked-RPA decode."""

import dataclasses
import math

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs
from tpu_inference.kernels.experimental.stacked_rpa.utils import align_to

VMEM_BUDGET_FRACTION = 0.8
_MAX_BATCH_SIZE = 8
NUM_BUFFERS = 2
_LONG_CONTEXT_THRESHOLD = 128 * 1024
_LONG_CONTEXT_BATCH_CANDIDATES = (1, 2, 4, _MAX_BATCH_SIZE)


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Decode tuning parameters for one query block per sequence."""

    bq_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int

    def __post_init__(self):
        if self.n_buffer != NUM_BUFFERS:
            raise ValueError(
                "Stacked RPA decode requires double buffering, got "
                f"n_buffer={self.n_buffer}.")

    def floor_bq_to_decode_q_len(self, decode_q_len: int) -> "BlockSizes":
        floor = max(int(decode_q_len), 1)
        if self.bq_sz >= floor:
            return self
        return dataclasses.replace(self, bq_sz=floor)


def required_window_bkv_size(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    decode_q_len: int,
) -> int:
    """Return the page-aligned KV span covering any anchored decode window."""
    if model.sliding_window is None:
        raise ValueError(
            "Sliding-window decode requires model.sliding_window.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}.")
    if serve.pages_per_seq < 1:
        raise ValueError(
            "Sliding-window decode requires at least one KV page.")

    max_model_len = serve.pages_per_seq * serve.page_size
    anchored_window = align_to(
        model.sliding_window + decode_q_len + serve.page_size - 1,
        serve.page_size,
    )
    return min(max_model_len, anchored_window)


def calculate_vmem_usage(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    block: BlockSizes,
) -> int:
    """Estimate the VMEM footprint of a decode tile."""
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

    compute_bytes = (
        block.bq_sz * model.num_q_heads * aligned_q_head_dim * q_bytes +
        block.bkv_sz * model.num_kv_heads * aligned_kv_head_dim * kv_bytes +
        block.bq_sz * block.bkv_sz * model.num_q_heads * accum_bytes)
    return (buffer_bytes + compute_bytes) * block.batch_size


def _align_bkv_to_page(block: BlockSizes, page_size: int) -> BlockSizes:
    aligned = max(page_size, align_to(block.bkv_sz, page_size))
    if aligned == block.bkv_sz:
        return block
    return dataclasses.replace(block, bkv_sz=aligned)


def choose_block_sizes(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_q_len: int,
) -> BlockSizes:
    """Choose a page-aligned KV tile for the configured context capacity.

    Short contexts retain the full-concurrency policy. At long context, score
    every supported double-buffered concurrency by useful KV tokens streamed
    per step for every decode query length. A lower batch size can fit a much
    wider KV tile and amortize DMA, softmax, and split-K combine overhead better
    than eight narrow cells.
    """
    bq_sz = max(decode_q_len, 1)
    bkv_stride = math.lcm(pltpu.get_tpu_info().mxu_column_size,
                          serve.page_size)
    max_model_len = serve.pages_per_seq * serve.page_size
    if max_model_len <= 0:
        raise ValueError("Stacked RPA decode requires at least one KV page.")
    max_bkv_sz = max_model_len
    budget = vmem_limit_bytes * VMEM_BUDGET_FRACTION

    def make_block(bkv_sz: int, batch_size: int) -> BlockSizes:
        return BlockSizes(bq_sz, bkv_sz, batch_size, NUM_BUFFERS)

    def fits(block: BlockSizes) -> bool:
        return calculate_vmem_usage(model, serve, block) <= budget

    def largest_fitting_block(batch_size: int) -> BlockSizes | None:
        block = make_block(min(bkv_stride, max_bkv_sz), batch_size)
        if not fits(block):
            return None

        while block.bkv_sz < max_bkv_sz and fits(candidate := make_block(
                min(block.bkv_sz + bkv_stride, max_bkv_sz),
                batch_size,
        )):
            block = candidate
        return block

    if max_model_len < _LONG_CONTEXT_THRESHOLD:
        for batch_size in range(_MAX_BATCH_SIZE, 0, -1):
            if (block := largest_fitting_block(batch_size)) is not None:
                return block

        raise ValueError(
            "Cannot fit the minimum double-buffered decode tile within the VMEM "
            "limit.")

    best_block = None
    best_score = -1
    for batch_size in _LONG_CONTEXT_BATCH_CANDIDATES:
        candidate = largest_fitting_block(batch_size)
        if candidate is None:
            continue
        score = candidate.batch_size * min(candidate.bkv_sz, max_bkv_sz)
        if score >= best_score:
            best_block = candidate
            best_score = score

    if best_block is None:
        raise ValueError(
            "Cannot fit the minimum double-buffered decode tile within the VMEM "
            "limit.")

    return best_block


def choose_non_sliding_window_block_sizes(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_q_len: int = 1,
    *,
    override: BlockSizes | None = None,
) -> BlockSizes:
    """Choose the global-attention decode tile."""
    default = choose_block_sizes(
        model,
        serve,
        vmem_limit_bytes,
        decode_q_len,
    )
    if override is not None:
        effective = override
    else:
        effective = default

    if decode_q_len > 1:
        effective = effective.floor_bq_to_decode_q_len(decode_q_len)
        if calculate_vmem_usage(model, serve,
                                effective) > (vmem_limit_bytes *
                                              VMEM_BUDGET_FRACTION):
            effective = default

    return _align_bkv_to_page(effective, serve.page_size)


def choose_sliding_window_block_sizes(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_q_len: int = 1,
    *,
    override: BlockSizes | None = None,
) -> BlockSizes:
    """Choose one page-aligned, double-buffered sliding-window tile."""
    default = choose_block_sizes(
        model,
        serve,
        vmem_limit_bytes,
        decode_q_len,
    )
    required_bkv = required_window_bkv_size(model, serve, decode_q_len)
    candidate = override or dataclasses.replace(
        default,
        bkv_sz=required_bkv,
        batch_size=_MAX_BATCH_SIZE,
        n_buffer=NUM_BUFFERS,
    )
    candidate = _align_bkv_to_page(
        candidate.floor_bq_to_decode_q_len(decode_q_len),
        serve.page_size,
    )
    if candidate.bkv_sz < required_bkv:
        raise ValueError(
            "Sliding-window decode requires one anchored KV block: "
            f"bkv_sz={candidate.bkv_sz} is smaller than {required_bkv}.")

    budget = vmem_limit_bytes * VMEM_BUDGET_FRACTION
    for batch_size in range(candidate.batch_size, 0, -1):
        effective = dataclasses.replace(
            candidate,
            batch_size=batch_size,
            n_buffer=NUM_BUFFERS,
        )
        if calculate_vmem_usage(model, serve, effective) <= budget:
            return effective

    raise ValueError(
        "The page-aligned sliding-window decode block does not fit in VMEM: "
        f"{candidate=} {vmem_limit_bytes=}.")
