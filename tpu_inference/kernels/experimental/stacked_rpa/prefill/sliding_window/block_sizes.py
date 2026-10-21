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
"""Block-size policy for one-KV-block sliding-window prefill."""

import dataclasses

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs
from tpu_inference.kernels.experimental.stacked_rpa.prefill import \
    block_sizes as prefill_block_sizes
from tpu_inference.kernels.experimental.stacked_rpa.utils import align_to

VMEM_BUDGET_FRACTION = 0.8
_BQ_ALIGNMENT = 8
_MAX_BQ_SIZE = 256


def required_window_bkv_size(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    bq_sz: int,
) -> int:
    """Return the page-aligned DMA span covering any Q block's window."""
    if model.sliding_window is None:
        raise ValueError(
            "Sliding-window prefill requires model.sliding_window.")
    if bq_sz < 1:
        raise ValueError(f"bq_sz must be >= 1, got {bq_sz}.")
    if serve.pages_per_seq < 1:
        raise ValueError(
            "Sliding-window prefill requires at least one KV page.")

    max_model_len = serve.pages_per_seq * serve.page_size
    anchored_window = align_to(
        model.sliding_window + bq_sz + serve.page_size - 1,
        serve.page_size,
    )
    return min(max_model_len, anchored_window)


def compute_window_bkv_size(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    block: prefill_block_sizes.BlockSizes,
) -> int:
    """Return the lane-aligned subset consumed by QK/PV after page DMA."""
    if model.sliding_window is None:
        raise ValueError(
            "Sliding-window prefill requires model.sliding_window.")
    num_lanes = pltpu.get_tpu_info().num_lanes
    compute_window = align_to(
        model.sliding_window + block.bq_sz + num_lanes - 1,
        num_lanes,
    )
    max_model_len = serve.pages_per_seq * serve.page_size
    return min(block.bkv_sz, max_model_len, compute_window)


def calculate_vmem_usage(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    block: prefill_block_sizes.BlockSizes,
) -> int:
    """Estimate the scratch-free pipeline and largest chunk's live values."""
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
    compute_bkv = compute_window_bkv_size(model, serve, block)

    q_bytes = jnp.dtype(serve.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve.dtype_out).itemsize
    accum_bytes = jnp.dtype(configs.accum_dtype(serve.dtype_out)).itemsize

    q_elements = block.bq_sz * aligned_q_heads * aligned_q_head_dim
    kv_elements = ((block.bkv_sz + 2 * serve.page_size) *
                   (model.num_kv_heads * 2) * aligned_kv_head_dim)
    buffered_bytes = (q_elements * q_bytes * block.n_buffer +
                      kv_elements * kv_bytes * block.n_buffer +
                      q_elements * out_bytes * 2)

    chunk_q_elements = block.bq_c_sz * aligned_q_heads * aligned_q_head_dim
    loaded_kv_elements = compute_bkv * model.num_kv_heads * aligned_kv_head_dim
    probability_elements = block.bq_c_sz * compute_bkv * aligned_q_heads
    statistic_elements = block.bq_c_sz * aligned_q_heads * tpu_info.num_lanes
    live_bytes = (chunk_q_elements * q_bytes +
                  2 * loaded_kv_elements * kv_bytes +
                  probability_elements * accum_bytes +
                  statistic_elements * accum_bytes +
                  chunk_q_elements * accum_bytes)
    return (buffered_bytes + live_bytes) * block.batch_size


def _chunk_size_for_bq(seed_chunk: int, bq_sz: int) -> int:
    chunk = min(seed_chunk, bq_sz)
    while bq_sz % chunk:
        chunk -= 1
    return chunk


def choose_block_sizes(
    model: configs.ModelConfigs,
    serve: configs.ServingConfigs,
    vmem_limit_bytes: int,
    *,
    override: prefill_block_sizes.BlockSizes | None = None,
) -> prefill_block_sizes.BlockSizes:
    """Choose a double-buffered Q tile with one anchored KV block per cell."""
    if model.sliding_window is None:
        raise ValueError(
            "Sliding-window prefill requires model.sliding_window.")

    seed = prefill_block_sizes.choose_block_sizes(
        model,
        serve,
        vmem_limit_bytes,
        override=override,
    )
    for name in ("bq_sz", "bq_c_sz", "batch_size"):
        if getattr(seed, name) < 1:
            raise ValueError(
                f"Sliding-window prefill requires positive {name}.")
    max_bq = align_to(
        max(min(seed.bq_sz, serve.total_q_tokens, _MAX_BQ_SIZE), 1),
        _BQ_ALIGNMENT,
    )
    budget = vmem_limit_bytes * VMEM_BUDGET_FRACTION

    # Preserve the seed concurrency while shrinking BQ if the anchored tile does
    # not fit. Capping the Q block also bounds runtime-loop work and ragged-tail
    # padding for the dynamic serving batches handled by this path.
    for batch_size in range(seed.batch_size, 0, -1):
        for bq_sz in range(max_bq, _BQ_ALIGNMENT - 1, -_BQ_ALIGNMENT):
            bq_c_sz = _chunk_size_for_bq(seed.bq_c_sz, bq_sz)
            candidate = dataclasses.replace(
                seed,
                bq_sz=bq_sz,
                bq_c_sz=bq_c_sz,
                bkv_sz=required_window_bkv_size(model, serve, bq_sz),
                batch_size=batch_size,
                n_buffer=prefill_block_sizes.NUM_BUFFERS,
            )
            if calculate_vmem_usage(model, serve, candidate) <= budget:
                return candidate

    raise ValueError(
        "The one-block sliding-window prefill tile does not fit in VMEM: "
        f"{model.sliding_window=} {vmem_limit_bytes=} {seed=}.")
