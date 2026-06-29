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

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
import jax.experimental.pallas as pl

from tpu_inference.kernels.experimental.batched_rpa import (configs, utils)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuningKey:
    case: Literal['decode', 'prefill']
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    # serve config parameters
    num_seqs: int
    page_size: int
    total_q_tokens: int
    num_page_indices: int
    dtype_q: str
    dtype_kv: str
    dtype_out: str
    scale_q: int | None = None
    scale_k: int | None = None
    scale_v: int | None = None

    # model config default params
    sliding_window: int | None = None

    @staticmethod
    def from_config(model_config: configs.ModelConfigs,
                    serve_config: configs.ServingConfigs,
                    case: Literal['decode', 'prefill']) -> 'TuningKey':
        return TuningKey(num_q_heads=model_config.num_q_heads,
                         num_kv_heads=model_config.num_kv_heads,
                         head_dim=model_config.head_dim,
                         sliding_window=model_config.sliding_window,
                         num_seqs=serve_config.num_seqs,
                         page_size=serve_config.page_size,
                         total_q_tokens=serve_config.total_q_tokens,
                         num_page_indices=serve_config.num_page_indices,
                         dtype_q=jnp.dtype(serve_config.dtype_q).name,
                         dtype_kv=jnp.dtype(serve_config.dtype_kv).name,
                         dtype_out=jnp.dtype(serve_config.dtype_out).name,
                         scale_q=serve_config.scale_q,
                         scale_k=serve_config.scale_k,
                         scale_v=serve_config.scale_v,
                         case=case)


@dataclass(frozen=True)
class TunableParams:
    """Tuning parameters for the RPA kernel."""
    bq_sz: int
    bq_c_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int
    is_baseline: bool = False

    def to_block_sizes(self) -> configs.BlockSizes:
        return configs.BlockSizes(bq_sz=self.bq_sz,
                                  bq_c_sz=self.bq_c_sz,
                                  bkv_sz=self.bkv_sz,
                                  batch_size=self.batch_size,
                                  n_buffer=self.n_buffer)

    # Define comparison operators for skipping tuning case when smaller block sizes hit OOM already
    def __ge__(self, other) -> bool:
        return self.bq_sz >= other.bq_sz and self.bq_c_sz >= other.bq_c_sz and self.bkv_sz >= other.bkv_sz and self.batch_size >= other.batch_size and self.n_buffer >= other.n_buffer

    def __le__(self, other) -> bool:
        return self.bq_sz <= other.bq_sz and self.bq_c_sz <= other.bq_c_sz and self.bkv_sz <= other.bkv_sz and self.batch_size <= other.batch_size and self.n_buffer <= other.n_buffer


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    # Calculate aligned model dimensions.
    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q)
    aligned_num_q_heads = (aligned_num_q_heads_per_kv_head *
                           model_cfgs.num_kv_heads)

    bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2, serve_cfgs.packing_kv)
    if utils.has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize

    def calculate_vmem_usage(batch_size: int, n_buffer: int, bq_sz: int,
                             bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        bkv_array_size = bkv_sz * aligned_num_kv_heads_x2 * aligned_head_dim

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * q_bytes
        bkv_bytes = bkv_array_size * kv_bytes
        bo_bytes = bo_array_size * out_bytes

        # Account for multiple buffers. For output, we always use double buffer.
        bq_bytes *= n_buffer
        bkv_bytes *= n_buffer
        bo_bytes *= 2

        # Sum up all buffer memory usage.
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

        # Step 2: Calculate worst case memory usage during computation.

        # Calculate the size of loaded bq and bkv size.
        loaded_bq_size = bq_sz * model_cfgs.num_q_heads * aligned_head_dim
        loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim

        # Calculate peak memory requirement of output - which is attention weight.
        qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

        # Convert to bytes.
        loaded_bq_bytes = loaded_bq_size * q_bytes
        loaded_bkv_bytes = loaded_bkv_size * kv_bytes
        qk_bytes = qk_size * out_bytes

        # Sum up all compute memory usage.
        compute_bytes = loaded_bq_bytes + loaded_bkv_bytes + qk_bytes

        # Step 3: Sum up all memory usage.
        total_bytes = buffer_bytes + compute_bytes

        # Account for batch size.
        total_bytes *= batch_size

        return total_bytes

    def calculate_compute_buffer_time(batch_size: int, bq_c_sz: int,
                                      bkv_sz: int) -> int:
        """Calculate computational complexity of a single compute block."""

        num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    def find_best_block_sizes(
            max_batch_size: int,
            max_n_buffer: int,
            fixed_bq_sz: int | None = None) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. Therefore, we conservatively only use 80% of the VMEM budget.
        capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size
        n_buffer = max_n_buffer

        # Step 1: Lower batch_size and/or n_buffer if even the smallest bq and bkv
        # size can trigger OOM.

        # If current batch size triggers OOM, decrease batch size until the kernel
        # fits within VMEM limit.
        while (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               > capped_vmem_limit_bytes):
            batch_size -= 1

        # As a last resort, attempt to decrease number of buffers to avoid OOM.
        while (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               > capped_vmem_limit_bytes):
            n_buffer -= 1

        # Indicates OOM was triggered even when batch_size=1 or n_buffer=1.
        # NOTE: If the function does not exit at this point even when either values
        # are zero, it will trigger infinite loop at the next while loop.
        if batch_size == 0 or n_buffer == 0:
            raise ValueError(
                "Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        while (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               < capped_vmem_limit_bytes):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        bkv_sz -= bkv_stride
        bq_sz -= bq_stride

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError(
                "Cannot find block sizes that fit within VMEM limit.")

        # Step 3: Given current tile size, calculate compute tile size.

        # Fixed threshold value based on hardware spec.
        # TODO(kyuyeunk): Use different threshold based on hardware and precision.
        threshold = 1500

        num_bq_c = 1
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz)
               > threshold or bq_c_rem != 0) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=last_valid_bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    # Default to triple buffer as its almost always beneficial.
    n_buffer = 3
    # Fixed value based on experimental results.
    decode_batch_size = 8
    prefill_batch_size = 2

    decode_block_sizes = find_best_block_sizes(decode_batch_size, n_buffer, 1)
    prefill_block_sizes = find_best_block_sizes(prefill_batch_size, n_buffer)

    return decode_block_sizes, prefill_block_sizes


def get_tuned_params(
        model_config: configs.ModelConfigs,
        serve_config: configs.ServingConfigs,
        vmem_limit_bytes: int | None = None,
        case: Literal['decode', 'prefill'] = 'decode') -> configs.BlockSizes:
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    tuning_key = TuningKey.from_config(model_config, serve_config, case=case)
    if tuning_key not in tuned_params_mapping:
        decode_block_sizes, prefill_block_sizes = calculate_block_sizes(
            model_config, serve_config, vmem_limit_bytes)
        block_sizes = decode_block_sizes if case == 'decode' else prefill_block_sizes
    else:
        block_sizes = tuned_params_mapping[tuning_key].to_block_sizes()
    return block_sizes


# This is a placeholder as we haven't found better tuned block sizes for the cases
tuned_params_mapping: dict[TuningKey, TunableParams] = {}
