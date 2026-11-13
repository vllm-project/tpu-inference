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

from dataclasses import asdict, dataclass
from typing import Literal

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs, utils
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

    def to_block_sizes(self) -> configs.BlockSizes:
        return configs.BlockSizes(**asdict(self))

    @staticmethod
    def from_block_sizes(block_sizes: configs.BlockSizes) -> 'TunableParams':
        return TunableParams(**asdict(block_sizes))

    # Define comparison operators for skipping tuning case when smaller block sizes hit OOM already
    def __ge__(self, other) -> bool:
        return self.bq_sz >= other.bq_sz and self.bq_c_sz >= other.bq_c_sz and self.bkv_sz >= other.bkv_sz and self.batch_size >= other.batch_size

    def __le__(self, other) -> bool:
        return self.bq_sz <= other.bq_sz and self.bq_c_sz <= other.bq_c_sz and self.bkv_sz <= other.bkv_sz and self.batch_size <= other.batch_size


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
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

    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_num_kv_heads_x2 = model_cfgs.num_kv_heads * 2
    else:
        bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2,
                             serve_cfgs.packing_kv)
        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_q) / 8
    kv_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_kv) / 8
    out_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_out) / 8

    is_packed_fp4 = serve_cfgs.dtype_kv == jnp.uint8
    physical_kv_head_dim = model_cfgs.head_dim
    if is_packed_fp4:
        physical_kv_head_dim = physical_kv_head_dim // 2

    num_sublanes = tpu_info.num_sublanes
    packing_kv = serve_cfgs.packing_kv

    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_scale_channels = serve_cfgs.scale_channels
        base_dim = physical_kv_head_dim + num_scale_channels
        aligned_kv_head_dim_val = utils.align_to(base_dim,
                                                 num_sublanes * packing_kv)
    else:
        aligned_kv_head_dim_val = utils.align_to(model_cfgs.head_dim,
                                                 num_lanes)

    def calculate_vmem_usage(batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        vmem_bkv_sz = bkv_sz
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # Allocate extra buffer to perform lane stitching. Must match
            # RpaConfigs.kv_vmem_lanes.
            vmem_bkv_sz += 2 * num_lanes
        bkv_array_size = (vmem_bkv_sz * aligned_num_kv_heads_x2 *
                          aligned_kv_head_dim_val)

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * q_bytes
        bkv_bytes = bkv_array_size * kv_bytes
        bo_bytes = bo_array_size * out_bytes

        # Account for multiple buffers. These counts are fixed by the pipeline; see
        # `configs.NUM_INPUT_SLOTS` / `configs.NUM_OUTPUT_SLOTS`.
        bq_bytes *= configs.NUM_INPUT_SLOTS
        bkv_bytes *= configs.NUM_INPUT_SLOTS
        bo_bytes *= configs.NUM_OUTPUT_SLOTS

        # Sum up all buffer memory usage (scales are co-located inside bkv_bytes).
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

        if serve_cfgs.return_lse:
            lse_array_size = bq_sz * aligned_num_q_heads * num_lanes
            lse_bytes = lse_array_size * out_bytes * 2
            buffer_bytes += lse_bytes

        # Step 2: Calculate worst case memory usage during computation.

        # Calculate the size of loaded bq and bkv size.
        loaded_bq_size = bq_sz * model_cfgs.num_q_heads * aligned_head_dim
        loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim

        # Calculate peak memory requirement of otuput - which is attention weight.
        qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

        # Convert to bytes.
        loaded_bq_bytes = loaded_bq_size * q_bytes
        compute_kv_bytes = kv_bytes
        loaded_bkv_bytes = loaded_bkv_size * compute_kv_bytes
        qk_bytes = qk_size * out_bytes

        # Calculate VMEM contribution of active unpacked scale arrays during
        # compute. Note: since flash_attention consumes quantized k/v directly
        # along with our active k_scale and v_scale tensors (no upfront dequantized
        # intermediate copy), we only need to account for unpacked k_scale + v_scale
        # during compute.
        loaded_scale_bytes = 0
        if serve_cfgs.per_token_scale:
            scale_bits = jax.dtypes.itemsize_bits(
                serve_cfgs.per_token_scale_dtype)
            scale_bytes = scale_bits // 8
            # Active k_scale and v_scale each have shape (num_kv_heads, bkv_sz)
            loaded_scale_bytes = 2 * bkv_sz * model_cfgs.num_kv_heads * scale_bytes

        # Sum up all compute memory usage.
        compute_bytes = (loaded_bq_bytes + loaded_bkv_bytes + qk_bytes +
                         loaded_scale_bytes)

        # Step 3: Sum up all memory usage.
        total_bytes = buffer_bytes + compute_bytes

        # Account for batch size.
        total_bytes *= batch_size

        # Online-softmax scratch (m, l, and the output accumulator)
        lm_array_size = (model_cfgs.num_kv_heads * bq_sz *
                         aligned_num_q_heads_per_kv_head)
        acc_array_size = (model_cfgs.num_kv_heads * bq_sz *
                          aligned_num_q_heads_per_kv_head)
        # m and l scratches share the same shape.
        scratch_bytes = (2 * lm_array_size + acc_array_size) * out_bytes
        total_bytes += scratch_bytes

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
        fixed_bq_sz: int | None = None,
        max_bkv_sz: int | None = None,
        max_bq_sz: int | None = None,
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. Therefore, we conservatively only use 80% of the VMEM budget.
        capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

        # To maximize MXU, we want bkv_sz to be at least MXU size.
        bkv_default = mxu_column_size
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # SEQ-ALONG-LANE does not support intra-page DMA offset.
            # TODO(kyuyeunk): Reduce granularity to lane size.
            bkv_default = utils.align_to(bkv_default, serve_cfgs.page_size)
        bkv_sz = bkv_stride = bkv_default
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size

        # Step 1: Lower batch_size if even the smallest bq and bkv size can trigger
        # OOM.

        # If current batch size triggers OOM, decrease batch size until the kernel
        # fits within VMEM limit.
        while (batch_size > 1 and calculate_vmem_usage(
                batch_size, bq_sz, bkv_sz) > capped_vmem_limit_bytes):
            batch_size -= 1

        # Indicates OOM was triggered even when batch_size=1.
        # NOTE: If the function does not exit at this point even when the value is
        # zero, it will trigger infinite loop at the next while loop.
        if batch_size == 0:
            raise ValueError(
                "Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        max_seq_len = serve_cfgs.pages_per_seq * serve_cfgs.page_size
        while (calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
               < capped_vmem_limit_bytes and bkv_sz <= max_seq_len
               and bkv_sz <= 4096  # Temporary cap on the KV block size.
               ):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        if bkv_sz > bkv_stride:
            bkv_sz -= bkv_stride
            if fixed_bq_sz is None:
                bq_sz -= bq_stride

        # Applied after the rollback, not inside the growth loop: the rollback
        # assumes the loop exited on the VMEM budget, so a cap in the loop
        # condition would roll back a step that was never taken. Caps must be
        # multiples of bkv_stride to keep KV tiles page-aligned. bq and bkv are
        # capped independently (see configs.SFS_PREFILL_MAX_*).
        if max_bkv_sz is not None and bkv_sz > max_bkv_sz:
            bkv_sz = max_bkv_sz
            if fixed_bq_sz is None:
                bq_sz = max_bkv_sz
        if fixed_bq_sz is None and max_bq_sz is not None and bq_sz > max_bq_sz:
            bq_sz = max_bq_sz

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError(
                "Cannot find block sizes that fit within VMEM limit.")

        # Step 2b: Step 2 grows bq and bkv together, so when bq hits the VMEM limit
        # bkv stops too. With many q heads per chip (e.g. Qwen3-32B at TP=2, 32 q
        # heads) that happens at 256, while a larger bkv still fits. Keep growing
        # bkv alone, up to a cap, while it fits. The cap is the measured GFC
        # optimum (512), further bounded by the chip's precision-aware cap when
        # one is set (see configs.SFS_PREFILL_MAX_*).
        if fixed_bq_sz is None:
            step2b_max_bkv_sz = 512
            if max_bkv_sz is not None:
                step2b_max_bkv_sz = min(step2b_max_bkv_sz, max_bkv_sz)
            while (bkv_sz + bkv_stride <= max_seq_len
                   and bkv_sz + bkv_stride <= step2b_max_bkv_sz and
                   calculate_vmem_usage(batch_size, bq_sz, bkv_sz + bkv_stride)
                   < capped_vmem_limit_bytes):
                bkv_sz += bkv_stride

        # Step 3: Given current tile size, calculate compute tile size.
        # For decode mode (fixed_bq_sz is not None), query chunking (bq_c_sz < bq_sz)
        # is unnecessary because the query tile size is already very small (<= 8 tokens).
        # Query chunking was designed for prefill workloads where query lengths are large;
        # applying it to decode causes redundant KV cache iterations and softmax carry
        # state save/restores. Therefore, decode mode directly keeps bq_c_sz = bq_sz.
        bq_c_sz = bq_sz

        if fixed_bq_sz is None:
            # Threshold value based on hardware generation and precision.
            tpu_info = pltpu.get_tpu_info()

            # The threshold budgets a compute block containing both matmuls, so it
            # is only raised when both of them run 8-bit.
            is_8bit = serve_cfgs.matmuls_are_8bit

            match tpu_info.generation:
                case 8 if configs.is_sfs(tpu_info):
                    threshold = 800
                case 7:
                    threshold = 1500
                case _:
                    threshold = 1500

            if is_8bit:
                flops_ratio = (tpu_info.fp8_ops_per_second //
                               tpu_info.bf16_ops_per_second)
                threshold *= flops_ratio
            num_bq_c = 1
            bq_c_rem = 0

            while (calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz)
                   > threshold or bq_c_rem != 0) and num_bq_c < bq_sz:
                num_bq_c += 1
                bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)
            # On exit bq_c_sz is the first candidate that divides bq_sz evenly and
            # fits the threshold.

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
        )

    # Fixed value based on experimental results.
    decode_batch_size = 8
    prefill_batch_size = 2

    decode_block_sizes = find_best_block_sizes(decode_batch_size,
                                               decode_query_size)
    prefill_max_bq_sz, prefill_max_bkv_sz = serve_cfgs.prefill_tile_caps(
        tpu_info)
    prefill_block_sizes = find_best_block_sizes(
        prefill_batch_size,
        max_bkv_sz=prefill_max_bkv_sz,
        max_bq_sz=prefill_max_bq_sz,
    )

    return decode_block_sizes, prefill_block_sizes


def get_tuned_params(model_config: configs.ModelConfigs,
                     serve_config: configs.ServingConfigs,
                     vmem_limit_bytes: int | None = None,
                     case: Literal['decode', 'prefill'] = 'decode',
                     decode_query_size: int = 1) -> configs.BlockSizes:
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    tuning_key = TuningKey.from_config(model_config, serve_config, case=case)
    if tuning_key not in tuned_params_mapping:
        decode_block_sizes, prefill_block_sizes = calculate_block_sizes(
            model_config,
            serve_config,
            vmem_limit_bytes,
            decode_query_size=decode_query_size)
        block_sizes = decode_block_sizes if case == 'decode' else prefill_block_sizes
    else:
        block_sizes = tuned_params_mapping[tuning_key].to_block_sizes()
    return block_sizes


# This is a placeholder as we haven't found better tuned block sizes for the cases
tuned_params_mapping: dict[TuningKey, TunableParams] = {}
