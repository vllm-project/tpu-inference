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
"""Wrapper for RPA kernel to match expected interface.

NOTE: all of the code in this directory is experimental and not fully tested!
To enable usage of this kernel in full run, you can pass the USE_BATCHED_RPA_KERNEL=1
environment variable.

Compared to the default RPA kernel, this kernel does the following:

1. Batches multiple sequences together to replace per-request flash_attention loops.

2. Enables triple-buffering via Pallas emit_pipeline

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via
scheduler.py kernel. Kernel is calculated once and ammortized across different layers in a model.

Note: batched_rpa is build on top / derived from RPA3.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import (configs, kernel,
                                                            schedule,
                                                            schedule_cp, utils)


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    k_scale: jax.Array | None = None,
    v_scale: jax.Array | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
) -> tuple[jax.Array, jax.Array]:

    total_q_tokens, actual_num_q_heads, actual_q_head_dim = q.shape
    total_kv_tokens, actual_num_kv_heads, actual_kv_head_dim = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_q_head_dim, num_lanes)

    # Compute aligned kv head dimension, accounting for per-token scale if
    # present.
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if k_scale is not None and v_scale is not None:
            scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
            aligned_kv_head_dim = utils.align_to(
                actual_kv_head_dim + num_scale_channels,
                num_sublanes * kv_packing)
        else:
            aligned_kv_head_dim = utils.align_to(actual_kv_head_dim,
                                                 num_sublanes * kv_packing)
    else:
        aligned_kv_head_dim = utils.align_to(actual_kv_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (jnp.pad(
        q.reshape(
            total_q_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head,
            actual_q_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_q_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        actual_num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    ).swapaxes(0, 1))

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2,
                                             kv_packing)

    if k_scale is not None and v_scale is not None:
        k_scale = k_scale.reshape(total_kv_tokens, actual_num_kv_heads)
        v_scale = v_scale.reshape(total_kv_tokens, actual_num_kv_heads)

        scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        num_scale_channels = max(1, scale_bits // kv_bits)

        # Bitcast the scale factors to the kv dtype and reshape scale factors to
        # (T, H_kv, num_scale_channels).
        k_scale_split = jax.lax.bitcast_convert_type(k_scale, k.dtype).reshape(
            total_kv_tokens, actual_num_kv_heads, num_scale_channels)
        v_scale_split = jax.lax.bitcast_convert_type(v_scale, v.dtype).reshape(
            total_kv_tokens, actual_num_kv_heads, num_scale_channels)

        # k/v have shape (T, H_kv, D) -> (T, H_kv, D + num_scale_channels)
        k = jnp.concatenate([k, k_scale_split], axis=-1)
        v = jnp.concatenate([v, v_scale_split], axis=-1)

        actual_kv_head_dim += num_scale_channels

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        align_tokens = max(num_lanes, page_size)
        padded_total_tokens = utils.align_to(total_kv_tokens, align_tokens)
        new_kv_hbm = (jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_kv_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_kv_head_dim),
            (
                (0, padded_total_tokens - total_kv_tokens),
                (0, 0),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        ).reshape(
            padded_total_tokens,
            actual_num_kv_heads_x2,
            aligned_kv_head_dim // kv_packing,
            kv_packing,
        ).transpose(1, 2, 3, 0))
    else:
        new_kv_hbm = jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_kv_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_kv_head_dim),
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_kv_tokens,
            num_kv_heads_x2_aligned // kv_packing,
            kv_packing,
            aligned_kv_head_dim,
        )
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    use_per_token_scale: bool = False,
    scale_dtype: jnp.dtype | None = None,
):
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_scale_channels = 0
        if use_per_token_scale:
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            scale_bits = jax.dtypes.itemsize_bits(scale_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
        base_dim = actual_head_dim + num_scale_channels
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            utils.align_to(base_dim, num_sublanes * kv_packing) // kv_packing,
            kv_packing,
            page_size,
        )
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


class BaseBlockSizeCalculator:
    """Chooses kernel block sizes that fit the VMEM budget.

    Subclass and override `prefill_block_sizes` to change how the budget is
    spent; the VMEM model and the compute-tile search are shared.
    """

    def __init__(
        self,
        model_cfgs: configs.ModelConfigs,
        serve_cfgs: configs.ServingConfigs,
        vmem_limit_bytes: int,
        decode_query_size: int = 1,
    ):
        self.model_cfgs = model_cfgs
        self.serve_cfgs = serve_cfgs
        self.vmem_limit_bytes = vmem_limit_bytes
        self.decode_query_size = decode_query_size

        tpu_info = pltpu.get_tpu_info()
        self.tpu_info = tpu_info
        num_lanes = tpu_info.num_lanes
        num_sublanes = tpu_info.num_sublanes
        self.mxu_column_size = tpu_info.mxu_column_size

        # Calculate aligned model dimensions.
        self.aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)

        aligned_num_q_heads_per_kv_head = utils.align_to(
            model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q)
        self.aligned_num_q_heads = (aligned_num_q_heads_per_kv_head *
                                    model_cfgs.num_kv_heads)

        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            self.aligned_num_kv_heads_x2 = model_cfgs.num_kv_heads * 2
        else:
            bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2,
                                 serve_cfgs.packing_kv)
            if utils.has_bank_conflicts(bkv_stride):
                bkv_stride += 1
            self.aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

        self.q_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_q) / 8
        self.kv_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_kv) / 8
        self.out_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_out) / 8

        is_packed_fp4 = serve_cfgs.dtype_kv == jnp.uint8
        physical_kv_head_dim = model_cfgs.head_dim
        if is_packed_fp4:
            physical_kv_head_dim = physical_kv_head_dim // 2

        packing_kv = serve_cfgs.packing_kv
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            base_dim = physical_kv_head_dim + serve_cfgs.scale_channels
            self.aligned_kv_head_dim_val = utils.align_to(
                base_dim, num_sublanes * packing_kv)
        else:
            self.aligned_kv_head_dim_val = utils.align_to(
                model_cfgs.head_dim, num_lanes)

    @property
    def capped_vmem_limit_bytes(self) -> float:
        # Even if we loose some potential performance, we want to avoid OOM at
        # all costs. Therefore, we conservatively only use 80% of the VMEM
        # budget.
        return self.vmem_limit_bytes * 0.8

    @property
    def max_bkv_sz(self) -> int:
        max_seq_len = self.serve_cfgs.pages_per_seq * self.serve_cfgs.page_size
        # TEMP PATCH, PLEASE SEE go/bkv_sz FOR MORE INFO.
        return min(max_seq_len, 4096)

    def calculate_vmem_usage(self, batch_size: int, n_buffer: int, bq_sz: int,
                             bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""
        model_cfgs = self.model_cfgs
        serve_cfgs = self.serve_cfgs

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * self.aligned_num_q_heads * self.aligned_head_dim
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            bkv_array_size = ((bkv_sz + 2 * serve_cfgs.page_size) *
                              self.aligned_num_kv_heads_x2 *
                              self.aligned_kv_head_dim_val)
        else:
            bkv_array_size = (bkv_sz * self.aligned_num_kv_heads_x2 *
                              self.aligned_kv_head_dim_val)

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * self.q_bytes
        bkv_bytes = bkv_array_size * self.kv_bytes
        bo_bytes = bo_array_size * self.out_bytes

        # Account for multiple buffers. For output, we always use double buffer.
        bq_bytes *= n_buffer
        bkv_bytes *= n_buffer
        bo_bytes *= 2

        # Sum up all buffer memory usage (scales are co-located inside bkv_bytes).
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

        # Step 2: Calculate worst case memory usage during computation.

        # Calculate the size of loaded bq and bkv size.
        loaded_bq_size = bq_sz * model_cfgs.num_q_heads * self.aligned_head_dim
        loaded_bkv_size = (bkv_sz * model_cfgs.num_kv_heads *
                           self.aligned_head_dim)

        # Calculate peak memory requirement of otuput - which is attention weight.
        qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

        # Convert to bytes.
        loaded_bq_bytes = loaded_bq_size * self.q_bytes
        loaded_bkv_bytes = loaded_bkv_size * self.kv_bytes
        qk_bytes = qk_size * self.out_bytes

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
            loaded_scale_bytes = (2 * bkv_sz * model_cfgs.num_kv_heads *
                                  scale_bytes)

        # Sum up all compute memory usage.
        compute_bytes = (loaded_bq_bytes + loaded_bkv_bytes + qk_bytes +
                         loaded_scale_bytes)

        # Step 3: Sum up all memory usage.
        total_bytes = buffer_bytes + compute_bytes

        # Account for batch size.
        total_bytes *= batch_size

        return total_bytes

    def scratch_vmem_bytes(self, batch_size: int, bq_sz: int) -> float:
        """VMEM the kernel holds outside the pipelined Q/KV/O buffers.

    The online-softmax scratch (m, l and the accumulator) and the LSE output
    are all sized by bq_sz, and none of them appear in
    `calculate_vmem_usage`. That under-count is harmless for a search that
    grows bq_sz and bkv_sz together from the MXU tile, but not for one that
    maximises bq_sz first -- at a full local query block these terms are the
    larger half of the budget.
    """
        num_lanes = self.tpu_info.num_lanes
        num_sublanes = self.tpu_info.num_sublanes
        num_kv_heads = self.model_cfgs.num_kv_heads
        aligned_q_per_kv = self.aligned_num_q_heads // num_kv_heads

        # m and l are [kv_heads, bq_sz * q_per_kv, num_lanes]; the accumulator
        # is [kv_heads, bq_sz * q_per_kv, kv_head_dim]. None are per-lane.
        rows = bq_sz * aligned_q_per_kv
        total = (2 * num_kv_heads * rows * num_lanes +
                 num_kv_heads * rows * self.aligned_kv_head_dim_val)
        total *= self.out_bytes

        if self.serve_cfgs.return_lse:
            # [batch, kv_heads, bq_sz * lse_rows, num_lanes], double buffered.
            lse_rows = bq_sz * utils.align_to(aligned_q_per_kv, num_sublanes)
            total += (2 * batch_size * num_kv_heads * lse_rows * num_lanes *
                      self.out_bytes)
        return total

    def fits(self, batch_size: int, n_buffer: int, bq_sz: int,
             bkv_sz: int) -> bool:
        used = (self.calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
                + self.scratch_vmem_bytes(batch_size, bq_sz))
        return used <= self.capped_vmem_limit_bytes

    def calculate_compute_buffer_time(self, batch_size: int, bq_c_sz: int,
                                      bkv_sz: int) -> int:
        """Calculate computational complexity of a single compute block."""
        model_cfgs = self.model_cfgs

        num_k_rows = pl.cdiv(bkv_sz, self.mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, self.mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    def compute_tile_size(self, batch_size: int, bq_sz: int,
                          bkv_sz: int) -> int:
        """Largest divisor of bq_sz whose compute block stays under threshold."""
        tpu_info = self.tpu_info
        is_8bit = jnp.dtype(self.serve_cfgs.dtype_q).itemsize == 1

        match tpu_info.generation:
            case 8 if "8i" in tpu_info.chip_version or "v8i" in tpu_info.chip_version:
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
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (self.calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz)
               > threshold or bq_c_rem != 0) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

        return last_valid_bq_c_sz

    def find_best_block_sizes(
            self,
            max_batch_size: int,
            max_n_buffer: int,
            fixed_bq_sz: int | None = None) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""
        capped_vmem_limit_bytes = self.capped_vmem_limit_bytes

        bkv_sz = bkv_stride = self.mxu_column_size
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
        while (batch_size > 1 and self.calculate_vmem_usage(
                batch_size, n_buffer, bq_sz, bkv_sz) > capped_vmem_limit_bytes):
            batch_size -= 1

        # As a last resort, attempt to decrease number of buffers to avoid OOM.
        while (self.calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               > capped_vmem_limit_bytes):
            n_buffer -= 1

        # Indicates OOM was triggered even when batch_size=1 or n_buffer=1.
        # NOTE: If the function does not exit at this point even when either values
        # are zero, it will trigger infinite loop at the next while loop.
        if batch_size == 0 or n_buffer == 0:
            raise ValueError(
                "Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        while (self.calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               < capped_vmem_limit_bytes and bkv_sz <= self.max_bkv_sz):
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

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError(
                "Cannot find block sizes that fit within VMEM limit.")

        # Step 3: Given current tile size, calculate compute tile size.
        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=self.compute_tile_size(batch_size, bq_sz, bkv_sz),
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    # Default to triple buffer as its almost always beneficial.
    n_buffer = 3
    # Fixed values based on experimental results.
    decode_batch_size = 8
    prefill_batch_size = 2

    def decode_block_sizes(self) -> configs.BlockSizes:
        return self.find_best_block_sizes(self.decode_batch_size,
                                          self.n_buffer,
                                          self.decode_query_size)

    def prefill_block_sizes(self) -> configs.BlockSizes:
        return self.find_best_block_sizes(self.prefill_batch_size,
                                          self.n_buffer)

    def calculate(self) -> tuple[configs.BlockSizes, configs.BlockSizes]:
        return self.decode_block_sizes(), self.prefill_block_sizes()


class CPBlockSizeCalculator(BaseBlockSizeCalculator):
    """Block sizes for context parallelism.

    The ring replays the whole KV walk once per Q block, so the default search
    -- which grows bq and bkv together from the MXU tile -- spends the budget
    exactly backwards: it leaves several Q blocks, each dragging every KV block
    around the ring again. This spends it in priority order instead: bq_sz
    comes first and is the whole local query block when VMEM allows, then the
    largest bkv_sz, for fewer and larger ring steps.

    It stays at one lane per step. A step's lanes hold different KV blocks of
    the *same* Q block, so a second lane duplicates the Q and output buffers --
    tens of MiB -- to add only its own KV block, which is a few MiB. Spending
    that budget on bkv_sz instead buys the same KV per step for less VMEM, and
    it keeps the ring schedule to a plain (block, hop) walk. This would be
    worth revisiting if Q/O were ever shared across lanes, or if bkv_sz became
    limited by `max_bkv_sz` rather than by VMEM.

    Only the ring's cache phase is reordered; everything else defers to the
    base search.
    """

    # The ring only ever needs the hop it is on and the one it is filling.
    ring_n_buffer = 2
    # One lane per step: the rotation moves the whole buffer in a single copy,
    # so every lane of a step would otherwise have to be on the same hop.
    ring_batch_size = 1

    def prefill_block_sizes(self) -> configs.BlockSizes:
        serve_cfgs = self.serve_cfgs
        if serve_cfgs.pcp_ring_axis_name is None:
            return super().prefill_block_sizes()

        n_buffer = self.ring_n_buffer
        batch_size = self.ring_batch_size
        min_bkv_sz = self.mxu_column_size

        # 1. The whole local Q in a single block, so the ring walks the cache
        #    once. Halve it until it fits next to the smallest KV block.
        bq_sz = utils.align_to(max(serve_cfgs.total_q_tokens, min_bkv_sz),
                               min_bkv_sz)
        while bq_sz > min_bkv_sz and not self.fits(batch_size, n_buffer, bq_sz,
                                                   min_bkv_sz):
            bq_sz //= 2
        if not self.fits(batch_size, n_buffer, bq_sz, min_bkv_sz):
            raise ValueError(
                "Cannot find block sizes that fit within VMEM limit.")

        # 2. Spend what is left on the KV block: fewer, larger ring steps.
        bkv_sz = min_bkv_sz
        while (bkv_sz * 2 <= self.max_bkv_sz
               and self.fits(batch_size, n_buffer, bq_sz, bkv_sz * 2)):
            bkv_sz *= 2

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=self.compute_tile_size(batch_size, bq_sz, bkv_sz),
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""
    calculator_cls = (CPBlockSizeCalculator
                      if serve_cfgs.cp_group_size is not None else
                      BaseBlockSizeCalculator)
    return calculator_cls(model_cfgs, serve_cfgs, vmem_limit_bytes,
                          decode_query_size).calculate()


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "use_per_token_scale",
        "per_token_scale_dtype",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "kv_layout",
        "decode_query_size",
        "cp_group_size",
        "attention_scope",
        "return_lse",
        "pcp_ring_axis_name",
        "pcp_ring_mesh_axis_names",
    ),
    # Donation of transient inputs can fail for some runtime buffer layouts in
    # the experimental tuning path. Keep donation only for kv_cache, which is
    # the intended long-lived mutable state.
    donate_argnames=("kv_cache", ),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array | None,
    values: jax.Array | None,
    kv_cache: jax.Array | None,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    use_per_token_scale: bool = False,
    per_token_scale_dtype: jnp.dtype | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: jax.Array | bool | None = None,
    q_positions: jax.Array | None = None,
    kv_new_lens: jax.Array | None = None,
    new_kv_page_indices: jax.Array | None = None,
    pcp_ring_axis_name: str | None = None,
    pcp_ring_mesh_axis_names: tuple[str, ...] | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

  Args:
    queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
    keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
    values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
    kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Stores existing kv cache data where k & vs are
      concatenated along num kv heads dim.
    kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
    page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
      sequence.
    cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
      sequences. distribution[2] represents total number of sequences.
    sm_scale: Softmax scale value.
    sliding_window: Size of sliding window (also known as local attention). kvs
      outside of the window is not fetched from hbm and masked out during
      computation.
    soft_cap: Cap values of softmax inputs.
    mask_value: Value to use for causal masking. Defaults to smallest
      representable value of the activation dtype.
    use_per_token_scale: Whether to use per-token quantization.
    per_token_scale_dtype: Dtype of per-token scale. Defaults to None.
    q_scale: Quantization scale value of queries.
    k_scale: Per-tensor quantization scale value of keys.
    v_scale: Per-tensor quantization scale value of values.
    dynamic_k_scale: Per-token quantization scale value of keys.
    dynamic_v_scale: Per-token quantization scale value of values.
    chunk_prefill_size: Not used.
    decode_block_sizes: Kernel block size to use during decode.
    prefill_block_sizes: Kernel block size to use during prefill.
    vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
      size of the hardware.
    debug_mode: Not used.
    out_dtype: Dtype of output. Defaults to dtype of queries.
    use_causal_mask: Not used.
    decode_query_size: Number of query tokens in decode (1 by default, can be
      higher in case of speculative decoding).
    cp_group_size: Size of the context parallelism (CP) group. KV cache is
      sharded across devices in this group. Defaults to None.
    cp_rank: Rank of the current device within the CP group, which determine
      the token ownership. Defaults to None.
    attention_scope: Which KV positions to attend to. FULL attends all
      positions, CACHE_ONLY skips new tokens, NEW_TOKENS_ONLY skips cached
      tokens. Defaults to FULL.
    update_kv_cache: [max_num_seqs] bool, or a scalar bool applied to every
      sequence. Whether a sequence writes its new KV back to the cache.
      Defaults to False under CACHE_ONLY and True otherwise.
    q_positions: [max_num_seqs]. Position of each sequence's first query token
      in the sequence's KV index space (cache tokens then new tokens); the
      causal mask is relative to it. Defaults to kv_cache_lens, i.e. the query
      block sits immediately after the cache. PCP passes
      `kv_cache_len + chunk_offset`, because a rank's head/tail chunks sit at
      two different places inside the all-gathered current KV.
    kv_new_lens: [max_num_seqs]. New KV tokens each sequence contributes.
      Defaults to the sequence's own Q length.  
    new_kv_page_indices: [max_num_seqs * new_pages_per_seq]. Page table for the
      new KV, mapping page p of a sequence's new KV in global token order to
      the page that actually holds it in keys/values. Defaults to None, i.e. a
      sequence's new KV is exactly its own Q, laid out contiguously. PCP passes
      a table so the all-gathered current KV can stay in its natural rank order
      instead of being reordered into token order first.
    pcp_ring_axis_name: not implemented yet; see `kernel.RingAttentionHooks`.
    pcp_ring_mesh_axis_names: not implemented yet.
    return_lse: If True, return log-sum-exp (lse) values along with the
      output. Defaults to False.

  Returns:
    out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
    new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Result of new kv cache where k & vs are
      concatenated along num kv heads dim.
    lse (only when return_lse=True): [max_num_tokens, num_q_heads].
      Log-sum-exp values (m + log(l)) for each query token and head,
      needed for merging partial attention results in CP.
  """

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if pcp_ring_axis_name is not None:
        if attention_scope != configs.AttentionScope.CACHE_ONLY:
            raise ValueError(
                "The PCP ring only applies to the cache phase; got "
                f"{attention_scope=}.")
        if cp_group_size is None:
            raise ValueError("pcp_ring_axis_name requires cp_group_size.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    paged_new_kv = new_kv_page_indices is not None
    if paged_new_kv:
        if new_kv_page_indices.shape[0] % max_num_seqs != 0:
            raise ValueError(
                f"Expected {new_kv_page_indices.shape[0]=} to be divisible by "
                f"{max_num_seqs=}.")
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            raise NotImplementedError(
                "A new-KV page table is only wired up for "
                "KVLayout.HEAD_ALONG_SUBLANE.")


    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )

    if k_scale is not None and dynamic_k_scale is not None:
        raise ValueError(
            "Only one of k_scale or dynamic_k_scale can be set. Got"
            f" {k_scale=} and {dynamic_k_scale=}")
    if v_scale is not None and dynamic_v_scale is not None:
        raise ValueError(
            "Only one of v_scale or dynamic_v_scale can be set. Got"
            f" {v_scale=} and {dynamic_v_scale=}")

    k_scale_config = k_scale
    v_scale_config = v_scale
    k_scale_tensor = dynamic_k_scale
    v_scale_tensor = dynamic_v_scale

    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        scale_q=q_scale,
        scale_k=k_scale_config,
        scale_v=v_scale_config,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        pcp_ring_axis_name=pcp_ring_axis_name,
        pcp_ring_mesh_axis_names=pcp_ring_mesh_axis_names,
        paged_new_kv=paged_new_kv,
        attention_scope=attention_scope,
        return_lse=return_lse,
    )

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        k_scale_tensor,
        v_scale_tensor,
        kv_layout=kv_layout,
        page_size=page_size,
    )

    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs,
        serve_cfgs,
        vmem_limit_bytes,
        decode_query_size=decode_query_size,
    )
    # Pre-allocate LSE buffer.
    lse_hbm_init: jax.Array | None = None
    if return_lse:
        num_lanes = pltpu.get_tpu_info().num_lanes
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        q_packing = utils.get_dtype_packing(queries.dtype)
        num_q_heads_per_kv_head = num_q_heads // num_kv_heads
        aligned_num_q_heads_per_kv_head = utils.align_to(
            num_q_heads_per_kv_head, q_packing)
        # LSE rows per token are padded to a whole sublane tile so the
        # copy-out offset `q_src * lse_rows_per_token` stays aligned; see
        # RpaConfigs.lse_rows_per_token.
        lse_rows_per_token = utils.align_to(aligned_num_q_heads_per_kv_head,
                                            num_sublanes)
        max_tokens = queries.shape[0]
        lse_hbm_init = jnp.full(
            [
                num_kv_heads,
                max_tokens * lse_rows_per_token,
                num_lanes,
            ],
            -jnp.inf,
            dtype=out_dtype,
        )

    # Compute per-sequence length parameters for the kernel.
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]

    # A sequence's new KV is its own Q unless the caller says otherwise. PCP
    # does: head and tail chunks of a request each carry the whole all-gathered
    # current KV.
    if kv_new_lens is None:
        kv_new_lens = q_lens
    global_kv_cache_lens = kv_lens - kv_new_lens

    # The cache phase attends no new tokens, whoever contributed them.
    if attention_scope == configs.AttentionScope.CACHE_ONLY:
        kv_new_lens = jnp.zeros_like(q_lens)

    # The kernel narrows the global length to this call's share of the cache
    # itself (`RpaConfigs.local_kv_cache_len`), so it only needs the rank.
    cp_rank_arg = (cp_rank if cp_rank is not None else jnp.zeros(
        (1, ), jnp.int32))

    if q_positions is None:
        # Queries sit right after the cache, so nothing of it is masked. The
        # global length works for the sharded cache phase too: it is at least
        # the local one, which is all that is indexed there.
        q_positions = global_kv_cache_lens

    if update_kv_cache is None:
        update_kv_cache = (attention_scope !=
                           configs.AttentionScope.CACHE_ONLY)
    # Accepts a scalar (applies to every sequence) or one flag per sequence.
    update_kv_cache = jnp.broadcast_to(
        jnp.asarray(update_kv_cache) != 0, q_lens.shape).astype(jnp.int32)

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
        lse_hbm_in: jax.Array | None,
    ):
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or default_decode
        else:
            effective_blocks = prefill_block_sizes or default_prefill


        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
        )
        # Select metadata computer class.
        if cp_group_size is not None:
            computer_cls = schedule_cp.CPMetadataComputer
            extra_scalars = (cp_rank, ) if cp_rank is not None else ()
        else:
            computer_cls = schedule.BaseMetadataComputer
            extra_scalars = ()

        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            q_positions,
            global_kv_cache_lens,
            kv_new_lens,
            update_kv_cache,
            distribution,
            cfgs=cfgs,
            computer_cls=computer_cls,
            extra_scalars=extra_scalars,
        )
        result = kernel.rpa_kernel(
            cu_q_lens,
            q_positions,
            global_kv_cache_lens,
            kv_new_lens,
            cp_rank_arg,
            page_indices,
            () if new_kv_page_indices is None else (new_kv_page_indices, ),
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            lse_hbm_in,
            cfgs=cfgs,
            computer_cls=computer_cls,
        )
        if return_lse:
            o_out, kv_out, lse_out = result
        else:
            o_out, kv_out, _ = result
            lse_out = None
        return o_out, kv_out, lse_out

    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init)
    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm)

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    if not return_lse:
        return o_hbm, kv_cache

    # Reshape LSE from [num_kv_heads, max_tokens *
    # aligned_num_q_heads_per_kv_head, num_lanes] to [max_tokens, num_q_heads].
    max_tokens = queries.shape[0]
    # Extract first lane (scalar LSE value per token-head pair).
    lse = lse_hbm.reshape(num_kv_heads, max_tokens, lse_rows_per_token,
                          num_lanes)[:, :max_tokens, :num_q_heads_per_kv_head,
                                     0]
    lse = lse.swapaxes(0, 1).reshape(max_tokens, num_q_heads)

    return o_hbm, kv_cache, lse
