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
import functools
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class RpaCase(Enum):
    """Represents the different cases for Ragged Paged Attention.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution):
        assert distribution.shape == (3, )
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        elif self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        elif self == RpaCase.MIXED:
            return distribution[1], distribution[2]
        else:
            raise ValueError(f"Unsupported RPA case: {self}")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=True)
class RPAConfig:
    num_seq: int
    bq_sz: int
    bkv_sz: int
    batch_size: int
    page_size: int
    bkv_p: int
    pages_per_seq: int
    max_steps_ub: int
    total_q_tokens: int
    total_num_pages: int | None = None
    head_dim: int = 128
    num_kv_heads: int = 8
    sm_scale: float = 1.0
    soft_cap: Any = None
    num_q_heads_per_kv_head: int = 1
    sliding_window: Any = None
    mask_value: float = -1e30
    q_dtype: Any = jnp.bfloat16
    kv_dtype: Any = jnp.bfloat16
    q_scale: Any = None
    k_scale: Any = None
    v_scale: Any = None
    vmem_limit_bytes: int | None = 60 * 1024 * 1024
    case: RpaCase = RpaCase.MIXED


def get_dtype_bitwidth(dtype):
    return dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def has_bank_conflicts(stride: int, distance=24, num_banks=32) -> bool:
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class RPASchedule:
    """Container for metadata arrays with integrated shape/spec logic."""

    s_idx: Any
    q_idx: Any
    k_idx: Any
    is_last_k: Any  # [steps * batch]
    do_writeback: Any  # [steps * batch]
    dma_q: Any  # [steps * batch * 2]
    dma_kv_cache: Any  # [steps * batch * bkv_p * 3]
    dma_kv_new: Any  # [steps * batch * bkv_p * 4]
    actual_steps: Any
    batch_size: int = dataclasses.field(default=0, metadata={"static": True})
    bkv_p: int = dataclasses.field(default=0, metadata={"static": True})

    def tree_flatten(self):
        return (
            self.s_idx,
            self.q_idx,
            self.k_idx,
            self.is_last_k,
            self.do_writeback,
            self.dma_q,
            self.dma_kv_cache,
            self.dma_kv_new,
            self.actual_steps,
        ), (self.batch_size, self.bkv_p)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, batch_size=aux[0], bkv_p=aux[1])

    def get_dma_kv_cache(self, step: int, batch_idx: int,
                         page_idx: int) -> tuple[int, int, int]:
        # Stride is 3: src_hbm, dst_vmem, size
        cache_step_stride = self.batch_size * self.bkv_p * 3
        cache_batch_stride = self.bkv_p * 3
        idx = (step * cache_step_stride + batch_idx * cache_batch_stride +
               page_idx * 3)
        # 0: src_hbm, 1: dst_vmem, 2: size
        src_off = self.dma_kv_cache[idx + 0]
        dst_off = self.dma_kv_cache[idx + 1]
        sz = self.dma_kv_cache[idx + 2]
        return src_off, dst_off, sz

    def get_dma_kv_new(self, step: int, batch_idx: int,
                       page_idx: int) -> tuple[int, int, int, int]:
        # Stride is 4: dst_hbm, src_hbm, dst_vmem, size
        new_step_stride = self.batch_size * self.bkv_p * 4
        base_idx = step * new_step_stride + batch_idx * self.bkv_p * 4 + page_idx * 4
        # 0: dst_hbm, 1: src_hbm, 2: dst_vmem, 3: size
        dst_hbm = self.dma_kv_new[base_idx + 0]
        src_hbm = self.dma_kv_new[base_idx + 1]
        dst_vmem = self.dma_kv_new[base_idx + 2]
        sz = self.dma_kv_new[base_idx + 3]
        return dst_hbm, src_hbm, dst_vmem, sz

    def get_dma_q(self, step: int, batch_idx: int) -> tuple[int, int]:
        # Stride is 2: src_hbm, size
        q_step_stride = self.batch_size * 2
        base_idx = step * q_step_stride + batch_idx * 2
        # 0: src_hbm, 1: size
        src_hbm = self.dma_q[base_idx + 0]
        sz = self.dma_q[base_idx + 1]
        return src_hbm, sz

    @classmethod
    def out_shape(cls, config: RPAConfig):
        """Returns a Pytree of ShapeDtypeStruct for pallas_call."""
        bs = config.batch_size
        steps = config.max_steps_ub
        return cls(
            s_idx=jax.ShapeDtypeStruct((steps * bs, ), jnp.int32),
            q_idx=jax.ShapeDtypeStruct((steps * bs, ), jnp.int32),
            k_idx=jax.ShapeDtypeStruct((steps * bs, ), jnp.int32),
            is_last_k=jax.ShapeDtypeStruct((steps * bs, ), jnp.int32),
            do_writeback=jax.ShapeDtypeStruct((steps * bs, ), jnp.int32),
            dma_q=jax.ShapeDtypeStruct((steps * bs * 2, ), jnp.int32),
            dma_kv_cache=jax.ShapeDtypeStruct(
                (steps * bs * config.bkv_p * 3, ), jnp.int32),
            dma_kv_new=jax.ShapeDtypeStruct((steps * bs * config.bkv_p * 4, ),
                                            jnp.int32),
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
            batch_size=bs,
            bkv_p=config.bkv_p,
        )

    @classmethod
    def out_specs(cls, config: RPAConfig):
        """Returns a Pytree of BlockSpecs matching the output structure."""

        def bs_spec(shape):
            return pl.BlockSpec(memory_space=pltpu.SMEM, block_shape=shape)

        bs = config.batch_size
        steps = config.max_steps_ub
        return cls(
            s_idx=bs_spec((steps * bs, )),
            q_idx=bs_spec((steps * bs, )),
            k_idx=bs_spec((steps * bs, )),
            is_last_k=bs_spec((steps * bs, )),
            do_writeback=bs_spec((steps * bs, )),
            dma_q=bs_spec((steps * bs * 2, )),
            dma_kv_cache=bs_spec((steps * bs * config.bkv_p * 3, )),
            dma_kv_new=bs_spec((steps * bs * config.bkv_p * 4, )),
            actual_steps=bs_spec((1, )),
            batch_size=bs,
            bkv_p=config.bkv_p,
        )

    @classmethod
    def smem_specs(cls, config: RPAConfig):
        """Returns a Pytree of pltpu.SMEM matching the output structure."""
        bs = config.batch_size
        steps = config.max_steps_ub
        return cls(
            s_idx=pltpu.SMEM((steps * bs, ), jnp.int32),
            q_idx=pltpu.SMEM((steps * bs, ), jnp.int32),
            k_idx=pltpu.SMEM((steps * bs, ), jnp.int32),
            is_last_k=pltpu.SMEM((steps * bs, ), jnp.int32),
            do_writeback=pltpu.SMEM((steps * bs, ), jnp.int32),
            dma_q=pltpu.SMEM((steps * bs * 2, ), jnp.int32),
            dma_kv_cache=pltpu.SMEM((steps * bs * config.bkv_p * 3, ),
                                    jnp.int32),
            dma_kv_new=pltpu.SMEM((steps * bs * config.bkv_p * 4, ),
                                  jnp.int32),
            actual_steps=pltpu.SMEM((1, ), jnp.int32),
            batch_size=bs,
            bkv_p=config.bkv_p,
        )


def rpa_metadata_schedule_kernel(
    cu_q_lens_ref,
    kv_lens_ref,
    page_indices_ref,
    distribution_ref,
    schedule: RPASchedule,
    lane_lengths_ref,
    *,
    config: RPAConfig,
):
    """Generates metadata for RPA scheduling."""
    for b in range(config.batch_size):
        lane_lengths_ref[b] = 0

    def seq_loop(s_idx, _):
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        k_len = kv_lens_ref[s_idx]
        q_len = q_end - q_start

        n_q = (q_len + config.bq_sz - 1) // config.bq_sz
        n_k = (k_len + config.bkv_sz - 1) // config.bkv_sz

        def q_loop(q_idx, _):
            target_lane = 0
            min_len = lane_lengths_ref[0]
            for b in range(1, config.batch_size):
                is_better = lane_lengths_ref[b] < min_len
                target_lane = jnp.where(is_better, b, target_lane)
                min_len = jnp.where(is_better, lane_lengths_ref[b], min_len)

            curr_ptr = lane_lengths_ref[target_lane]
            q_src = q_start + q_idx * config.bq_sz
            q_sz_task = jnp.clip(q_end - q_src, 0, config.bq_sz)

            start_k_idx = 0
            if config.sliding_window is not None:
                sw_start_idx = (k_len - q_len + q_idx * config.bq_sz -
                                config.sliding_window + 1)
                start_k_idx = jnp.maximum(0, sw_start_idx) // config.bkv_sz

            end_k_idx_causal = (k_len - q_len + q_idx * config.bq_sz +
                                q_sz_task - 1) // config.bkv_sz + 1
            end_k_idx = jnp.minimum(n_k, end_k_idx_causal)

            def k_loop(k_idx, curr_ptr):
                idx = curr_ptr * config.batch_size + target_lane
                schedule.s_idx[idx] = s_idx
                schedule.q_idx[idx] = q_idx
                schedule.k_idx[idx] = k_idx
                schedule.is_last_k[idx] = jnp.asarray(k_idx == end_k_idx - 1,
                                                      dtype=jnp.int32)

                q_idx_base = curr_ptr * (config.batch_size *
                                         2) + target_lane * 2
                schedule.dma_q[q_idx_base + 0] = q_src
                schedule.dma_q[q_idx_base + 1] = q_sz_task

                kv_len_start = k_idx * config.bkv_sz
                kv_p_start = k_idx * config.bkv_p
                kv_left = k_len - kv_len_start
                kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
                p_offset = s_idx * config.pages_per_seq + kv_p_start

                kv_cache_base = curr_ptr * (config.batch_size * config.bkv_p *
                                            3) + target_lane * (config.bkv_p *
                                                                3)
                for i in range(config.bkv_p):
                    sz = jnp.clip(kv_left_frm_cache - i * config.page_size, 0,
                                  config.page_size)
                    p_idx = jnp.minimum(p_offset + i,
                                        page_indices_ref.shape[0] - 1)

                    base_i = kv_cache_base + i * 3
                    schedule.dma_kv_cache[base_i +
                                          0] = (page_indices_ref[p_idx] *
                                                config.page_size)
                    schedule.dma_kv_cache[base_i + 1] = i * config.page_size
                    schedule.dma_kv_cache[base_i + 2] = sz

                kv_left_frm_new = kv_left - kv_left_frm_cache
                bkv_sz_cache = jnp.minimum(kv_left_frm_cache, config.bkv_sz)
                new_sz = jnp.minimum(config.bkv_sz - bkv_sz_cache,
                                     kv_left_frm_new)

                # Writeback logic: each new k block is written back by the first q block that attends to it.
                q_wb = jnp.maximum(0, (k_idx * config.bkv_sz -
                                       (k_len - q_len)) // config.bq_sz)
                schedule.do_writeback[idx] = jnp.asarray(
                    (new_sz > 0) & (q_idx == q_wb), dtype=jnp.int32)

                kv_new_base = curr_ptr * (config.batch_size * config.bkv_p *
                                          4) + target_lane * (config.bkv_p * 4)
                for i in range(config.bkv_p):
                    slot_start = i * config.page_size
                    slot_end = (i + 1) * config.page_size

                    start_in_slot = jnp.maximum(slot_start, bkv_sz_cache)
                    end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                    sz = jnp.maximum(0, end_in_slot - start_in_slot)

                    p_idx = (kv_len_start + start_in_slot) // config.page_size
                    p_idx = jnp.minimum(p_idx, config.pages_per_seq - 1)
                    p_off = (kv_len_start + start_in_slot) % config.page_size
                    hbm_page = page_indices_ref[s_idx * config.pages_per_seq +
                                                p_idx]

                    base_i = kv_new_base + i * 4
                    schedule.dma_kv_new[
                        base_i + 0] = hbm_page * config.page_size + p_off
                    schedule.dma_kv_new[base_i + 1] = q_end - kv_left_frm_new
                    schedule.dma_kv_new[base_i + 2] = start_in_slot
                    schedule.dma_kv_new[base_i + 3] = sz
                return curr_ptr + 1

            lane_lengths_ref[target_lane] = jax.lax.fori_loop(
                start_k_idx, end_k_idx, k_loop, curr_ptr)
            return None

        jax.lax.fori_loop(0, n_q, q_loop, None)

    start_seq_idx, end_seq_idx = config.case.get_range(distribution_ref)
    jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, None)

    max_steps = lane_lengths_ref[0]
    for b in range(1, config.batch_size):
        max_steps = jnp.where(lane_lengths_ref[b] > max_steps,
                              lane_lengths_ref[b], max_steps)
    schedule.actual_steps[0] = max_steps

    def mask_lane(b, _):
        start_step = lane_lengths_ref[b]

        def mask_step(step, _):
            idx = step * config.batch_size + b

            # Mark as invalid and zero out control flags
            schedule.s_idx[idx] = -1
            schedule.q_idx[idx] = 0
            schedule.k_idx[idx] = 0
            schedule.is_last_k[idx] = 0
            schedule.do_writeback[idx] = 0

            # TODO: theoretically we just need to zero out the size, check later
            q_base = step * config.batch_size * 2 + b * 2
            schedule.dma_q[q_base + 0] = 0
            schedule.dma_q[q_base + 1] = 0

            cache_base = (step * config.batch_size * config.bkv_p * 3 +
                          b * config.bkv_p * 3)
            for i in range(config.bkv_p):
                base_i = cache_base + i * 3
                schedule.dma_kv_cache[base_i + 0] = 0
                schedule.dma_kv_cache[base_i + 1] = 0
                schedule.dma_kv_cache[base_i + 2] = 0

            new_base = (step * config.batch_size * config.bkv_p * 4 +
                        b * config.bkv_p * 4)
            for i in range(config.bkv_p):
                base_i = new_base + i * 4
                schedule.dma_kv_new[base_i + 0] = 0
                schedule.dma_kv_new[base_i + 1] = 0
                schedule.dma_kv_new[base_i + 2] = 0
                schedule.dma_kv_new[base_i + 3] = 0

        jax.lax.fori_loop(start_step, max_steps, mask_step, None)

    jax.lax.fori_loop(0, config.batch_size, mask_lane, None)


def generate_rpa_metadata(
    cu_q_lens,
    kv_lens,
    page_indices,
    bq_sz,
    bkv_sz,
    page_size,
    batch_size,
    interpret=False,
    max_steps_ub=-1,
):
    # this method is just for testing, it's not used in the final kernel.
    num_seq = len(kv_lens)
    bkv_p = bkv_sz // page_size
    pages_per_seq = len(page_indices) // num_seq

    if max_steps_ub == -1:
        # Static upper bound calculation
        n_q = (cu_q_lens[1:] - cu_q_lens[:-1] + bq_sz - 1) // bq_sz
        n_k = (kv_lens + bkv_sz - 1) // bkv_sz
        if batch_size == 1:
            max_steps_ub = int(jnp.sum(n_q * n_k))
        else:
            max_steps_ub = int(
                jnp.sum(n_q * n_k) // batch_size + jnp.max(n_q * n_k))

        # Round up to nearest 128
        max_steps_ub = (max_steps_ub + 127) // 128 * 128

    config = RPAConfig(
        num_seq=num_seq,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        page_size=page_size,
        bkv_p=bkv_p,
        pages_per_seq=pages_per_seq,
        max_steps_ub=max_steps_ub,
        total_q_tokens=cu_q_lens[-1],
        case=RpaCase.MIXED,
    )

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, config=config),
        out_shape=RPASchedule.out_shape(config),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[],
            out_specs=RPASchedule.out_specs(config),
            scratch_shapes=[pltpu.SMEM((batch_size, ), jnp.int32)],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule",
    )(cu_q_lens, kv_lens, page_indices, jnp.array([0, 0, num_seq]))
