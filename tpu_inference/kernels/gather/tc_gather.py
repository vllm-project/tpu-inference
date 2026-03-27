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
"""TensorCore-based Pallas ragged gather kernel."""

import dataclasses
import functools

import jax
from jax import numpy as jnp
from jax import tree_util
from jax._src.pallas.mosaic import pipeline
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NUM_BUFFERS = 3


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GatherBufferedRef(pipeline.BufferedRef):
    """Custom BufferedRef managing async DMA for gathering operations.

  Overrides `copy_in` and `wait_in` to break standard contiguous block
  iteration. Instead, it dynamically orchestrates DMA transfers using
  `idx_aligned_ref` to fetch 8-element contiguous segments from HBM. The fetched
  data sits in a `(block_size, 8, hidden_dim)` VMEM scratch buffer.
  """

    block_size: int = dataclasses.field(metadata={"static": True}, default=0)

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        source_array: jax.Array,
        block_size: int,
        buffer_count: int = 2,
    ):
        standard_ref = pipeline.BufferedRef.create(
            spec=spec,
            dtype_or_type=pipeline._ref_to_value_aval(source_array),
            buffer_type=pipeline.BufferType.INPUT,
            buffer_count=buffer_count,
            grid_rank=1,
            source_memory_space=pltpu.HBM,
        )
        return cls.from_ref(
            standard_ref,
            block_size=block_size,
        )

    @classmethod
    def from_ref(
        cls,
        ref: pipeline.BufferedRef,
        *,
        block_size: int = 0,
    ):
        return cls(
            block_size=block_size,
            **{
                f.name: getattr(ref, f.name)
                for f in dataclasses.fields(pipeline.BufferedRef)
            },
        )

    def copy_in(self, src_ref, grid_indices):
        x_hbm_ref, idx_aligned_ref, aligned_start_ref = src_ref
        slot = self.current_copy_in_slot
        block_idx = grid_indices[0]

        global_block_start = aligned_start_ref[0] + block_idx * self.block_size

        for i in range(self.block_size):
            global_token_idx = global_block_start + i
            idx_aligned = pl.multiple_of(idx_aligned_ref[global_token_idx], 8)

            assert self.sem_recvs is not None
            pltpu.make_async_copy(
                x_hbm_ref.at[pl.ds(idx_aligned, 8), :],
                self.window_ref.at[slot, i, :, :],
                self.sem_recvs.at[slot],
            ).start()

    def wait_in(self, src_ref, grid_indices):
        wait_slot = self.current_wait_in_slot

        assert self.sem_recvs is not None
        pltpu.make_async_copy(
            self.window_ref.at[wait_slot, :self.block_size, :, :],
            self.window_ref.at[wait_slot, :self.block_size, :, :],
            self.sem_recvs.at[wait_slot],
        ).wait()


def inner_kernel(
    block_size: int,
    aligned_start_ref,
    end_idx_ref,
    local_start_ref,
    idx_mod_8_ref,
    x_vmem,
    o_vmem,
):
    """Inner kernel to perform the actual gather operation for a single block.

    Args:
        block_size: The number of elements to process per block.
        aligned_start_ref: The absolute start index, rounded down to the nearest
        multiple of `block_size`.
        end_idx_ref: The absolute end index. Used to mask out-of-bounds calculations
        in the final execution block.
        local_start_ref: Offset (0-7) between the true `start_idx` and
        `aligned_start_ref`. Used to mask out invalid elements inside the very
        first execution block.
        idx_mod_8_ref: A tensor of shape (total_indices + block_size,) that contains
        `indices & 7`, providing the local sub-row offsets within the 8-element
        chunks physically loaded from HBM.
        x_vmem: 8-element chunks of prefetched data in VMEM with shape (block_size,
        8, hidden_dim).
        o_vmem: Output tensor of shape (block_size, hidden_dim).
    """

    block_size, num_sublanes, hidden_dim = x_vmem.shape
    block_idx = pl.program_id(0)

    global_block_start = aligned_start_ref[0] + block_idx * block_size

    for start in range(0, block_size, num_sublanes):
        end = start + num_sublanes

        out = jnp.zeros((num_sublanes, hidden_dim), dtype=jnp.float32)
        for sublane in range(num_sublanes):
            i = start + sublane
            global_token_idx = global_block_start + i
            mod_8 = idx_mod_8_ref[global_token_idx]

            shift = sublane - mod_8
            x = pltpu.roll(x_vmem[i].astype(jnp.float32), shift=shift, axis=0)

            mask = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) == sublane
            out = jnp.where(mask, x, out)

        o_vmem[start:end] = out.astype(o_vmem.dtype)


def tensorcore_gather(
    x: jax.Array,
    indices: jax.Array,
    start_idx: int | jax.Array | None = None,
    end_idx: int | jax.Array | None = None,
    block_size: int = 32,
) -> jax.Array:
    """Gathers a range of tokens from x using TensorCore."""
    assert (block_size %
            8 == 0), f"block_size must be divisible by 8, got {block_size}"
    total_indices = indices.shape[0]
    hidden_dim = x.shape[1]
    dtype = x.dtype

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = total_indices

    if total_indices % block_size != 0:
        raise ValueError(
            f"total_indices ({total_indices}) must be a multiple of block_size"
            f" ({block_size}).")

    aligned_start = (start_idx // block_size) * block_size
    aligned_end = pl.cdiv(end_idx, block_size) * block_size
    num_blocks = pl.cdiv(aligned_end - aligned_start, block_size)
    num_blocks = jnp.where(start_idx == end_idx, 0, num_blocks)
    local_start = start_idx - aligned_start

    idx_aligned_padded = jnp.pad(indices & ~7, (0, block_size))
    idx_mod_8_padded = jnp.pad(indices & 7, (0, block_size))

    @jax.named_scope("tensorcore_gather_kernel")
    def gather_kernel(
        num_blocks_ref,
        aligned_start_ref,
        end_idx_ref,
        local_start_ref,
        idx_aligned_ref,
        idx_mod_8_ref,
        x_hbm_ref,
        o_hbm_ref,
    ):
        """Executes the Gather pipeline over a perfectly tiled local execution grid.

    Args:
      num_blocks_ref: Scalar value of the number of blocks to process.
      aligned_start_ref: The absolute start index, rounded down to the nearest
        multiple of `block_size`.
      end_idx_ref: The absolute end index. Used to mask out-of-bounds
        calculations in the final execution block.
      local_start_ref: Offset (0-7) between the true `start_idx` and
        `aligned_start_ref`. Used to mask out invalid elements inside the very
        first execution block.
      idx_aligned_ref: A tensor of shape (total_indices + block_size,) that
        contains `indices & ~7`. Used to dispatch aligned HBM fetches for each
        token.
      idx_mod_8_ref: A tensor of shape (total_indices + block_size,) that
        contains `indices & 7`, providing the local sub-row offsets within the
        8-element chunks physically loaded from HBM.
      x_hbm_ref: The input tensor referenced in HBM logic.
      o_hbm_ref: The output tensor referenced in HBM logic.
    """
        inner_kernel_partial = functools.partial(
            inner_kernel,
            block_size,
            aligned_start_ref,
            end_idx_ref,
            local_start_ref,
            idx_mod_8_ref,
        )

        _in_specs = [
            pl.BlockSpec(
                index_map=lambda *idx: idx,
                memory_space=pltpu.VMEM,
                block_shape=(block_size, 8, hidden_dim),
                pipeline_mode=pl.Buffered(buffer_count=_NUM_BUFFERS),
            ),
        ]

        def o_index_map(i):
            start_block_idx = aligned_start_ref[0] // block_size
            return (start_block_idx + i, 0)

        _out_specs = [
            pl.BlockSpec(
                index_map=o_index_map,
                memory_space=pltpu.VMEM,
                block_shape=(block_size, hidden_dim),
            ),
        ]

        pipeline_func = pipeline.emit_pipeline(
            inner_kernel_partial,
            grid=(num_blocks_ref[0], ),
            in_specs=_in_specs,
            out_specs=_out_specs,
        )

        x_alloc = GatherBufferedRef.create(
            spec=_in_specs[0],
            source_array=x_hbm_ref,
            block_size=block_size,
            buffer_count=_NUM_BUFFERS,
        )

        o_alloc = pipeline.BufferedRef.create(
            spec=_out_specs[0],
            dtype_or_type=pipeline._ref_to_value_aval(o_hbm_ref),
            buffer_type=pipeline.BufferType.OUTPUT,
            grid_rank=1,
            source_memory_space=pltpu.HBM,
            buffer_count=2,
        )

        def _run(allocs):
            pipeline_func(
                (x_hbm_ref, idx_aligned_ref, aligned_start_ref),
                o_hbm_ref,
                allocations=allocs,
            )

        pl.run_scoped(_run, (x_alloc, o_alloc))

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=6,
        in_specs=[
            pl.BlockSpec(
                memory_space=pltpu.HBM,
                pipeline_mode=pl.Buffered(buffer_count=_NUM_BUFFERS),
            ),
        ],
        out_specs=pl.BlockSpec(
            memory_space=pltpu.HBM,
            pipeline_mode=pl.Buffered(buffer_count=_NUM_BUFFERS),
        ),
        scratch_shapes=[],
    )

    def to_arr(x):
        return jnp.array([x], dtype=jnp.int32)

    res = pl.pallas_call(
        gather_kernel,
        out_shape=jax.ShapeDtypeStruct((total_indices, hidden_dim), dtype),
        grid_spec=grid_spec,
        name=
        f"tc_gather_hidden{hidden_dim}_numidx{total_indices}_block{block_size}",
        metadata={
            "block_size": str(block_size),
            "hidden_dim": str(hidden_dim),
            "total_indices": str(total_indices),
            "dtype": str(dtype),
            "num_buffers": str(_NUM_BUFFERS),
        },
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes *
                                 0.7),
            disable_bounds_checks=True,
        ),
    )(
        to_arr(num_blocks),
        to_arr(aligned_start),
        to_arr(end_idx),
        to_arr(local_start),
        idx_aligned_padded,
        idx_mod_8_padded,
        x,
    )

    return res
