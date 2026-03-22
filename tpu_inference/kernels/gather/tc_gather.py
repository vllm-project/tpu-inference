"""TensorCore-based Pallas ragged gather kernel."""

import dataclasses
import functools

import jax
from jax import numpy as jnp
from jax import tree_util
from jax._src.pallas.mosaic import pipeline
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NUM_BUFFERS = 2


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
      buffer_count: int = _NUM_BUFFERS,
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
        self.window_ref.at[wait_slot, : self.block_size, :, :],
        self.window_ref.at[wait_slot, : self.block_size, :, :],
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
  block_idx = pl.program_id(0)

  def _inner_kernel(is_first_block: bool, is_last_block: bool):
    global_block_start = (
        pl.multiple_of(aligned_start_ref[0], 8) + block_idx * block_size
    )

    local_start = local_start_ref[0]
    local_end = end_idx_ref[0] - global_block_start

    for i in range(block_size):
      global_token_idx = global_block_start + i
      mod_8 = idx_mod_8_ref[global_token_idx]
      row_indices = jnp.broadcast_to(mod_8, (8, 128)).astype(jnp.int32)

      if is_first_block and is_last_block:
        is_valid_mask = (i >= local_start) & (i < local_end)
      elif is_first_block:
        is_valid_mask = i >= local_start
      elif is_last_block:
        is_valid_mask = i < local_end
      else:
        is_valid_mask = None

      # Iterate over 128-width chunks of hidden dims to reuse row_indices.
      hidden_dim = x_vmem.shape[-1]
      for c in range(0, hidden_dim, 128):
        cols = pl.ds(c, 128)
        extracted = jnp.take_along_axis(
            x_vmem[i, :, cols].astype(jnp.float32),
            row_indices,
            axis=0,
        )
        if is_valid_mask is not None:
          result = jnp.where(
              is_valid_mask, extracted[0], jnp.zeros_like(extracted[0])
          )
        else:
          result = extracted[0]
        o_vmem[i, cols] = result.astype(o_vmem.dtype)

  @jax.named_scope("gather_first_last")
  def gather_first_last():
    _inner_kernel(is_first_block=True, is_last_block=True)

  @jax.named_scope("gather_first")
  def gather_first():
    _inner_kernel(is_first_block=True, is_last_block=False)

  @jax.named_scope("gather")
  def gather():
    _inner_kernel(is_first_block=False, is_last_block=False)

  @jax.named_scope("gather_last")
  def gather_last():
    _inner_kernel(is_first_block=False, is_last_block=True)

  is_first_block = block_idx == 0
  is_last_block = block_idx == (pl.num_programs(0) - 1)

  jax.lax.cond(
      is_first_block,
      lambda: jax.lax.cond(
          is_last_block,
          gather_first_last,
          gather_first,
      ),
      lambda: jax.lax.cond(
          is_last_block,
          gather_last,
          gather,
      ),
  )


def tensorcore_gather(
    x: jax.Array,
    indices: jax.Array,
    start_idx: int | jax.Array | None = None,
    end_idx: int | jax.Array | None = None,
    block_size: int = 32,
) -> jax.Array:
  """Gathers a range of tokens from x using TensorCore."""
  assert (
      block_size % 8 == 0
  ), f"block_size must be divisible by 8, got {block_size}"
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
        f" ({block_size})."
    )

  aligned_start = (start_idx // block_size) * block_size
  aligned_end = pl.cdiv(end_idx, block_size) * block_size
  num_blocks = pl.cdiv(aligned_end - aligned_start, block_size)
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
            pipeline_mode=pl.Buffered(buffer_count=_NUM_BUFFERS),
        ),
    ]

    pipeline_func = pipeline.emit_pipeline(
        inner_kernel_partial,
        grid=(num_blocks_ref[0],),
        in_specs=_in_specs,
        out_specs=_out_specs,
    )

    x_alloc = GatherBufferedRef.create(
        spec=_in_specs[0],
        source_array=x_hbm_ref,
        block_size=block_size,
    )

    o_alloc = pipeline.BufferedRef.create(
        spec=_out_specs[0],
        dtype_or_type=pipeline._ref_to_value_aval(o_hbm_ref),
        buffer_type=pipeline.BufferType.OUTPUT,
        buffer_count=_NUM_BUFFERS,
        grid_rank=1,
        source_memory_space=pltpu.HBM,
    )

    def _run(allocs):
      pipeline_func(
          (x_hbm_ref, idx_aligned_ref, aligned_start_ref),
          o_hbm_ref,
          allocations=allocs,
      )

    pl.run_scoped(_run, (x_alloc, o_alloc))

  x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
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
  to_arr = lambda x: jnp.array([x], dtype=jnp.int32)

  res = pl.pallas_call(
      gather_kernel,
      out_shape=jax.ShapeDtypeStruct((total_indices, hidden_dim), dtype),
      grid_spec=grid_spec,
      name=f"tc_gather_hidden{hidden_dim}_numidx{total_indices}_block{block_size}",
      metadata={
          "block_size": str(block_size),
          "hidden_dim": str(hidden_dim),
          "total_indices": str(total_indices),
          "dtype": str(dtype),
          "num_buffers": str(_NUM_BUFFERS),
      },
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
