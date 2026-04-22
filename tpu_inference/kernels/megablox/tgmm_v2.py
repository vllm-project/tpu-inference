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

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox.gmm_v2 import (
    Dimensions,
    GmmConfigs,
    InputConfigs,
    MetadataRef,
    TileSizes,
    TileTgmmFn,
    align_to,
    fill_metadata,
    get_metadata,
    get_scope_name,
)


def calculate_tgmm_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
) -> TileSizes:
  """Calculate optimal tile sizes for TGMM kernel."""
  # In tgmm, we calculate lhs.T @ dout which doesn't require quantization.
  # Since we use it in MOE, the m can be dynamic and small. So we don't
  # want it to be too big.
  bf16_bf16_tile_m = 128
  tile_m = min(bf16_bf16_tile_m, dims.size_m)
  tile_m = max(tile_m, dims.size_lhs_sublane)

  num_k_tiles = num_n_tiles = 1
  num_lanes = pltpu.get_tpu_info().num_lanes
  tile_n = align_to(dims.size_n, num_lanes)
  # To avoid stalling MXU, we add some buffer room where tile_n cannot go
  # smaller than 2x of mxu_column_size.
  tile_n_lower_bound = pltpu.get_tpu_info().mxu_column_size * 2
  tile_n_lower_bound = min(tile_n_lower_bound, dims.size_n)
  tile_k = align_to(dims.size_k, num_lanes)

  def within_vmem_limit(tile_m, tile_k, tile_n):
    acc_bytes = jax.dtypes.itemsize_bits(acc_dtype) // 8
    out_bytes = jax.dtypes.itemsize_bits(out_dtype) // 8
    lhs_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    rhs_bytes = jax.dtypes.itemsize_bits(rhs_cfgs.dtype) // 8
    num_buffers = 2
    budget = tile_k * tile_n * (acc_bytes + num_buffers * out_bytes) + num_buffers * (tile_m*tile_k*lhs_bytes + tile_m*tile_n*rhs_bytes)
    return budget <= vmem_limit_bytes

  prev_tile_n = tile_n
  while not within_vmem_limit(tile_m, tile_k, tile_n):
    num_n_tiles += 1
    # The reason why we do "tile_n * num_n_tiles must cover size_n." is
    # tile_n must be a multiple of num_lanes and
    # tile_n * num_n_tiles must cover size_n.
    tile_n = align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles
    # If size_n is small and awkwardly sized (e.g., size_n=100, num_lanes=128),
    # align_to(100, N*128) // N can get stuck at a constant value (128) as N
    # grows. If that constant value is above the floor and budget still
    # doesn't fit, the loop never terminates. That's why we need to check if
    # "tile_n >= prev_tile_n".
    if tile_n < tile_n_lower_bound or tile_n >= prev_tile_n:
      break
    prev_tile_n = tile_n

  if tile_n >= tile_n_lower_bound and within_vmem_limit(tile_m, tile_k, tile_n):
    return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

  if tile_n < tile_n_lower_bound:
    num_n_tiles -= 1
    tile_n = align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles

  prev_tile_k = tile_k
  while not within_vmem_limit(tile_m, tile_k, tile_n):
    num_k_tiles += 1
    tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles
    if tile_k < num_lanes or tile_k >= prev_tile_k:
      break
    prev_tile_k = tile_k


  if tile_k < num_lanes:
    num_k_tiles -= 1
    tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

  if not within_vmem_limit(tile_m, tile_k, tile_n):
    raise ValueError(f"Could not find valid tile sizes for tgmm. dims={dims}, tiles=({tile_m},{tile_k},{tile_n}), vmem={vmem_limit_bytes}")
  return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def make_tgmm_configs(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [m, n]
    group_sizes: jax.Array,
    num_actual_groups: int,
    *,
    tile_info: TileSizes | TileTgmmFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype | None,
):
  """Fills the GMM config for the TGMM kernel."""
  assert lhs.shape[0] == rhs.shape[0], f'lhs and rhs m-dim mismatch: {lhs.shape[0]}!={rhs.shape[0]} {lhs.shape} vs {rhs.shape}'
  size_m, size_k = lhs.shape
  _, size_n = rhs.shape
  # size_lhs_sublane is used in tgmm_inner_kernel to set the
  # (m/size_lhs_sublane, size_lhs_sublane, ...) reshape tile used on the m-axis
  # for both 'tiled_lhs_ref' and 'tiled_rhs_ref'.
  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
  size_lhs_sublane = min(size_lhs_sublane, size_m)
  size_rhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(rhs.dtype)
  size_rhs_sublane = min(size_rhs_sublane, size_m)
  assert size_lhs_sublane == size_rhs_sublane, f"size_lhs_sublane should be the same as size_rhs_sublane {lhs.dtype=}, {rhs.dtype=}"
  dims = Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=num_actual_groups,  # weight.shape[0]
      size_lhs_group=group_sizes.shape[0],
      size_lhs_sublane=size_lhs_sublane
  )

  rhs_cfgs = InputConfigs(
      quant_dtype=None,
      quant_block_size=-1,
      dtype=rhs.dtype,
  )
  lhs_cfgs = InputConfigs(
      quant_dtype=None,
      quant_block_size=-1,
      dtype=lhs.dtype,
  )

  fuse_act = None  # fuse_act has to be None in tgmm.
  if acc_dtype is None:
    acc_dtype = jnp.float32.dtype
  if isinstance(tile_info, TileSizes):
    tiles = tile_info
  else:
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, out_dtype, acc_dtype)

  return GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_cfgs=lhs_cfgs,
      rhs_cfgs=rhs_cfgs,
      out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype),
      # GMM's 'zero_init' zeros unvisited m-rows via DMA, which doesn't apply to tgmm's [num_groups, k, n] output. The actual zero-initialization for tgmm accumulation happens at the 'pallas_call' level
      zero_init=False,
      fuse_act=fuse_act,
  )


def tgmm_inner_kernel(
    tiled_lhs_ref: jax.Array, # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: jax.Array, # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    tiled_out_ref: jax.Array, # [None, tile_k, tile_n]
    # scratch
    acc_ref: jax.Array,  # for accumulation [tile_k, tile_n]
    metadata_ref: MetadataRef, # contains gm_id_to_group_id and gm_id_to_m_offset in SMEM.
    *,
    cfgs: GmmConfigs,
):
  # NB: grid=(num_n, num_k, num_gm)
  tiled_lhs_ref = tiled_lhs_ref.reshape(-1, tiled_lhs_ref.shape[-1])
  tiled_rhs_ref = tiled_rhs_ref.reshape(-1, tiled_rhs_ref.shape[-1])
  gm_id = pl.program_id(2)

  def _matmul(is_new_group: bool, is_group_changing: bool):
    if is_new_group:
      acc_ref[...] = jnp.zeros_like(acc_ref)

    # Mask out invalid rows in the LHS/RHS tiles.
    # The DMA loads tiles aligned to sublane boundaries, but the actual group
    # data may not start/end on those boundaries.
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset
    lhs_iota = lax.broadcasted_iota(jnp.int32, tiled_lhs_ref.shape, 0)
    lhs_mask = jnp.logical_and(m_start_local <= lhs_iota, lhs_iota < m_end_local)
    lhs_masked = jnp.where(lhs_mask, tiled_lhs_ref[...], 0)
    # masking both lhs and rhs shouldn't be necessary. as long as there isn't
    # NaNs, we can just mask out one of either argument and get the same
    # numeric result.

    acc = acc_ref[...] + jax.lax.dot_general(
        lhs_masked,
        tiled_rhs_ref[...],
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    if is_group_changing:
      tiled_out_ref[...] = acc.astype(tiled_out_ref.dtype)
    acc_ref[...] = acc

  @jax.named_scope("matmul_new_group_and_changing")
  def matmul_new_group_and_changing():
    _matmul(is_new_group=True, is_group_changing=True)

  @jax.named_scope("matmul_new_group")
  def matmul_new_group():
    _matmul(is_new_group=True, is_group_changing=False)

  @jax.named_scope("matmul")
  def matmul():
    _matmul(is_new_group=False, is_group_changing=False)

  @jax.named_scope("matmul_group_changing")
  def matmul_group_changing():
    _matmul(is_new_group=False, is_group_changing=True)

  prev_gm_id = jnp.where(gm_id > 0, gm_id - 1, 0)
  is_first_gm = gm_id == 0
  group_id_changed = (
      metadata_ref.gm_id_to_group_id[gm_id]
      != metadata_ref.gm_id_to_group_id[prev_gm_id]
  )
  new_group = jnp.logical_or(is_first_gm, group_id_changed)

  is_last_gm = gm_id == (pl.num_programs(2) - 1)
  next_gm_id = jnp.where(is_last_gm, gm_id, gm_id + 1)
  next_group_id = metadata_ref.gm_id_to_group_id[next_gm_id]
  cur_group_id = metadata_ref.gm_id_to_group_id[gm_id]
  group_is_changing = jnp.logical_or(is_last_gm, cur_group_id != next_group_id)

  lax.cond(
      new_group,
      lambda: lax.cond(
          group_is_changing,
          matmul_new_group_and_changing,
          matmul_new_group,
      ),
      lambda: lax.cond(
          group_is_changing,
          matmul_group_changing,
          matmul,
      ),
  )


class TgmmIndexMaps:

  def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs

  def lhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, k_id)

  def rhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, n_id)

  def out_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, k_id, n_id)

def generate_tgmm_block_specs(
        metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
  """Generates block specs for the given lhs, rhs, and out refs."""
  index_map = TgmmIndexMaps(metadata_ref, cfgs)
  # NB: in tgmm, LHS is reshaped from (M, K) to (-1, size_lhs_sublane, K) so that DMA transfers are aligned to sublane boundaries. The first dimension after this reshape has size tile_m // size_lhs_sublane — i.e., the number of "sublane-rows" in a tile.
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m //
                                     cfgs.dims.size_lhs_sublane)
  lhs_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      index_map.lhs_index_map,
  )
  rhs_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      index_map.rhs_index_map,
  )
  out_block_spec = pl.BlockSpec(
      (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
      index_map.out_index_map,
  )

  return (lhs_block_spec, rhs_block_spec), out_block_spec

def zero_out_start(
    lhs_group_sizes_ref,  # int32[size_lhs_group]
    group_offset_ref,  # int32[1]
    out_ref,  # [num_actual_groups, k, n]
    zero_ref,  # [tile_zero_k, num_lanes]
    semaphore_ref,  # [1]
):
  """
  If group_sizes[i]==0, we want to zero out drhs[i].
  We could have initialized the output to be zero before the kernel starts.
  But that process turns out to be costly in performance.
  """
  zero_ref[...] = jnp.zeros_like(zero_ref)
  num_actual_groups = out_ref.shape[0]
  tile_zero_k = zero_ref.shape[0]
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert out_ref.shape[2] % num_lanes == 0
  dma_issued = jnp.bool_(False)

  def fill_zero(local_group_id):
    dma_issued = jnp.bool_(False)
    for i in range(pl.cdiv(out_ref.shape[1], tile_zero_k)):
      for j in range(out_ref.shape[2]//num_lanes):
        dma_issued = jnp.bool_(True)
        size_k_to_copy = min(tile_zero_k, out_ref.shape[1] - i*tile_zero_k)
        pltpu.make_async_copy(
            src_ref=zero_ref.at[pl.ds(0, size_k_to_copy)],
            dst_ref=out_ref.at[local_group_id, pl.ds(i*tile_zero_k, size_k_to_copy), pl.ds(j*num_lanes, num_lanes)],
            sem=semaphore_ref.at[0],
        ).start(priority=1)
    return dma_issued

  for i in range(len(lhs_group_sizes_ref)):
    local_group_id = i - group_offset_ref[0]
    should_zero = jnp.logical_and(
        jnp.logical_and(i >= group_offset_ref[0], i < group_offset_ref[0] + num_actual_groups),
        lhs_group_sizes_ref[i] == 0,
    )
    dma_issued |= lax.cond(should_zero, fill_zero, lambda local_group_id: jnp.bool_(False), local_group_id)
  
  return dma_issued

def zero_out_end(
    out_ref: jax.Array,  # [num_actual_groups, k, n]
    semaphore_ref: jax.Array,  # [1]
    zero_out_dma_issued: bool,
):
  # if no zero_out dma is issued, the sem will be 0 so we shouldn't wait.

  def wait_for_dms():
    # Here src_ref should be the same as dst_ref. You just need any valid slice
    # to attach the .wait() to. The only thing that matters is that .wait() is
    # called on the same semaphore, which ensures all prior .start() calls have
    # completed. Also, doing one '.wait()' is sufficient because we only have 
    # one sem. Each .start() increments the semaphore's pending count, and the
    # hardware decrements it as each DMA completes. The final .wait() blocks
    # until the count reaches zero.
    pltpu.make_async_copy(
        src_ref=out_ref.at[0],
        dst_ref=out_ref.at[0],
        sem=semaphore_ref.at[0],
    ).wait()
  lax.cond(zero_out_dma_issued, wait_for_dms, lambda: None)

def tgmm_kernel_main(
    lhs_group_sizes_ref,  # int32[size_lhs_group]
    group_offset_ref,  # int32[1]
    lhs_ref,  # [m, k]
    rhs_ref,  # [m, n]
    out_ref,  # [num_actual_groups, k, n]
    # scratch memory
    acc_ref: jax.Array,  # [tile_k, tile_n]
    metadata_ref: MetadataRef, # contains gm_id_to_group_id and gm_id_to_m_offset in SMEM.
    zero_ref: jax.Array,  # [tile_zero_k, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    *, cfgs,
):
  """ Entry point for TGMM kernel.

  Args:
    lhs_group_sizes_ref: Reference to the group sizes of GMM lhs.
    group_offset_ref: Reference to the group offset.
    lhs_ref: Reference to the lhs.
    rhs_ref: Reference to the rhs.
    out_ref: Reference to the out.
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    zero_ref: Scratch memory for storing zero values used in initialization.
    semaphore_ref: Semaphore for zero initialization DMAs.
    cfgs: GmmConfigs.
  """
  zero_init_dma_issued = zero_out_start(
      lhs_group_sizes_ref,
      group_offset_ref,
      out_ref,
      zero_ref,
      semaphore_ref,
  )
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.dims.size_n, cfgs.tiles.tile_n)
  num_gm = fill_metadata(
      lhs_group_sizes_ref,
      group_offset_ref,
      metadata_ref,
      cfgs=cfgs,
  )

  in_specs, out_specs = generate_tgmm_block_specs(metadata_ref, cfgs)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(tgmm_inner_kernel, cfgs=cfgs),
      grid=(num_n, num_k, num_gm),
      in_specs=in_specs,
      out_specs=out_specs,
  )
  lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
  rhs_in = rhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, rhs_ref.shape[-1])
  scratches = [acc_ref, metadata_ref]
  pipeline_fn(lhs_in, rhs_in, out_ref, scratches=scratches)
  zero_out_end(
      out_ref,
      semaphore_ref,
      zero_init_dma_issued,
  )

@functools.partial(
    jax.jit,
    static_argnames=[
        "num_actual_groups",
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
    ],
)
def _tgmm_v2_impl(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_m, size_n]
    group_sizes: jax.Array,
    num_actual_groups: int,
    group_offset: jax.Array | None = None,
    *,
    tile_info: TileSizes | TileTgmmFn = calculate_tgmm_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
):
  # Compute grad_rhs=lhs[sizes[i-1]:sizes[i], :].T @ rhs[sizes[i-1]:sizes[i], :]
  # aka grad_rhs = lhs.T @ grad

  # Step 1. delete precision, normalize group_offset, set vmem_limit_bytes
  del precision
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  # Step 2. Make gmm configs (create a 'Dimensions' and a 'GmmConfigs):
  cfgs = make_tgmm_configs(
      lhs,
      rhs,
      group_sizes,
      num_actual_groups,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=preferred_element_type,
      acc_dtype=acc_dtype,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles

  # 4. Form pl.pallas_call calling tgmm_kernel_main
  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = align_to(dims.size_n, num_lanes)
  out_init = jax.ShapeDtypeStruct((num_actual_groups, dims.size_k, aligned_n), cfgs.out_dtype)
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  scratch_shapes = [
      # acc_ref
      pltpu.VMEM((tiles.tile_k, tiles.tile_n), cfgs.acc_dtype),
      # metadata_ref
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]
  
  # Prepare zero initializing the drhs[i, :, :] where the group_size[i] is 0.
  target_zero_ref_bytes = 2 * 1024 * 1024
  out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
  tile_zero_k = target_zero_ref_bytes // num_lanes // out_bytes
  tile_zero_k = min(tile_zero_k, dims.size_k)
  size_out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
  tile_zero_k = (tile_zero_k // size_out_sublane) * size_out_sublane
  assert tile_zero_k > 0
  scratch_shapes += [
      pltpu.VMEM((tile_zero_k, num_lanes), cfgs.out_dtype),
      pltpu.SemaphoreType.DMA((1,)),
  ]

  return pl.pallas_call(
      functools.partial(tgmm_kernel_main, cfgs=cfgs),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM), # x
              pl.BlockSpec(memory_space=pltpu.HBM), # dout
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=get_scope_name(cfgs),
      # the metadata here is for profiling, debugging, and cost modeling. It does not affect the kernel's computation.
      metadata=get_metadata(cfgs),
  )(group_sizes, group_offset, lhs, rhs)[:, :, :dims.size_n]
