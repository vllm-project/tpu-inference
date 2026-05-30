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
"""Hierarchical Recursive Doubling Reduce-Scatter Implementation."""

import dataclasses
import math
from typing import Any, Callable


import jax
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu

import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RemoteWaitBufferedRef(pltpu.BufferedRef):
  """Subclass of BufferedRef that implements semaphore-synchronized memory copies.

  Used to wait for remote device writes before initiating local HBM-to-VMEM
  copies.
  """

  index_fn_with_recv_sem: Callable[..., Any] | None = dataclasses.field(
      metadata={"static": True}, default=None
  )

  def __post_init__(self):
    pass

  @classmethod
  def from_ref(
      cls,
      ref: pltpu.BufferedRef,
      *,
      index_fn_with_recv_sem: Callable | None = None,
  ):
    return cls(
        index_fn_with_recv_sem=index_fn_with_recv_sem,
        **{
            f.name: getattr(ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  @property
  def has_indirect(self) -> bool:
    return True

  def copy_in(self, src_ref, grid_indices):
    if self.index_fn_with_recv_sem is None:
      super().copy_in(src_ref, grid_indices)
      return
    slot = self.current_copy_in_slot
    hbm_index, sem, size = self.index_fn_with_recv_sem(grid_indices, src_ref)

    vmem_index = (slot, slice(None), pl.ds(0, size))
    if sem is not None:
      pltpu.make_async_copy(
          self.window_ref.at[vmem_index], self.window_ref.at[vmem_index], sem
      ).wait()

    hbm_array_ref = (
        src_ref[0] if isinstance(src_ref, (tuple, list)) else src_ref
    )
    assert self.sem_recvs is not None
    pltpu.make_async_copy(
        hbm_array_ref.at[hbm_index],
        self.window_ref.at[vmem_index],
        self.sem_recvs.at[slot],
    ).start()

  def wait_in(self, src_ref, grid_indices):
    if self.index_fn_with_recv_sem is None:
      super().wait_in(src_ref, grid_indices)
      return
    wait_slot = self.current_wait_in_slot
    _, _, size = self.index_fn_with_recv_sem(grid_indices, src_ref)

    vmem_index = (wait_slot, slice(None), pl.ds(0, size))
    assert self.sem_recvs is not None
    pltpu.make_async_copy(
        self.window_ref.at[vmem_index],
        self.window_ref.at[vmem_index],
        self.sem_recvs.at[wait_slot],
    ).wait()


# ================================================================================
#                          CHUNK PARTITIONING MAP
# ================================================================================
#         |<----------------------- hidden_size_dim ------------------------>|
#         |<----------- mb_size ----------->|                                |
#         |<-- hc_chunk_size ->|            |                                |
#         +--------------------+------------+---------------+----------------+ ---
#       ^ |          |         |            |               |                |  ^
#       | |  Chunk   |  Chunk  |  MB1 Slice |   MB2 Slice   |   MB3 Slice    |  |
# seq_cs| |          |         |            |               |                |  | seqlen
#       v +--------------------+------------+---------------+----------------+  |
#       ^ |                                 |               |                |  |
# seq_cs| |      Device Slice 1             |               |                |  |
#       v +---------------------------------+---------------+----------------+  v
#                                                                              ---
# ================================================================================
# (Legend: seq_cs = seq_chunk_size, seqlen = local_seq_len)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RsDimensions:
  # Number of physical TPU chips in the mesh (num_devices // 2)
  num_chips: int
  # ICI hypercube logical network dimensions (log2(num_chips))
  num_hcube_dims: int
  # Pipelining unrolling factor for overlapping ALU/DMA
  num_micro_batches: int
  # Total hidden size dimension (e.g., 4096)
  hidden_size_dim: int
  # Local sequence slice per device (seq_len // num_devices)
  seq_chunk_size: int
  # Phase 1 (C2C) slice size (hidden_size_dim // num_micro_batches)
  mb_size: int
  # Phase 2, Hypercube chunk slice size (align(mb_size // num_hcube_dims, 128))
  hc_chunk_size: int


def _get_capped_bounds(start, length, max_size):
  """Calculates capped start and size for a slice to prevent out-of-bounds."""
  capped_start = min(start, max_size)
  capped_end = min(start + length, max_size)
  return capped_start, capped_end - capped_start


class Topology:
  """Abstracts JAX axis index calls and partner logic."""

  def __init__(self, axis_name: str = "x"):
    self.cur_id = jax.lax.axis_index(axis_name)
    self.cur_chip_id = self.cur_id // 2
    self.cur_chiplet_bit = self.cur_id % 2
    self.partner_id = jax.lax.select(
        self.cur_chiplet_bit == 0, self.cur_id + 1, self.cur_id - 1
    )

  def get_device_id(self, chip_id, chiplet_bit):
    return chip_id * 2 + chiplet_bit

  def get_neighbor_chip_id(self, dim):
    return self.cur_chip_id ^ (1 << dim)

  def get_phase2_neighbor_device_id(self, dim):
    return self.get_device_id(
        self.get_neighbor_chip_id(dim), self.cur_chiplet_bit
    )


class ChunkLocator:
  """Encapsulates all algebraic index math and BlockSpec mapping builders.

  Hides the bitwise operations, keeping the main loop strictly focused on
  the orchestration of the reducion algorithm.
  """

  def __init__(
      self,
      dims: RsDimensions,
      topo: Topology,
  ):
    self.dims = dims
    self.topo = topo
    self.mb_stride = dims.num_hcube_dims * dims.hc_chunk_size

  def get_slice(self, chunk_idx, start, size):
    """Returns a 2D HBM slice for a given chunk index and hidden dimension range."""
    return (
        pl.ds(chunk_idx * self.dims.seq_chunk_size, self.dims.seq_chunk_size),
        pl.ds(start, size),
    )

  def get_phase1_chunk_idx(self, device_id, chip_idx):
    """Locates the chunk owned by a device in Phase 1.

    In Phase 1, Device `device_id` owns the chunk corresponding to its
    chiplet bit (parity) in Chip `chip_idx`.

    Args:
      device_id: The global logical ID of the device.
      chip_idx: The index of the physical chip.

    Returns:
      The global chunk index (0 to num_devices - 1).
    """
    chiplet_bit = device_id % 2
    return chip_idx * 2 + chiplet_bit

  def get_phase1_chunk_idxes(self, device_id):
    """Returns the list of all chunk indices owned by a device in Phase 1.

    Args:
      device_id: The global logical ID of the device.

    Returns:
      A list of global chunk indices.
    """
    chiplet_bit = device_id % 2
    return [
        chip_idx * 2 + chiplet_bit for chip_idx in range(self.dims.num_chips)
    ]

  def get_phase2_chunk_idx(self, device_id, step_idx, op_idx, hcube_dim_idx):
    """Locates the chunk owned by a device in Phase 2.

    In Phase 2, ownership is dynamic and depends on the hypercube step,
    the parallel op index, and the dimension index.

    Args:
      device_id: The global logical ID of the device.
      step_idx: The current step in the hypercube reduction.
      op_idx: The index of the parallel reduction operation.
      hcube_dim_idx: The current logical dimension of the hypercube.

    Returns:
      The global chunk index (0 to num_devices - 1).
    """
    dim = (hcube_dim_idx + step_idx) % self.dims.num_hcube_dims
    chip_id = device_id // 2
    my_dim_bit = (chip_id >> dim) & 1

    prev_dims = [
        (hcube_dim_idx + j) % self.dims.num_hcube_dims for j in range(step_idx)
    ]
    future_dims = [
        (hcube_dim_idx + j) % self.dims.num_hcube_dims
        for j in range(step_idx + 1, self.dims.num_hcube_dims)
    ]

    my_base_chunk_idx = self.get_hcube_chunk_idx(
        device_id, op_idx, future_dims, prev_dims, dim, my_dim_bit
    )
    chiplet_bit = device_id % 2
    return my_base_chunk_idx * 2 + chiplet_bit

  def get_hcube_chunk_idx(
      self, device_id, loop_idx, future_dims, prev_dims, target_dim, dim_val
  ):
    chip_id = device_id // 2
    base = 0
    for d in prev_dims:
      bit = (chip_id >> d) & 1
      base |= bit << d
    for bit_pos, d in enumerate(future_dims):
      bit = (loop_idx >> bit_pos) & 1
      base |= bit << d
    base |= dim_val << target_dim
    return base

  def make_phase1_index_fn(self, mb_idx):
    def phase1_index_fn(chip_idx):
      c_me = self.get_phase1_chunk_idx(self.topo.cur_id, chip_idx)
      return (c_me, mb_idx)

    return phase1_index_fn

  def make_phase1_in_index_fn_with_recv_sem(self, mb_idx):
    def phase1_in_index_fn_with_recv_sem(grid_indices, ref):
      (chip_idx,) = grid_indices
      mb_start = mb_idx * self.dims.mb_size
      c_me = self.get_phase1_chunk_idx(self.topo.cur_id, chip_idx)
      chunk_slice = self.get_slice(
          chunk_idx=c_me, start=mb_start, size=self.dims.mb_size
      )
      return chunk_slice, self.dims.mb_size

    return phase1_in_index_fn_with_recv_sem

  def make_phase2_in_index_fn_with_recv_sem(self, step_idx, mb_idx):
    def phase2_in_index_fn_with_recv_sem(grid_indices, ref):
      op_idx, hcube_dim_idx = grid_indices
      my_chunk_idx = self.get_phase2_chunk_idx(
          self.topo.cur_id, step_idx, op_idx, hcube_dim_idx
      )

      mb_start = mb_idx * self.mb_stride
      mb_start_idx = mb_start + hcube_dim_idx * self.dims.hc_chunk_size

      chunk_slice = self.get_slice(
          chunk_idx=my_chunk_idx,
          start=mb_start_idx,
          size=self.dims.hc_chunk_size,
      )
      return chunk_slice, self.dims.hc_chunk_size

    return phase2_in_index_fn_with_recv_sem

  def make_phase2_index_fn(self, step_idx, mb_idx):
    def phase2_index_fn(op_idx, hcube_dim_idx):
      my_chunk_idx = self.get_phase2_chunk_idx(
          self.topo.cur_id, step_idx, op_idx, hcube_dim_idx
      )

      mb_start_idx = mb_idx * self.dims.num_hcube_dims + hcube_dim_idx
      return (my_chunk_idx, mb_start_idx)

    return phase2_index_fn

  def make_phase2_out_index_fn(self, step_idx, mb_idx):
    def phase2_out_index_fn(op_idx, hcube_dim_idx):
      mb_start_idx = mb_idx * self.dims.num_hcube_dims + hcube_dim_idx
      return (0, mb_start_idx)

    return phase2_out_index_fn


# ==============================================================================
# LAYER 3: DMA EXECUTION ENGINE
# ==============================================================================


class DmaManager:
  """Handles Pallas pipeline emission and explicit async DMA dispatching."""

  def __init__(
      self,
      dims: RsDimensions,
      topo: Topology,
      locator: ChunkLocator,
      recv_bref,
      run_bref,
      out_bref,
      phase1_send_sems,
      phase1_recv_sems,
      phase2_send_sems,
      phase2_recv_sems,
  ):
    self.dims = dims
    self.topo = topo
    self.locator = locator
    self.recv_bref = recv_bref
    self.run_bref = run_bref
    self.out_bref = out_bref
    self.phase1_send_sems = phase1_send_sems
    self.phase1_recv_sems = phase1_recv_sems
    self.phase2_send_sems = phase2_send_sems
    self.phase2_recv_sems = phase2_recv_sems

  def start_phase1_d2d_copies(self, src, dst, mb_idx):
    ops = []
    mb_start = mb_idx * self.dims.mb_size
    mb_start, mb_slice_size = _get_capped_bounds(
        mb_start, self.dims.mb_size, self.dims.hidden_size_dim
    )
    partner_chunks = self.locator.get_phase1_chunk_idxes(self.topo.partner_id)
    for chip_idx, c_neigh in enumerate(partner_chunks):
      mb_slice = self.locator.get_slice(
          chunk_idx=c_neigh, start=mb_start, size=mb_slice_size
      )
      op = pltpu.make_async_remote_copy(
          src_ref=src.at[mb_slice],
          dst_ref=dst.at[mb_slice],
          send_sem=self.phase1_send_sems.at[chip_idx, mb_idx],
          recv_sem=self.phase1_recv_sems.at[chip_idx, mb_idx],
          device_id=self.topo.partner_id,
          device_id_type=pl.DeviceIdType.LOGICAL,
      )
      op.start()
      ops.append(op)
    return ops

  def start_phase2_c2c_copies(self, src, dst, mb_idx, step_idx):
    mb_ops = []
    exponent = self.dims.num_hcube_dims - 1 - step_idx
    num_ops_in_step = 1 << exponent if exponent >= 0 else 0

    for op_idx in range(num_ops_in_step):
      for hcube_dim_idx in range(self.dims.num_hcube_dims):
        dim = (hcube_dim_idx + step_idx) % self.dims.num_hcube_dims

        mb_start = mb_idx * self.locator.mb_stride
        chunk_start = mb_start + hcube_dim_idx * self.dims.hc_chunk_size
        chunk_start, k_size = _get_capped_bounds(
            chunk_start, self.dims.hc_chunk_size, self.dims.hidden_size_dim
        )

        neigh_device_id = self.topo.get_phase2_neighbor_device_id(dim)
        my_chunk_idx = self.locator.get_phase2_chunk_idx(
            self.topo.cur_id, step_idx, op_idx, hcube_dim_idx
        )
        neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
            neigh_device_id, step_idx, op_idx, hcube_dim_idx
        )

        if k_size > 0:
          mb_slice = self.locator.get_slice(
              neighbor_chunk_idx, chunk_start, k_size
          )
          op = pltpu.make_async_remote_copy(
              src_ref=src.at[mb_slice],
              dst_ref=dst.at[mb_slice],
              send_sem=self.phase2_send_sems.at[
                  step_idx, mb_idx, hcube_dim_idx, op_idx
              ],
              recv_sem=self.phase2_recv_sems.at[
                  step_idx, mb_idx, hcube_dim_idx, op_idx
              ],
              device_id=neigh_device_id,
              device_id_type=pl.DeviceIdType.LOGICAL,
          )
          op.start()
          mb_ops.append((
              op,
              step_idx,
              mb_idx,
              hcube_dim_idx,
              op_idx,
              my_chunk_idx,
              chunk_start,
              k_size,
          ))
    return mb_ops

  def run_phase1_accumulate_pipeline(
      self,
      src1,
      src2,
      dst,
      in_index_fn,
      out_index_fn,
      in_index_fn_with_recv_sem,
      block_size,
      mb_idx,
  ):
    """Orchestrates a pairwise D2D accumulation pipeline on a 1D chip grid."""
    def accum_body(s1_ref, s2_ref, d_ref):
      d_ref[...] = s1_ref[...] + s2_ref[...]

    grid = (self.dims.num_chips,)

    def sync_in_index_fn_with_recv_sem(grid_indices, ref):
      hbm_index, size = in_index_fn_with_recv_sem(grid_indices, ref)
      (chip_idx,) = grid_indices
      sem = self.phase1_recv_sems.at[chip_idx, mb_idx]
      return hbm_index, sem, size

    in_spec = pl.BlockSpec(
        block_shape=(self.dims.seq_chunk_size, block_size),
        index_map=in_index_fn,
    )
    out_spec = pl.BlockSpec(
        block_shape=(self.dims.seq_chunk_size, block_size),
        index_map=out_index_fn,
    )

    s1_bref = RemoteWaitBufferedRef.from_ref(
        self.recv_bref.with_spec(in_spec),
        index_fn_with_recv_sem=sync_in_index_fn_with_recv_sem,
    )
    s2_bref = RemoteWaitBufferedRef.from_ref(self.run_bref.with_spec(in_spec))
    d_bref = RemoteWaitBufferedRef.from_ref(self.out_bref.with_spec(out_spec))

    pltpu.emit_pipeline(
        accum_body,
        grid=grid,
        in_specs=[in_spec, in_spec],
        out_specs=[out_spec],
    )(src1, src2, dst, allocations=[s1_bref, s2_bref, d_bref])

  def run_phase2_accumulate_pipeline(
      self,
      src1,
      src2,
      dst,
      in_index_fn,
      out_index_fn,
      in_index_fn_with_recv_sem,
      block_size,
      mb_idx,
      step_idx,
  ):
    """Orchestrates a ring hypercube C2C accumulation pipeline on a 2D grid."""

    def accum_body(s1_ref, s2_ref, d_ref):
      d_ref[...] = s1_ref[...] + s2_ref[...]

    exponent = self.dims.num_hcube_dims - 1 - step_idx
    num_ops_in_step = 1 << exponent if exponent >= 0 else 0
    grid = (num_ops_in_step, self.dims.num_hcube_dims)

    def sync_in_index_fn_with_recv_sem(grid_indices, ref):
      hbm_index, size = in_index_fn_with_recv_sem(grid_indices, ref)
      op_idx, hcube_dim_idx = grid_indices
      sem = self.phase2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx, op_idx]
      return hbm_index, sem, size

    in_spec = pl.BlockSpec(
        block_shape=(self.dims.seq_chunk_size, block_size),
        index_map=in_index_fn,
    )
    out_spec = pl.BlockSpec(
        block_shape=(self.dims.seq_chunk_size, block_size),
        index_map=out_index_fn,
    )

    s1_bref = RemoteWaitBufferedRef.from_ref(
        self.recv_bref.with_spec(in_spec),
        index_fn_with_recv_sem=sync_in_index_fn_with_recv_sem,
    )
    s2_bref = RemoteWaitBufferedRef.from_ref(self.run_bref.with_spec(in_spec))
    d_bref = RemoteWaitBufferedRef.from_ref(self.out_bref.with_spec(out_spec))

    pltpu.emit_pipeline(
        accum_body,
        grid=grid,
        in_specs=[in_spec, in_spec],
        out_specs=[out_spec],
    )(src1, src2, dst, allocations=[s1_bref, s2_bref, d_bref])


def _next_multiple_of(val: int, multiple: int) -> int:
  return ((val + multiple - 1) // multiple) * multiple


def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    recv_bref,
    run_bref,
    out_bref,
    phase1_send_sems,
    phase1_recv_sems,
    phase2_send_sems,
    phase2_recv_sems,
    *,
    dims: RsDimensions,
    axis_name: str = "x",
):
  topo = Topology(axis_name)
  locator = ChunkLocator(dims, topo)
  dma = DmaManager(
      dims,
      topo,
      locator,
      recv_bref,
      run_bref,
      out_bref,
      phase1_send_sems,
      phase1_recv_sems,
      phase2_send_sems,
      phase2_recv_sems,
  )

  all_phase1_ops = []
  all_phase2_ops = []

  # =========================================================================================================
  #                                  HIERARCHICAL REDUCE-SCATTER TIMELINE (D2D + C2C Step 0)
  # =========================================================================================================
  #
  # Time -------->  t0                  t1                                      t2                                      t3
  #                 | Global Prologue   |              Loop m=0                 |              Loop m=1                 |
  #                 |                   |                                       |                                       |
  # D2D/DMA (P1)    [A]========[B]      |       [D]========[E]                  |       [D]========[E]                  |
  #                 |   P1 MB0          |       |   P1 MB1                      |       |   P1 MB2                      |
  #                 |                   |       |                               |       |                               |
  # C2C (P2)        |                   [C]=====================================[G]                                     |
  #                 |                   |             P2 MB0                    |                                       |
  #                 |                   |                                       [F]=====================================[G]
  #                 |                   |                                       |             P2 MB1                    |
  #                 |                   |                                       |                                       [F]========> (to t4)
  #                 |                   |                                       |                                       |  P2 MB2
  # Accumulate      |          [B]======|                  [E]======|           [G]======|                 [I]======|   [J]======|
  #                 |            AC P1  |                    AC P1  |           |  AC P2 |                   AC P1  |   |  AC P2 |
  #                 |            (MB0)  |                    (MB1)  |           |  (MB0) |                   (MB2)  |   |  (MB1) |
  # =========================================================================================================

  # =========== Global Prologue: PHASE 1 Micro-Batch 0 D2D REDUCTIONS ===========
  # [Step A]: Start remote D2D copies for micro-batch 0
  with jax.named_scope("start_phase1_mb_0"):
    all_phase1_ops.extend(
        dma.start_phase1_d2d_copies(src=input_ref, dst=recv_buf_ref, mb_idx=0)
    )

  # [Step B]: Wait for micro-batch 0 copies to finish, and accumulate locally
  with jax.named_scope("accumulate_phase1_mb_0"):
    dma.run_phase1_accumulate_pipeline(
        src1=recv_buf_ref,
        src2=input_ref,
        dst=running_sum_ref,
        in_index_fn=locator.make_phase1_index_fn(mb_idx=0),
        out_index_fn=locator.make_phase1_index_fn(mb_idx=0),
        in_index_fn_with_recv_sem=locator.make_phase1_in_index_fn_with_recv_sem(
            mb_idx=0
        ),
        block_size=dims.mb_size,
        mb_idx=0,
    )

  # [Step C]: Start Phase 2 Ring ICI copies for micro-batch 0
  mb_ops = dma.start_phase2_c2c_copies(
      src=running_sum_ref, dst=recv_buf_ref, mb_idx=0, step_idx=0
  )
  all_phase2_ops.extend([item[0] for item in mb_ops])

  for m in range(dims.num_micro_batches):
    with jax.named_scope(f"phase2_step0_mb_{m}"):

      # OVERLAP: Start next micro-batch transfers
      if m < dims.num_micro_batches - 1:
        # [Step D]: Start overlap Phase 1 D2D copies for next micro-batch
        with jax.named_scope(f"start_phase1_mb_{m+1}"):
          all_phase1_ops.extend(
              dma.start_phase1_d2d_copies(
                  src=input_ref, dst=recv_buf_ref, mb_idx=m + 1
              )
          )

        # [Step E]: Wait and Accumulate Phase 1 for next micro-batch
        with jax.named_scope(f"accumulate_phase1_mb_{m+1}"):
          dma.run_phase1_accumulate_pipeline(
              src1=recv_buf_ref,
              src2=input_ref,
              dst=running_sum_ref,
              in_index_fn=locator.make_phase1_index_fn(m + 1),
              out_index_fn=locator.make_phase1_index_fn(m + 1),
              in_index_fn_with_recv_sem=locator.make_phase1_in_index_fn_with_recv_sem(
                  m + 1
              ),
              block_size=dims.mb_size,
              mb_idx=m + 1,
          )

        # [Step F]: Pre-start next micro-batch Phase 2 Ring ICI copies
        with jax.named_scope(f"start_phase2_step0_mb_{m+1}"):
          mb_ops = dma.start_phase2_c2c_copies(
              src=running_sum_ref, dst=recv_buf_ref, mb_idx=m + 1, step_idx=0
          )
          all_phase2_ops.extend([item[0] for item in mb_ops])

      # [Phase 2, Step 1] Pre-start Step 1 MB0 during the last iteration of Step 0
      if m == dims.num_micro_batches - 1 and dims.num_micro_batches > 1:
        with jax.named_scope("start_phase2_step1_mb_0"):
          mb_ops = dma.start_phase2_c2c_copies(
              src=running_sum_ref, dst=recv_buf_ref, mb_idx=0, step_idx=1
          )
          all_phase2_ops.extend([item[0] for item in mb_ops])

      # [Step G]: Wait and Accumulate Phase 2 Step 0 for current micro-batch
      if dims.num_hcube_dims >= 1:
        is_last_step = dims.num_hcube_dims == 1

        dma.run_phase2_accumulate_pipeline(
            src1=recv_buf_ref,
            src2=running_sum_ref,
            dst=output_ref if is_last_step else running_sum_ref,
            in_index_fn=locator.make_phase2_index_fn(0, m),
            out_index_fn=locator.make_phase2_out_index_fn(0, m)
            if is_last_step
            else locator.make_phase2_index_fn(0, m),
            in_index_fn_with_recv_sem=locator.make_phase2_in_index_fn_with_recv_sem(
                0, m
            ),
            block_size=dims.hc_chunk_size,
            mb_idx=m,
            step_idx=0,
        )

  # If we only have 1 micro-batch, we couldn't pre-start Step 1 MB0 in the loop due to race.
  # Start it now, after Step 0 MB0 accumulation is fully complete.
  if dims.num_micro_batches == 1:
    with jax.named_scope("start_phase2_step1_mb_0"):
      mb_ops = dma.start_phase2_c2c_copies(
          src=running_sum_ref, dst=recv_buf_ref, mb_idx=0, step_idx=1
      )
      all_phase2_ops.extend([item[0] for item in mb_ops])

  # ================= STEP 1 LOOP =================
  for m in range(dims.num_micro_batches):
    with jax.named_scope(f"phase2_step1_mb_{m}"):

      # Start next micro-batch transfers
      if m < dims.num_micro_batches - 1:
        with jax.named_scope(f"start_phase2_step1_mb_{m+1}"):
          mb_ops = dma.start_phase2_c2c_copies(
              src=running_sum_ref, dst=recv_buf_ref, mb_idx=m + 1, step_idx=1
          )
          all_phase2_ops.extend([item[0] for item in mb_ops])

      # Accumulate Step 1
      if dims.num_hcube_dims > 1:
        dma.run_phase2_accumulate_pipeline(
            src1=recv_buf_ref,
            src2=running_sum_ref,
            dst=output_ref,
            in_index_fn=locator.make_phase2_index_fn(1, m),
            out_index_fn=locator.make_phase2_out_index_fn(1, m),
            in_index_fn_with_recv_sem=locator.make_phase2_in_index_fn_with_recv_sem(
                1, m
            ),
            block_size=dims.hc_chunk_size,
            mb_idx=m,
            step_idx=1,
        )

  for op in all_phase1_ops:
    op.wait_send()
  for op in all_phase2_ops:
    op.wait_send()


def _make_unified_scratch_shapes(
    seq_chunk_size: int,
    mb_size: int,
    dtype: Any,
    num_chips: int,
    num_hcube_dims: int,
    num_micro_batches: int,
) -> list[Any]:
  block_spec = pl.BlockSpec(
      block_shape=(seq_chunk_size, mb_size), index_map=lambda *args: (0, 0)
  )
  recv_bref = pltpu.BufferedRef.input(block_spec, dtype, buffer_count=2)
  run_bref = pltpu.BufferedRef.input(block_spec, dtype, buffer_count=2)
  out_bref = pltpu.BufferedRef.output(block_spec, dtype, buffer_count=2)

  scratch_shapes = [
      recv_bref,
      run_bref,
      out_bref,
      pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),
      pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),
  ]
  p2_sem_shape = (
      num_hcube_dims,
      num_micro_batches,
      num_hcube_dims,
      2 ** (num_hcube_dims - 1),
  )
  scratch_shapes.extend([
      pltpu.SemaphoreType.DMA(p2_sem_shape),
      pltpu.SemaphoreType.DMA(p2_sem_shape),
  ])
  return scratch_shapes


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int | None = None,
    axis_name: str | tuple[str, ...] = "x",
) -> jax.Array:
  num_chips = num_devices // 2
  num_hcube_dims = int(math.log2(num_chips))
  local_seq_len, hidden_size_dim = local_x.shape

  seq_chunk_size_orig = local_seq_len // num_devices
  needs_padding = seq_chunk_size_orig < 8

  if needs_padding:
    # Pad local_x to be 8 at least
    reshaped_x = local_x.reshape(num_devices, -1, hidden_size_dim)
    padded_x = jnp.pad(
        reshaped_x, ((0, 0), (0, 8 - seq_chunk_size_orig), (0, 0))
    )
    local_x = padded_x.reshape(-1, hidden_size_dim)
    local_seq_len = local_x.shape[0]

  if num_micro_batches is None:
    if local_seq_len >= 8192:
      num_micro_batches = 8
    elif local_seq_len >= 2048:
      num_micro_batches = 4
    elif local_seq_len >= 512:
      num_micro_batches = 2
    else:
      num_micro_batches = 1

  vector_width = pltpu.get_tpu_info().num_lanes
  mb_size = _next_multiple_of(
      hidden_size_dim // num_micro_batches, vector_width
  )
  assert (num_micro_batches - 1) * mb_size < hidden_size_dim, (
      f"Unsupported micro-batches config: num_micro_batches={num_micro_batches}"
      f" is too large for hidden_size_dim={hidden_size_dim} with"
      f" mb_size={mb_size} (due to padding)."
  )
  hc_chunk_size = _next_multiple_of(mb_size // num_hcube_dims, vector_width)
  seq_chunk_size = local_seq_len // num_devices

  out_shape = jax.ShapeDtypeStruct(
      (seq_chunk_size, hidden_size_dim), local_x.dtype
  )
  running_sum_shape = jax.ShapeDtypeStruct(
      (local_seq_len, hidden_size_dim), local_x.dtype
  )
  recv_buf_shape = jax.ShapeDtypeStruct(
      (local_seq_len, hidden_size_dim), local_x.dtype
  )

  scratch_shapes = _make_unified_scratch_shapes(
      seq_chunk_size,
      mb_size,
      local_x.dtype,
      num_chips,
      num_hcube_dims,
      num_micro_batches,
  )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
      out_specs=(
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
      ),
      scratch_shapes=tuple(scratch_shapes),
      grid=(1,),
  )

  dims = RsDimensions(
      num_chips=num_chips,
      num_hcube_dims=num_hcube_dims,
      num_micro_batches=num_micro_batches,
      hidden_size_dim=hidden_size_dim,
      seq_chunk_size=seq_chunk_size,
      hc_chunk_size=hc_chunk_size,
      mb_size=mb_size,
  )

  hier_rs = pl.pallas_call(
      jax.tree_util.Partial(hier_rs_kernel, dims=dims, axis_name=axis_name),
      out_shape=(out_shape, running_sum_shape, recv_buf_shape),
      grid_spec=grid_spec,
      name=f"hier_rs_kernel.mb{num_micro_batches}",
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.95),
          disable_bounds_checks=True,
      ),
  )
  out = hier_rs(local_x)[0]
  if needs_padding:
    out = out[:seq_chunk_size_orig, :]
  return out


def hierarchical_reduce_scatter(
    x: jax.Array, mesh: jax.sharding.Mesh, num_micro_batches: int | None = None
) -> jax.Array:
  return shard_map.shard_map(
      lambda local_x: hierarchical_reduce_scatter_local(
          local_x,
          num_devices=mesh.devices.size,
          num_micro_batches=num_micro_batches,
      ),
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec("x", None),
      out_specs=jax.sharding.PartitionSpec("x", None),
      check_rep=False,
  )(x)

