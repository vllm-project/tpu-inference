"""TPU SparseCore Reduce-Scatter Pallas Kernel.

Abbreviations used in this file:
- scs: Scalar Subcore
- tec: Vector Subcore
- mb: Micro-Batch
- hc: Hypercube
- d2d: Die-to-Die (Phase 1, intra-chip ICI link)
- c2c: Chip-to-Chip (Phase 2, inter-chip ICI link)
"""

import dataclasses
import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RsDimensions:
  """Dimensions and sharding sizes for TPU SparseCore Reduce-Scatter.

  Contains the configuration of the network mesh, pipeline micro-batches, and
  low-level SparseCore core/subcore/column partitioning parameters.
  """

  # Number of physical TPU chips in the mesh (num_devices // 2)
  num_chips: int
  # ICI hypercube logical network dimensions (log2(num_chips))
  num_hcube_dims: int
  # Pipelining unrolling factor for overlapping ALU/DMA
  num_micro_batches: int
  # Total hidden size dimension (e.g., 4096)
  hidden_dim_size: int
  # Local sequence slice per device (= seq_len // num_devices)
  chunk_size: int
  # Sequence slice size assigned to each physical core on a device
  core_chunk_size: int
  # Sequence slice size assigned to each subcore row
  subcore_chunk_size: int
  # Micro batch slice size (= hidden_dim_size // num_micro_batches)
  mb_size: int
  # Phase 2 (C2C) hypercube chunk slice size (= mb_size // num_hcube_dims)
  hc_chunk_size: int
  # Number of subcore rows used for row-wise sequence partitioning
  num_subcores_row: int
  # Number of subcore columns used for column-wise hidden size partitioning
  num_subcores_col: int
  # Column slice size for Phase 1 DMA (= mb_size // num_subcores_col)
  subcore_col_chunk_size_p1: int
  # Column slice size for Phase 2 DMA (= hc_chunk_size // num_subcores_col)
  subcore_col_chunk_size_p2: int

  @property
  def num_subcores(self) -> int:
    return self.num_subcores_row * self.num_subcores_col


class Topology:
  """Abstracts JAX axis index calls and partner logic."""

  def __init__(self, axis_name: str):
    self.cur_id = jax.lax.axis_index(axis_name)
    self.cur_chip_id = self.cur_id // 2
    self.cur_chiplet_bit = self.cur_id % 2
    self.partner_id = jax.lax.select(
        self.cur_chiplet_bit == 0, self.cur_id + 1, self.cur_id - 1
    )

  def get_device_id(self, chip_id, chiplet_bit):
    """Returns the global device ID from physical chip `chip_id` and chiplet coordinate `chiplet_bit` (0 or 1)."""
    return chip_id * 2 + chiplet_bit

  def get_neighbor_chip_id(self, dim):
    """Returns the physical chip ID of the logical neighbor in hypercube dimension `dim`.

    For example, on a 2D hypercube of 4 physical chips (IDs: 0, 1, 2, 3):
    - If current chip is 0 (binary 00):
      - Neighbor along dimension 0 is: 0 ^ (1 << 0) = 1 (binary 01).
      - Neighbor along dimension 1 is: 0 ^ (1 << 1) = 2 (binary 10).
    """
    return self.cur_chip_id ^ (1 << dim)

  def get_neighbor_device_id(self, dim):
    """Returns the ID of the neighbor device along hypercube dimension `dim` sharing the same chiplet position.

    For example, on a 2D hypercube of 4 chips (IDs 0-3) containing 8 logical
    devices (IDs 0-7):
    - If current device is 0 (physical chip 0, chiplet bit 0):
      - Neighbor along dimension 0 is: get_device_id(neighbor_chip=1, chiplet=0)
      = 2.
      - Neighbor along dimension 1 is: get_device_id(neighbor_chip=2, chiplet=0)
      = 4.
    """
    return self.get_device_id(
        self.get_neighbor_chip_id(dim), self.cur_chiplet_bit
    )


class ChunkLocator:
  """Encapsulates sequence and HBM indexing math for TPU SparseCore Reduce-Scatter."""

  def __init__(
      self, dims: RsDimensions, topo: Topology, core_idx=0, subcore_idx=0
  ):
    self.dims = dims
    self.topo = topo
    self.core_idx = core_idx
    self.subcore_row_idx = subcore_idx // dims.num_subcores_col
    self.subcore_col_idx = subcore_idx % dims.num_subcores_col
    self.mb_stride = dims.num_hcube_dims * dims.hc_chunk_size

  def _get_row_slice(self, chunk_idx, for_tec):
    """Returns a row slice for `chunk_idx` of core-level size,

    or subcore-level if `for_tec` is True.
    """
    row_offset = (
        chunk_idx * self.dims.chunk_size
        + self.core_idx * self.dims.core_chunk_size
    )
    if for_tec:
      row_offset += self.subcore_row_idx * self.dims.subcore_chunk_size
      row_size = self.dims.subcore_chunk_size
    else:
      row_size = self.dims.core_chunk_size
    return pl.ds(pl.multiple_of(row_offset, 8), row_size)

  def _get_col_slice(self, base_col_offset, col_size, col_chunk_size, for_tec):
    """Returns a column slice from `base_col_offset` of width `col_size`,

    or sharded to `col_chunk_size` if `for_tec` is True.
    """
    if for_tec:
      col_offset = base_col_offset + self.subcore_col_idx * col_chunk_size
      col_width = col_chunk_size
    else:
      col_offset = base_col_offset
      col_width = col_size
    return pl.ds(col_offset, col_width)

  def get_phase1_slice(self, chunk_idx, mb_idx, *, for_tec=False):
    """Returns a 2D HBM slice for Phase 1 (D2D) for `chunk_idx` and `mb_idx`,

    mapped to subcore if `for_tec` is True.
    """
    return (
        self._get_row_slice(chunk_idx, for_tec),
        self._get_col_slice(
            mb_idx * self.dims.mb_size,
            self.dims.mb_size,
            self.dims.subcore_col_chunk_size_p1,
            for_tec,
        ),
    )

  def get_phase2_slice(
      self, chunk_idx, mb_idx, hcube_dim_idx, *, for_tec=False
  ):
    """Returns a 2D HBM slice for Phase 2 (C2C) for `chunk_idx`, `mb_idx`

    and `hcube_dim_idx`, mapped to subcore if `for_tec` is True.
    """
    return (
        self._get_row_slice(chunk_idx, for_tec),
        self._get_col_slice(
            mb_idx * self.dims.mb_size
            + hcube_dim_idx * self.dims.hc_chunk_size,
            self.dims.hc_chunk_size,
            self.dims.subcore_col_chunk_size_p2,
            for_tec,
        ),
    )

  def get_phase1_chunk_idx(self, device_id, chip_idx):
    """Calculates the chunk index processed by `device_id` for `chip_idx`.

    In Phase 1, global token chunks are sharded across the topology. A device
    processes token chunks corresponding to all physical chips `chip_idx` in
    the mesh, filtered by its own chiplet position (even/odd device ID).
    """
    chiplet_bit = device_id % 2
    return chip_idx * 2 + chiplet_bit

  def get_phase1_chunk_idxes(self, device_id):
    """Returns all global chunk indices processed by the chiplet group of device `device_id`."""
    chiplet_bit = device_id % 2
    return [
        chip_idx * 2 + chiplet_bit for chip_idx in range(self.dims.num_chips)
    ]

  def get_phase2_chunk_idx(
      self, device_id, step_idx, chunk_group_idx, hcube_dim_idx
  ):
    """Calculates the chunk index owned by a device `device_id` for chunk group `chunk_group_idx` during Phase 2 (C2C RS).

    During Phase 2, devices perform a hypercube reduction. At step `step_idx` of
    the hypercube reduction, the topology is partitioned into independent
    parallel sub-cubes/groups of devices exchanging along hypercube dimension
    `hcube_dim_idx`.
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
        device_id, chunk_group_idx, future_dims, prev_dims, dim, my_dim_bit
    )
    chiplet_bit = device_id % 2
    return my_base_chunk_idx * 2 + chiplet_bit

  def get_hcube_chunk_idx(
      self,
      device_id,
      chunk_group_idx,
      future_dims,
      prev_dims,
      target_dim,
      dim_val,
  ):
    """Calculates the mapped HBM chunk index for the hypercube communication ring of device `device_id` at iteration `chunk_group_idx` along active dimension `target_dim` with bit value `dim_val`, given the processed dimensions `prev_dims` and unprocessed dimensions `future_dims`."""
    chip_id = device_id // 2
    base = 0
    for d in prev_dims:
      bit = (chip_id >> d) & 1
      base |= bit << d
    for bit_pos, d in enumerate(future_dims):
      bit = (chunk_group_idx >> bit_pos) & 1
      base |= bit << d
    base |= dim_val << target_dim
    return base


def _get_packing_factor(dtype) -> int:
  """Returns the number of array elements packed into a single 32-bit (4-byte) word."""
  return 4 // dtype.itemsize


def _accumulate(
    *,
    target,
    addend,
    num_rows,
    num_cols,
    col_step=16,
):
  """Performs in-place element-wise addition: `target += addend`.

  Both `target` and `addend` are 2D references of shape (`num_rows`,
  `num_cols`).
  """
  packing = _get_packing_factor(target.dtype)

  @plsc.parallel_loop(0, num_cols, step=col_step)
  def _loop(c_in):
    c_slice = pl.ds(c_in, col_step)

    num_iters = num_rows // packing
    for i in range(num_iters):
      r = i * packing
      r_idx = r if packing == 1 else pl.ds(r, packing)
      val = target[r_idx, c_slice] + addend[r_idx, c_slice]
      target[r_idx, c_slice] = val


class RemoteDmaManager:
  """Handles remote inter-device (ICI) DMA copies on SCS."""

  def __init__(
      self,
      dims: RsDimensions,
      topo: Topology,
      core_idx,
      subcore_idx=0,
      *,
      p1_send_sem,
      p2_send_sem,
      p1_recv_sem,
      p2_recv_sem,
  ):
    self.dims = dims
    self.topo = topo
    self.locator = ChunkLocator(dims, topo, core_idx, subcore_idx)
    self.p1_send_sem = p1_send_sem
    self.p2_send_sem = p2_send_sem
    self.p1_recv_sem = p1_recv_sem
    self.p2_recv_sem = p2_recv_sem

  @jax.named_scope("start_phase1_d2d_copies")
  def start_phase1_d2d_copies(
      self,
      *,
      mb_idx,
      src,
      dst,
  ):
    """Triggers remote D2D copies from `src` to `dst` for the micro-batch index `mb_idx`."""
    recv_slot = mb_idx % 2
    for c, send_chunk_idx in enumerate(
        self.locator.get_phase1_chunk_idxes(self.topo.partner_id)
    ):
      row_slice, col_slice = self.locator.get_phase1_slice(
          send_chunk_idx, mb_idx
      )
      slice_src = src.at[row_slice, col_slice]
      slice_dst = dst.at[row_slice, col_slice]
      pltpu.async_remote_copy(
          slice_src,
          slice_dst,
          self.p1_send_sem.at[recv_slot, c],
          self.p1_recv_sem.at[recv_slot, c],
          device_id=self.topo.partner_id,
          device_id_type=pl.DeviceIdType.LOGICAL,
      )

  @jax.named_scope("start_phase2_c2c_copies")
  def start_phase2_c2c_copies(
      self,
      *,
      mb_idx,
      step_idx,
      src,
      dst,
  ):
    """Triggers remote C2C copies from `src` to `dst` for `mb_idx` and `step_idx`."""
    slot_idx = mb_idx % 2
    num_hcube_dims = self.dims.num_hcube_dims
    num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)

    @pl.loop(0, num_chunk_groups)
    def _(chunk_group_idx):
      @pl.loop(0, num_hcube_dims)
      def _(hcube_dim_idx):
        dim = (hcube_dim_idx + step_idx) % num_hcube_dims
        neighbor_device_id = self.topo.get_neighbor_device_id(dim)
        neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
            neighbor_device_id,
            step_idx,
            chunk_group_idx=chunk_group_idx,
            hcube_dim_idx=hcube_dim_idx,
        )

        row_slice, col_slice = self.locator.get_phase2_slice(
            neighbor_chunk_idx, mb_idx, hcube_dim_idx
        )
        slice_src = src.at[step_idx, row_slice, col_slice]
        slice_dst = dst.at[row_slice, col_slice]
        pltpu.async_remote_copy(
            slice_src,
            slice_dst,
            self.p2_send_sem.at[
                slot_idx, step_idx, chunk_group_idx, hcube_dim_idx
            ],
            self.p2_recv_sem.at[
                slot_idx, step_idx, chunk_group_idx, hcube_dim_idx
            ],
            device_id=neighbor_device_id,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

  @jax.named_scope("wait_phase1_d2d_copies")
  def wait_phase1_d2d_copies(
      self, mb_idx, src, dst, *, wait_send: bool = False
  ):
    """Waits for Phase 1 (D2D) remote copies from `src` to `dst` for the index `mb_idx`.

    If `wait_send` is True, it blocks until the local send completes, releasing
    the source buffer slice; otherwise, it blocks until the remote receive
    completes.
    """
    recv_slot = mb_idx % 2
    p1_recv_sem_slice = self.p1_recv_sem.at[recv_slot]
    p1_send_sem_slice = self.p1_send_sem.at[recv_slot]

    dummy_send = src.at[
        pl.ds(0, self.dims.core_chunk_size),
        pl.ds(0, self.dims.mb_size),
    ]
    dummy_recv = dst.at[
        pl.ds(0, self.dims.core_chunk_size),
        pl.ds(0, self.dims.mb_size),
    ]
    my_id = self.topo.cur_id
    for c, _ in enumerate(self.locator.get_phase1_chunk_idxes(my_id)):
      dma = pltpu.make_async_remote_copy(
          src_ref=dummy_send,
          dst_ref=dummy_recv,
          send_sem=p1_send_sem_slice.at[c],
          recv_sem=p1_recv_sem_slice.at[c],
          device_id=self.topo.partner_id,
          device_id_type=pl.DeviceIdType.LOGICAL,
      )
      if wait_send:
        dma.wait_send()
      else:
        dma.wait_recv()

  @jax.named_scope("wait_phase2_c2c_copies")
  def wait_phase2_c2c_copies(
      self, mb_idx, step_idx, src, dst, *, wait_send: bool = False
  ):
    """Waits for Phase 2 (C2C) remote copies from `src` to `dst` for `mb_idx` and `step_idx`.

    If `wait_send` is True, it blocks until the local send completes, releasing
    the source buffer slice; otherwise, it blocks until the remote receive
    completes.
    """
    slot_idx = mb_idx % 2
    num_hcube_dims = self.dims.num_hcube_dims
    num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)
    p2_recv_sem_slice = self.p2_recv_sem.at[slot_idx, step_idx]
    p2_send_sem_slice = self.p2_send_sem.at[slot_idx, step_idx]

    dummy_send = src.at[
        step_idx,
        pl.ds(0, self.dims.core_chunk_size),
        pl.ds(0, self.dims.hc_chunk_size),
    ]
    dummy_recv = dst.at[
        pl.ds(0, self.dims.core_chunk_size),
        pl.ds(0, self.dims.hc_chunk_size),
    ]
    @pl.loop(0, num_chunk_groups)
    def _(chunk_group_idx):
      @pl.loop(0, num_hcube_dims)
      def _(hcube_dim_idx):
        dim = (hcube_dim_idx + step_idx) % num_hcube_dims
        neighbor_device_id = self.topo.get_neighbor_device_id(dim)

        dma = pltpu.make_async_remote_copy(
            src_ref=dummy_send,
            dst_ref=dummy_recv,
            send_sem=p2_send_sem_slice.at[chunk_group_idx, hcube_dim_idx],
            recv_sem=p2_recv_sem_slice.at[chunk_group_idx, hcube_dim_idx],
            device_id=neighbor_device_id,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        if wait_send:
          dma.wait_send()
        else:
          dma.wait_recv()


class LocalDmaManager:
  """Handles local (HBM <-> VMEM) DMA copies and pipelines on TEC."""

  def __init__(
      self,
      dims: RsDimensions,
      topo: Topology,
      core_idx,
      subcore_idx,
  ):
    self.dims = dims
    self.topo = topo
    self.locator = ChunkLocator(dims, topo, core_idx, subcore_idx)

  def _local_subcore_copy(self, src_ref, dst_ref, sem):
    """Issues asynchronous DMAs to copy a subcore chunk between `src_ref` and `dst_ref` using `sem`."""
    if self.dims.subcore_chunk_size % 8 == 0:
      pltpu.make_async_copy(src_ref, dst_ref, sem).start()
    else:
      # If the number of rows is not a multiple of 8, it iterates over
      # rows to satisfy layout alignment requirements.
      src_32b = src_ref.bitcast(jax.numpy.uint32)
      dst_32b = dst_ref.bitcast(jax.numpy.uint32)
      for i in range(src_32b.shape[0]):
        pltpu.make_async_copy(src_32b.at[i, :], dst_32b.at[i, :], sem).start()

  def _wait_subcore_copies(self, src_dummy, dst_dummy, sem, num_copies=1):
    """Waits for `num_copies` subcore chunks to complete on `sem` using dummy async copies."""
    for _ in range(num_copies):
      if self.dims.subcore_chunk_size % 8 == 0:
        pltpu.make_async_copy(src_dummy, dst_dummy, sem).wait()
      else:
        src_32b = src_dummy.bitcast(jax.numpy.uint32)
        dst_32b = dst_dummy.bitcast(jax.numpy.uint32)
        for i in range(src_32b.shape[0]):
          pltpu.make_async_copy(src_32b.at[i, :], dst_32b.at[i, :], sem).wait()

  # TODO: Use emit_pipeline once correctness bug is fixed.
  def _accumulate_pipeline(
      self,
      num_iters: int,
      get_src1_fn,
      get_src2_fn,
      get_out_fn,
      dtype,
      max_col_size,
  ):
    """Double-buffered accumulation of chunks across `num_iters` iterations."""
    if num_iters == 0:
      return

    @functools.partial(
        pl.run_scoped,
        src1_vmem_ref=pltpu.VMEM(
            (2, max(2, self.dims.subcore_chunk_size), max_col_size), dtype
        ),
        src2_vmem_ref=pltpu.VMEM(
            (2, max(2, self.dims.subcore_chunk_size), max_col_size), dtype
        ),
        src_sems=pltpu.SemaphoreType.DMA((num_iters,)),
        out_sems=pltpu.SemaphoreType.DMA((num_iters,)),
    )
    def _run(src1_vmem_ref, src2_vmem_ref, src_sems, out_sems):
      def copy_in_fn(idx, slot):
        src1 = get_src1_fn(idx)
        src2 = get_src2_fn(idx)
        r, c = src1.shape
        self._local_subcore_copy(
            src1, src1_vmem_ref.at[slot, :r, :c], src_sems.at[idx]
        )
        self._local_subcore_copy(
            src2, src2_vmem_ref.at[slot, :r, :c], src_sems.at[idx]
        )

      def wait_in_fn(idx, slot):
        src1 = get_src1_fn(idx)
        r, c = src1.shape
        with jax.named_scope("wait"):
          self._wait_subcore_copies(
              src1,
              src1_vmem_ref.at[0, :r, :c],
              src_sems.at[idx],
              num_copies=2,
          )

      def compute_fn(idx, slot):
        src1 = get_src1_fn(idx)
        r, c = src1.shape
        with jax.named_scope("accumulate"):
          _accumulate(
              target=src1_vmem_ref.at[slot],
              addend=src2_vmem_ref.at[slot],
              num_rows=r,
              num_cols=c,
          )

      def copy_out_fn(idx, slot):
        out = get_out_fn(idx)
        r, c = out.shape
        self._local_subcore_copy(
            src1_vmem_ref.at[slot, :r, :c], out, out_sems.at[idx]
        )

      def wait_out_fn(idx, slot):
        out = get_out_fn(idx)
        r, c = out.shape
        self._wait_subcore_copies(
            src1_vmem_ref.at[0, :r, :c],
            out,
            out_sems.at[idx],
            num_copies=1,
        )

      def _slot(idx):
        return idx % 2

      copy_in_fn(0, 0)
      @pl.loop(0, num_iters - 1)
      def _(idx):
        @pl.when(idx >= 1)
        def _():
          wait_out_fn(idx - 1, _slot(idx - 1))

        copy_in_fn(idx + 1, _slot(idx + 1))
        wait_in_fn(idx, _slot(idx))
        compute_fn(idx, _slot(idx))
        copy_out_fn(idx, _slot(idx))

      last_idx = num_iters - 1
      wait_in_fn(last_idx, _slot(last_idx))
      compute_fn(last_idx, _slot(last_idx))
      copy_out_fn(last_idx, _slot(last_idx))

      if last_idx >= 1:
        wait_out_fn(last_idx - 1, _slot(last_idx - 1))
      wait_out_fn(last_idx, _slot(last_idx))

  def run_phase1_accumulate_pipeline(
      self,
      *,
      mb_idx,
      src1_ref,
      src2_ref,
      out_ref,
  ):
    """Executes Phase 1 pipelined accumulation for `mb_idx`."""
    num_iters = self.dims.num_chips

    def get_src1_fn(idx):
      chunk_idx = self.locator.get_phase1_chunk_idx(self.topo.cur_id, idx)
      rs, cs = self.locator.get_phase1_slice(chunk_idx, mb_idx, for_tec=True)
      return src1_ref.at[rs, cs]

    def get_src2_fn(idx):
      chunk_idx = self.locator.get_phase1_chunk_idx(self.topo.cur_id, idx)
      rs, cs = self.locator.get_phase1_slice(chunk_idx, mb_idx, for_tec=True)
      return src2_ref.at[rs, cs]

    def get_out_fn(idx):
      chunk_idx = self.locator.get_phase1_chunk_idx(self.topo.cur_id, idx)
      rs, cs = self.locator.get_phase1_slice(chunk_idx, mb_idx, for_tec=True)
      return out_ref.at[0].at[rs, cs]

    self._accumulate_pipeline(
        num_iters,
        get_src1_fn,
        get_src2_fn,
        get_out_fn,
        src1_ref.dtype,
        self.dims.subcore_col_chunk_size_p1,
    )

  def run_phase2_accumulate_pipeline(
      self,
      *,
      mb_idx,
      step_idx,
      src1_ref,
      src2_ref,
      final_out_ref,
  ):
    """Executes Phase 2 pipelined reduction for `mb_idx` and `step_idx`."""
    num_hcube_dims = self.dims.num_hcube_dims
    num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)
    num_iters = num_chunk_groups * num_hcube_dims
    is_last_step = step_idx == num_hcube_dims - 1

    def get_chunk_idx(idx):
      chunk_group_idx = idx // num_hcube_dims
      hcube_dim_idx = idx % num_hcube_dims
      return self.locator.get_phase2_chunk_idx(
          self.topo.cur_id,
          step_idx,
          chunk_group_idx=chunk_group_idx,
          hcube_dim_idx=hcube_dim_idx,
      )

    def get_src1_fn(idx):
      chunk_idx = get_chunk_idx(idx)
      hcube_dim_idx = idx % num_hcube_dims
      rs, cs = self.locator.get_phase2_slice(
          chunk_idx, mb_idx, hcube_dim_idx, for_tec=True
      )
      return src1_ref.at[step_idx, rs, cs]

    def get_src2_fn(idx):
      chunk_idx = get_chunk_idx(idx)
      hcube_dim_idx = idx % num_hcube_dims
      rs, cs = self.locator.get_phase2_slice(
          chunk_idx, mb_idx, hcube_dim_idx, for_tec=True
      )
      return src2_ref.at[step_idx + 1, rs, cs]

    def get_out_fn(idx):
      chunk_idx = 0 if is_last_step else get_chunk_idx(idx)
      hcube_dim_idx = idx % num_hcube_dims
      rs, cs = self.locator.get_phase2_slice(
          chunk_idx, mb_idx, hcube_dim_idx, for_tec=True
      )
      final_dst = final_out_ref if is_last_step else src1_ref.at[step_idx + 1]
      return final_dst.at[rs, cs]

    self._accumulate_pipeline(
        num_iters,
        get_src1_fn,
        get_src2_fn,
        get_out_fn,
        src1_ref.dtype,
        self.dims.subcore_col_chunk_size_p2,
    )


# ==============================================================================
#                 HIERARCHICAL REDUCE-SCATTER TIMELINE (D2D + C2C Step 0)
# ==============================================================================
# Time -> t0            t1                      t2                      t3
#         | Prologue    |  Loop m=0             |  Loop m=1             |
#         |             |                       |                       |
# D2D/DMA [A]====[B]    |  [D]====[E]           |  [D]====[E]           |
# (P1)    | P1 MB0      |  | P1 MB1             |  | P1 MB2             |
#         |             |  |                    |  |                    |
# C2C     |             [C]=====================[G]                     |
# (P2)    |             |       P2 MB0          |                       |
#         |             |                       [F]=====================[G]
#         |             |                       |       P2 MB1          |
#         |             |                       |                       [F]====>
#         |             |                       |                       | P2 MB2
# Accum   |      [B]====|          [E]====|     [G]====|         [I]====|  [J]===|
#         |        AC P1|            AC P1|     | AC P2|           AC P1|        | AC P2
#         |        (MB0)|            (MB1)|     | (MB0)|           (MB2)|        | (MB1)
# ==============================================================================


def scs_kernel(
    x_ref,
    _,
    running_sum_ref,
    recv_buf_ref,
    *,
    dims: RsDimensions,
    axis_name: str | tuple[str, ...],
    scs_to_tec,
    tec_to_scs,
    p1_recv_sem,
    p2_recv_sem,
    p1_send_sem,
    p2_send_sem,
    **unused_scratch,
):
  """Executes SparseCore Sequencer (SCS) execution logic for Reduce-Scatter.

  SCS is in charge of handling D2D and C2C ICI operations.
  """
  core_idx = jax.lax.axis_index("core")
  topo = Topology(axis_name)
  dma_manager = RemoteDmaManager(
      dims,
      topo,
      core_idx,
      p1_send_sem=p1_send_sem,
      p2_send_sem=p2_send_sem,
      p1_recv_sem=p1_recv_sem,
      p2_recv_sem=p2_recv_sem,
  )

  @jax.named_scope("wait_tec")
  def _signal_and_wait_tec(mb_idx, step):
    slot = mb_idx % 2
    for s in range(dims.num_subcores):
      pl.semaphore_signal(scs_to_tec.at[slot, step], device_id={"subcore": s})
    pl.semaphore_wait(tec_to_scs.at[slot, step], value=dims.num_subcores)

  with jax.named_scope("prologue_mb0"):
    # [Step A]: Start remote D2D copies for micro-batch 0
    dma_manager.start_phase1_d2d_copies(
        mb_idx=0,
        src=x_ref,
        dst=recv_buf_ref.at[0],
    )
    # [Step B]: Wait for micro-batch 0 copies to finish
    dma_manager.wait_phase1_d2d_copies(
        mb_idx=0, src=x_ref, dst=recv_buf_ref.at[0]
    )
    # Wait for TEC to finish Accumulate Phase 1 MB 0, storing the result in `running_sum`
    _signal_and_wait_tec(mb_idx=0, step=0)
    # [Step C]: Start Phase 2 Ring ICI copies for micro-batch 0
    dma_manager.start_phase2_c2c_copies(
        mb_idx=0,
        step_idx=0,
        src=running_sum_ref,
        dst=recv_buf_ref.at[1],
    )

  @pl.loop(0, dims.num_micro_batches)
  @jax.named_scope(f"phase2_step0_loop")
  def step0_loop(mb_idx):
    @pl.when(mb_idx < dims.num_micro_batches - 1)
    def _():
      # [Step D]: Start overlap Phase 1 D2D copies for next micro-batch
      dma_manager.start_phase1_d2d_copies(
          mb_idx=mb_idx + 1,
          src=x_ref,
          dst=recv_buf_ref.at[0],
      )
      dma_manager.wait_phase1_d2d_copies(
          mb_idx=mb_idx + 1, src=x_ref, dst=recv_buf_ref.at[0]
      )

      # Wait for TEC for Accumulate Phase 1 MB i+1, storing the result in `running_sum`
      _signal_and_wait_tec(mb_idx=mb_idx + 1, step=0)

      # [Step F]: Pre-start next micro-batch Phase 2 Ring ICI copies
      dma_manager.start_phase2_c2c_copies(
          mb_idx=mb_idx + 1,
          step_idx=0,
          src=running_sum_ref,
          dst=recv_buf_ref.at[1],
      )

    # Pre-start Phase 2 Step 1 for MB 0 for better overlapping if we're in the last micro-batch
    if dims.num_micro_batches > 1:
      @pl.when(mb_idx == dims.num_micro_batches - 1)
      @jax.named_scope(f"prestart_phase2_c2c_step1_mb0")
      def _():
        dma_manager.start_phase2_c2c_copies(
            mb_idx=0,
            step_idx=1,
            src=running_sum_ref,
            dst=recv_buf_ref.at[2],
        )

    # [Step G]: Wait and Accumulate Phase 2 Step 0 for current micro-batch.
    with jax.named_scope(f"wait_phase2_step0_mb_curr"):
      dma_manager.wait_phase2_c2c_copies(
          mb_idx=mb_idx, step_idx=0, src=running_sum_ref, dst=recv_buf_ref.at[1]
      )
    _signal_and_wait_tec(mb_idx=mb_idx, step=1)

  # Start Phase 2 step 1 copies for MB 0 when there is only 1 micro-batch.
  if dims.num_micro_batches == 1:
    dma_manager.start_phase2_c2c_copies(
        mb_idx=0,
        step_idx=1,
        dst=recv_buf_ref.at[2],
        src=running_sum_ref,
    )

  # Phase 2 step 1 - N
  def do_phase2_step(mb_idx, step_idx):
    dma_manager.wait_phase2_c2c_copies(
        mb_idx=mb_idx,
        step_idx=step_idx,
        src=running_sum_ref,
        dst=recv_buf_ref.at[step_idx + 1],
    )

    @pl.when(mb_idx < dims.num_micro_batches - 1)
    def _():
      dma_manager.start_phase2_c2c_copies(
          mb_idx=mb_idx + 1,
          step_idx=step_idx,
          src=running_sum_ref,
          dst=recv_buf_ref.at[step_idx + 1],
      )

    if dims.num_micro_batches > 1 and step_idx < dims.num_hcube_dims - 1:
      @pl.when(mb_idx == dims.num_micro_batches - 1)
      def _():
        with jax.named_scope(f"start_phase2_step{step_idx + 1}_mb_0"):
          dma_manager.start_phase2_c2c_copies(
              mb_idx=0,
              step_idx=step_idx + 1,
              src=running_sum_ref,
              dst=recv_buf_ref.at[step_idx + 2],
          )

    _signal_and_wait_tec(mb_idx, step_idx + 1)

  # Subsequent Phase 2 hypercube steps
  for step_idx in range(1, dims.num_hcube_dims):

    @pl.loop(0, dims.num_micro_batches)
    @jax.named_scope(f"phase2_step1_loop")
    def step_loop(mb_idx):
      do_phase2_step(mb_idx, step_idx)

    if dims.num_micro_batches == 1 and step_idx < dims.num_hcube_dims - 1:
      with jax.named_scope(f"start_phase2_step{step_idx + 1}_mb_0"):
        dma_manager.start_phase2_c2c_copies(
            mb_idx=0,
            step_idx=step_idx + 1,
            src=running_sum_ref,
            dst=recv_buf_ref.at[step_idx + 2],
        )

  # Resolve un-waited send semaphores
  @pl.loop(0, dims.num_micro_batches)
  def wait_p1_sends_loop(mb_idx):
    dma_manager.wait_phase1_d2d_copies(
        mb_idx=mb_idx, src=x_ref, dst=recv_buf_ref.at[0], wait_send=True
    )

  @pl.loop(0, dims.num_hcube_dims)
  def wait_p2_sends_step_loop(step_idx):
    @pl.loop(0, dims.num_micro_batches)
    def wait_p2_sends_mb_loop(mb_idx):
      dma_manager.wait_phase2_c2c_copies(
          mb_idx=mb_idx,
          step_idx=step_idx,
          src=running_sum_ref,
          dst=recv_buf_ref.at[step_idx + 1],
          wait_send=True,
      )


def tec_kernel(
    x_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    *,
    dims: RsDimensions,
    axis_name: str | tuple[str, ...],
    scs_to_tec,
    tec_to_scs,
    **unused_scratch,
):
  """Executes TensorCore (TEC) execution logic for Reduce-Scatter."""
  num_hcube_dims = dims.num_hcube_dims
  core_idx = jax.lax.axis_index("core")
  subcore_idx = jax.lax.axis_index("subcore")

  topo = Topology(axis_name)
  dma_manager = LocalDmaManager(
      dims,
      topo,
      core_idx,
      subcore_idx,
  )

  # [Step B]: Wait for micro-batch 0 copies to finish, and accumulate locally
  with jax.named_scope("wait_scs"):
    pl.semaphore_wait(scs_to_tec.at[0, 0], value=1)
  with jax.named_scope("accumulate_phase1_mb_0"):
    # [AC P1 MB0]: Accumulate Phase 1 MB 0
    dma_manager.run_phase1_accumulate_pipeline(
        mb_idx=0,
        src1_ref=x_ref,
        src2_ref=recv_buf_ref.at[0],
        out_ref=running_sum_ref,
    )
    # [AC P1 MB0 -> C]: Signal SCS that Accumulate Phase 1 MB 0 is done
    pl.semaphore_signal(tec_to_scs.at[0, 0])

  @pl.loop(0, dims.num_micro_batches)
  def step0_loop(mb_idx):
    curr_slot = mb_idx % 2
    next_slot = 1 - curr_slot

    @pl.when(mb_idx < dims.num_micro_batches - 1)
    def _():
      with jax.named_scope("wait_scs"):
        pl.semaphore_wait(scs_to_tec.at[next_slot, 0], value=1)
      with jax.named_scope(f"accumulate_phase1_mb_{mb_idx + 1}"):
        # [Step E]: Wait and Accumulate Phase 1 for next micro-batch
        dma_manager.run_phase1_accumulate_pipeline(
            mb_idx=mb_idx + 1,
            src1_ref=x_ref,
            src2_ref=recv_buf_ref.at[0],
            out_ref=running_sum_ref,
        )
      pl.semaphore_signal(tec_to_scs.at[next_slot, 0])

    with jax.named_scope("wait_scs"):
      pl.semaphore_wait(scs_to_tec.at[curr_slot, 1], value=1)
    with jax.named_scope(f"accumulate_phase2_step0_mb_curr"):
      # [Step G]: Wait and Accumulate Phase 2 Step 0 for current micro-batch
      dma_manager.run_phase2_accumulate_pipeline(
          mb_idx=mb_idx,
          step_idx=0,
          src1_ref=running_sum_ref,
          src2_ref=recv_buf_ref,
          final_out_ref=output_ref,
      )
    pl.semaphore_signal(tec_to_scs.at[curr_slot, 1])

  def do_phase2_step(mb_idx, step_idx):
    curr_slot = mb_idx % 2
    with jax.named_scope("wait_scs"):
      pl.semaphore_wait(scs_to_tec.at[curr_slot, 1 + step_idx], value=1)
    with jax.named_scope(f"accumulate_phase2_step{step_idx}"):
      dma_manager.run_phase2_accumulate_pipeline(
          mb_idx=mb_idx,
          step_idx=step_idx,
          src1_ref=running_sum_ref,
          src2_ref=recv_buf_ref,
          final_out_ref=output_ref,
      )
    pl.semaphore_signal(tec_to_scs.at[curr_slot, 1 + step_idx])

  # [Step H/I/J equivalent]: Subsequent Phase 2 hypercube steps
  for step_idx in range(1, num_hcube_dims):

    @pl.loop(0, dims.num_micro_batches)
    def step_loop(mb_idx):
      do_phase2_step(mb_idx, step_idx)


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int | None = None,
    axis_name: str | tuple[str, ...] = "x",
) -> jax.Array:
  """Performs a hierarchical Reduce-Scatter on SparseCore.

    local_x: Input array shard local to the device.
    num_devices: Total number of devices involved in the reduction.
    num_micro_batches: Pipeline unrolling factor for D2D/ICI/ALU/DMA
    overlapping.
    axis_name: Mesh axis name mapped to ShardMap.

  Returns:
    The reduced output array.
  """
  num_tokens, hidden_dim_size = local_x.shape
  chunk_size_orig = num_tokens // num_devices
  min_chunk_size = 16
  needs_padding = chunk_size_orig < min_chunk_size

  if needs_padding:
    # TODO: Enable lower chunk size
    reshaped_x = local_x.reshape(num_devices, -1, hidden_dim_size)
    padded_x = jnp.pad(
        reshaped_x,
        ((0, 0), (0, min_chunk_size - chunk_size_orig), (0, 0)),
    )
    local_x = padded_x.reshape(-1, hidden_dim_size)
    num_tokens = local_x.shape[0]

  chunk_size = num_tokens // num_devices

  if num_micro_batches is None:
    if num_tokens >= 4096:
      num_micro_batches = 8
    elif num_tokens >= 2048:
      num_micro_batches = 4
    elif num_tokens >= 256:
      num_micro_batches = 2
    else:
      num_micro_batches = 1

  assert hidden_dim_size % num_micro_batches == 0, (
      f"hidden_dim_size {hidden_dim_size} must be divisible by "
      f"num_micro_batches {num_micro_batches}"
  )
  mb_size = hidden_dim_size // num_micro_batches
  num_chips = num_devices // 2
  assert (
      num_chips & (num_chips - 1)
  ) == 0, f"num_chips {num_chips} must be a power of 2"
  num_hcube_dims = int(math.log2(num_chips))
  assert num_hcube_dims >= 1, f"num_hcube_dims {num_hcube_dims} must be >= 1"
  hc_chunk_size = mb_size // num_hcube_dims

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None

  # Use both cores to maximize aggregate HBM memory bandwidth.
  num_cores = 2
  # Shard column as much as possible as long as the chunk's width >= 128, which is for dma alignment. This is because we usually have limited number of rows.
  num_subcores_col = min(16, hc_chunk_size // 128)
  # Rows are sharded both core and remaining subcores.
  num_subcores_row = min(
      16 // num_subcores_col,
      chunk_size // (num_cores * _get_packing_factor(local_x.dtype)),
  )

  dims = RsDimensions(
      num_chips=num_chips,
      num_hcube_dims=num_hcube_dims,
      num_micro_batches=num_micro_batches,
      hidden_dim_size=hidden_dim_size,
      chunk_size=chunk_size,
      core_chunk_size=chunk_size // num_cores,
      subcore_chunk_size=chunk_size // num_cores // num_subcores_row,
      mb_size=mb_size,
      hc_chunk_size=hc_chunk_size,
      num_subcores_row=num_subcores_row,
      num_subcores_col=num_subcores_col,
      subcore_col_chunk_size_p1=mb_size // num_subcores_col,
      subcore_col_chunk_size_p2=hc_chunk_size // num_subcores_col,
  )

  scs_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=num_cores)
  tec_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core",
      subcore_axis_name="subcore",
      num_cores=num_cores,
      num_subcores=dims.num_subcores,
  )

  out, _, _ = pl.kernel(
      interpret=False,
      body=[
          functools.partial(scs_kernel, dims=dims, axis_name=axis_name),
          functools.partial(tec_kernel, dims=dims, axis_name=axis_name),
      ],
      mesh=[scs_mesh, tec_mesh],
      out_type=(
          # output
          jax.ShapeDtypeStruct(
              (dims.chunk_size, hidden_dim_size), local_x.dtype
          ),
          # running_sum[i, ...]: The accumulated result at each step (i=0 is reserved for Phase 1)
          jax.ShapeDtypeStruct((num_hcube_dims, *local_x.shape), local_x.dtype),
          # recv_buf[i, ...]: The received data from peer at each step (i=0 is reserved for Phase 1)
          jax.ShapeDtypeStruct(
              (num_hcube_dims + 1, *local_x.shape), local_x.dtype
          ),
      ),
      scratch_types=dict(
          scs_to_tec=pltpu.SemaphoreType.REGULAR((2, num_hcube_dims + 1))
          @ tec_mesh,
          tec_to_scs=pltpu.SemaphoreType.REGULAR((2, num_hcube_dims + 1))
          @ scs_mesh,
          p1_send_sem=pltpu.SemaphoreType.DMA((2, num_devices // 2)) @ scs_mesh,
          p2_send_sem=pltpu.SemaphoreType.DMA(
              (2, num_hcube_dims, dims.num_chips // 2, num_hcube_dims)
          )
          @ scs_mesh,
          p1_recv_sem=pltpu.SemaphoreType.DMA((2, num_devices // 2)) @ scs_mesh,
          p2_recv_sem=pltpu.SemaphoreType.DMA(
              (2, num_hcube_dims, dims.num_chips // 2, num_hcube_dims)
          )
          @ scs_mesh,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          needs_layout_passes=False,
          has_side_effects=True,
          disable_bounds_checks=True,
      ),
  )(local_x)

  if needs_padding:
    out = out[:chunk_size_orig, :]
  return out


def hierarchical_reduce_scatter(
    x: jax.Array,
    mesh: jax.sharding.Mesh,
    num_micro_batches: int | None = None,
) -> jax.Array:
  """Performs a hierarchical Reduce-Scatter on SparseCore."""
  out_specs = jax.sharding.PartitionSpec("x", None)

  return shard_map.shard_map(
      lambda local_x: hierarchical_reduce_scatter_local(
          local_x,
          num_devices=mesh.devices.size,
          num_micro_batches=num_micro_batches,
      ),
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec("x", None),
      out_specs=out_specs,
      check_rep=False,
  )(x)

