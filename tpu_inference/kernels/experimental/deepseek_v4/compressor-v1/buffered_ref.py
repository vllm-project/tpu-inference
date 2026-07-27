import dataclasses
import os
import sys
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
  sys.path.insert(0, _DIR)

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _BufferedRef(pltpu.BufferedRef):
  cfgs: config.Configs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: config.Configs,
  ):
    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfgs=cfgs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(_BufferedRef):

  def copy_in(self, src_ref, grid_indices):
    cos_sin_cache_ref, positions_ref = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    pid = grid_indices[0]

    dest_ref = self.window_ref.at[slot]  # (tile_n, rope_head_dim)

    tile_n = self.cfgs.tile_sizes.tile_n
    compress_ratio = self.cfgs.dims.compress_ratio

    @pl.loop(0, tile_n, unroll=True)
    def loop_body(i):
      global_idx = pid * tile_n + i
      position = positions_ref[global_idx]
      compressed_pos = (position // compress_ratio) * compress_ratio

      src = cos_sin_cache_ref.at[
          compressed_pos, pl.ds(0, self.window_ref.shape[-1])
      ]
      dest = dest_ref.at[i, :]
      pltpu.make_async_copy(src, dest, sem).start()

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class PageBufferRef(_BufferedRef):
  # VMEM block: cfgs.page_buffer_shape()
  #   (tile_n, pages_to_buffer_per_token, page_size, hbm_pack, LANE) uint8

  def copy_in(self, src_ref, grid_indices):
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    pid = grid_indices[0]

    (
        state_cache_ref,
        positions_ref,
        block_table_ref,
        token_to_req_indices_ref,
    ) = src_ref

    tile_n = self.cfgs.tile_sizes.tile_n
    pages_to_buffer_per_token = self.cfgs.pages_to_buffer_per_token
    block_size = self.cfgs.dims.block_size

    page_buffer_ref = self.window_ref.at[slot]
    global_idx = pid * tile_n

    @pl.loop(0, tile_n * pages_to_buffer_per_token, unroll=True)
    def loop_body(idx):
      n = idx // pages_to_buffer_per_token
      p = idx % pages_to_buffer_per_token
      idx_n = global_idx + n

      position = positions_ref[idx_n]
      req_idx = token_to_req_indices_ref[idx_n]

      block_idx = position // block_size - (pages_to_buffer_per_token - 1) + p
      safe_block_idx = jnp.maximum(block_idx, 0)
      page = block_table_ref[req_idx, safe_block_idx]

      src = state_cache_ref.at[page, :, :, :]
      dest = page_buffer_ref.at[n, p, :, :, :]
      pltpu.make_async_copy(src, dest, sem).start()

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutputRef(_BufferedRef):
  # VMEM block: cfgs.output_shape()
  #   (tile_n, record_slots, hbm_pack, LANE) uint8 (bf16 payload for HCA)

  def copy_out(self, dest_ref, grid_indices):
    nope_cache_ref, kv_slot_mapping_ref = dest_ref
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]

    src_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]

      @pl.when(kv_slot >= 0)
      def _write_nope():
        p = kv_slot // self.cfgs.page_size
        s = kv_slot % self.cfgs.page_size

        @pl.loop(0, self.cfgs.record_slots, unroll=True)
        def write_loop(o):
          src_nope = src_ref.at[n, o, :, :]
          dest_nope = nope_cache_ref.at[p, s + o, :, :]
          pltpu.make_async_copy(src_nope, dest_nope, sem).start()

  def wait_out(self, dest_ref, grid_indices):
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    _, kv_slot_mapping_ref = dest_ref

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]

      @pl.when(kv_slot >= 0)
      def _wait_nope():
        @pl.loop(0, self.cfgs.record_slots, unroll=True)
        def wait_loop(o):
          pltpu.make_async_copy(
              vmem_ref.at[n, o, :, :],
              vmem_ref.at[n, o, :, :],
              sem,
          ).wait()


def rope_target(cfgs: config.Configs, kv_slot):
  """Maps a logical rope slot to its (page, packed row, sub-slot)."""
  page_size = cfgs.page_size
  rope_pack = cfgs.rope_pack
  page = kv_slot // page_size
  offset = kv_slot % page_size
  return page, offset // rope_pack, offset % rope_pack


def rope_store_pred(
    cfgs: config.Configs,
    idx,
    kv_slot_mapping_ref,
    token_to_req_indices_ref,
    distribution_ref,
):
  """Whether token `idx`'s rope record belongs to this invocation's store.

  The kernel handles one class of token per invocation, so a token counts only
  if it is a boundary token *and* on the side of ``distribution[0]`` that
  matches ``cfgs.rope_store_mode``. Every place that touches the rope cache --
  the copy-in, the merge in the kernel body, and the copy-out -- has to agree
  on this, or a DMA gets started without a matching wait.

  Args:
    cfgs: kernel configuration.
    idx: token index, possibly past the end of the padded batch.
    kv_slot_mapping_ref: [size_n] destination slots, negative for "skip".
    token_to_req_indices_ref: [size_n] request index per token.
    distribution_ref: i32[3] decode / chunked-prefill / mixed request split.

  Returns:
    (store, kv_slot) where kv_slot is safe to index with (clamped in range).
  """
  in_range = idx < cfgs.dims.size_n
  safe_idx = jnp.where(in_range, idx, 0)
  kv_slot = kv_slot_mapping_ref[safe_idx]
  is_decode = token_to_req_indices_ref[safe_idx] < distribution_ref[0]
  in_mode = is_decode if cfgs.rope_store_mode == "decode" else ~is_decode
  store = (kv_slot >= 0) & in_mode & in_range
  return store, jnp.maximum(kv_slot, 0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoPERef(_BufferedRef):
  """Read-modify-write of one packed rope row per boundary decode token.

  VMEM block: cfgs.rope_output_shape() -> (tile_n, rope_pack, LANE) uint8.
  Only used for CSA (has_rope_cache=True), in "decode" rope_store_mode.

  A token's rope record is a single LANE-byte sub-slot, but the cache packs
  rope_pack records per row, so an isolated store has to read the whole row in,
  let the kernel body overwrite its own sub-slot, and write the row back.

  Decode tokens are sparse -- one per request, and requests do not share pages
  -- so no two of them ever land in the same row and the per-token writes can
  never collide. Prefill tokens are boundary-dense and would; they go through
  RoPERowRef instead. Both start and wait are gated on the same predicate so
  the DMA semaphore counts stay balanced.
  """

  def _target(self, kv_slot):
    """Maps a logical rope slot to its (page, packed row, sub-slot)."""
    return rope_target(self.cfgs, kv_slot)

  def _store_pred(self, idx_n, dest_ref):
    """True when token `idx_n` is a boundary decode token we must store."""
    _, kv_slot_mapping_ref, token_to_req_indices_ref, distribution_ref = dest_ref
    store, _ = rope_store_pred(
        self.cfgs,
        idx_n,
        kv_slot_mapping_ref,
        token_to_req_indices_ref,
        distribution_ref,
    )
    return store

  def _rows(self, dest_ref, grid_indices, *, vmem_ref, hbm_ref, sem, to_hbm):
    """Starts the per-token row DMAs in whichever direction `to_hbm` picks."""
    _, kv_slot_mapping_ref, _, _ = dest_ref
    pid = grid_indices[0]
    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n

      @pl.when(self._store_pred(idx_n, dest_ref))
      def _copy_row():
        p, row, _ = self._target(kv_slot_mapping_ref[idx_n])
        vmem_row = vmem_ref.at[n, :, :]
        hbm_row = hbm_ref.at[p, row, :, :]
        src, dst = (vmem_row, hbm_row) if to_hbm else (hbm_row, vmem_row)
        pltpu.make_async_copy(src, dst, sem).start()

  def _wait_rows(self, dest_ref, grid_indices, *, vmem_ref, sem):
    """Waits exactly the DMAs `_rows` started for this grid step."""
    pid = grid_indices[0]
    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n

      @pl.when(self._store_pred(idx_n, dest_ref))
      def _wait_row():
        pltpu.make_async_copy(
            vmem_ref.at[n, :, :], vmem_ref.at[n, :, :], sem
        ).wait()

  def copy_in(self, src_ref, grid_indices):
    slot = self.current_copy_in_slot
    self._rows(
        src_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        hbm_ref=src_ref[0],
        sem=self.sem_recvs.at[slot],
        to_hbm=False,
    )

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    self._wait_rows(
        src_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        sem=self.sem_recvs.at[slot],
    )

  def copy_out(self, dest_ref, grid_indices):
    slot = self.current_copy_out_slot
    self._rows(
        dest_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        hbm_ref=dest_ref[0],
        sem=self.sem_sends.at[slot],
        to_hbm=True,
    )

  def wait_out(self, dest_ref, grid_indices):
    slot = self.current_wait_out_slot
    self._wait_rows(
        dest_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        sem=self.sem_sends.at[slot],
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoPERowRef(_BufferedRef):
  """Read-modify-write of the one packed rope row a whole grid step shares.

  VMEM block: cfgs.rope_output_shape() -> (1, rope_pack, LANE) uint8, used in
  "prefill" rope_store_mode.

  Prefill boundary tokens are dense: consecutive tokens of a request land in
  consecutive sub-slots, so up to rope_pack of them share one packed row. The
  caller is required to lay the batch out one row per tile (see
  compressor._row_tiled_boundary_batch), which makes the whole tile a single
  row: fetch it once, let the kernel body merge every record of the tile into
  it, write it back once. A per-token read-modify-write would instead have each
  token write back its own stale copy of the row.

  The row is still read first, not just overwritten: a row can be filled across
  two calls when a chunk boundary splits it, and a tile can cover it only
  partially.
  """

  def _tile_row(self, args, grid_indices):
    """The (page, row) every storing token of this grid step shares."""
    _, kv_slot_mapping_ref, token_to_req_indices_ref, distribution_ref = args
    global_idx = grid_indices[0] * self.cfgs.tile_sizes.tile_n

    # All storing tokens of the tile share a row, so the first one names it.
    kv_slot = jnp.int32(-1)
    for n in range(self.cfgs.tile_sizes.tile_n):
      store, slot_n = rope_store_pred(
          self.cfgs,
          global_idx + n,
          kv_slot_mapping_ref,
          token_to_req_indices_ref,
          distribution_ref,
      )
      kv_slot = jnp.where(store & (kv_slot < 0), slot_n, kv_slot)

    page, row, _ = rope_target(self.cfgs, jnp.maximum(kv_slot, 0))
    return page, row, kv_slot >= 0

  def _copy_row(self, args, grid_indices, *, vmem_ref, hbm_ref, sem, to_hbm):
    page, row, any_store = self._tile_row(args, grid_indices)

    @pl.when(any_store)
    def _row_dma():
      vmem_row = vmem_ref.at[0, :, :]
      hbm_row = hbm_ref.at[page, row, :, :]
      src, dst = (vmem_row, hbm_row) if to_hbm else (hbm_row, vmem_row)
      pltpu.make_async_copy(src, dst, sem).start()

  def _wait_row(self, args, grid_indices, *, vmem_ref, sem):
    _, _, any_store = self._tile_row(args, grid_indices)

    @pl.when(any_store)
    def _row_wait():
      pltpu.make_async_copy(
          vmem_ref.at[0, :, :], vmem_ref.at[0, :, :], sem
      ).wait()

  def copy_in(self, src_ref, grid_indices):
    slot = self.current_copy_in_slot
    self._copy_row(
        src_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        hbm_ref=src_ref[0],
        sem=self.sem_recvs.at[slot],
        to_hbm=False,
    )

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    self._wait_row(
        src_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        sem=self.sem_recvs.at[slot],
    )

  def copy_out(self, dest_ref, grid_indices):
    slot = self.current_copy_out_slot
    self._copy_row(
        dest_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        hbm_ref=dest_ref[0],
        sem=self.sem_sends.at[slot],
        to_hbm=True,
    )

  def wait_out(self, dest_ref, grid_indices):
    slot = self.current_wait_out_slot
    self._wait_row(
        dest_ref,
        grid_indices,
        vmem_ref=self.window_ref.at[slot],
        sem=self.sem_sends.at[slot],
    )


def create_allocs_and_specs(
    cfgs: config.Configs,
    *,
    cache_ref,
    rope_cache_ref,
    cos_sin_cache_ref,
    positions_ref,
    block_table_ref,
    token_to_req_indices_ref,
    kv_slot_mapping_ref,
    distribution_ref,
) -> tuple[
    tuple[CosSinRef | None, OutputRef, PageBufferRef, RoPERef | None],
    tuple[pl.BlockSpec | None, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec | None],
    tuple[
        tuple[Any, ...] | None,
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[Any, ...] | None,
    ],
]:
  # cos_sin
  if cfgs.dims.has_rope:
    tile_n = cfgs.tile_sizes.tile_n
    cos_sin_spec = pl.BlockSpec(
        block_shape=cfgs.cos_sin_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i * tile_n, 0),
    )
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec,
        dtype_or_type=cfgs.dims.cos_sin_dtype,
        buffer_type=pltpu.BufferType.INPUT,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    cos_sin_args = (cos_sin_cache_ref, positions_ref)
  else:
    cos_sin_alloc = cos_sin_spec = cos_sin_args = None

  # output
  output_spec = pl.BlockSpec(
      block_shape=cfgs.output_shape(),
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0, 0, 0),
  )
  output_alloc = OutputRef.create(
      spec=output_spec,
      dtype_or_type=jnp.uint8,
      buffer_type=pltpu.BufferType.OUTPUT,
      buffer_count=1,
      use_lookahead=False,
      cfgs=cfgs,
  )
  output_args = (cache_ref, kv_slot_mapping_ref)

  # page_buffer
  page_buffer_spec = pl.BlockSpec(
      block_shape=cfgs.page_buffer_shape(),
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0, 0, 0),
  )
  page_buffer_alloc = PageBufferRef.create(
      spec=page_buffer_spec,
      dtype_or_type=jnp.uint8,
      buffer_type=pltpu.BufferType.INPUT,
      buffer_count=2,
      use_lookahead=False,
      cfgs=cfgs,
  )
  page_buffer_args = (
      cache_ref,
      positions_ref,
      block_table_ref,
      token_to_req_indices_ref,
  )

  # rope
  if cfgs.dims.has_rope_cache:
    rope_spec = pl.BlockSpec(
        block_shape=cfgs.rope_output_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0),
    )
    rope_cls = RoPERef if cfgs.rope_store_mode == "decode" else RoPERowRef
    rope_alloc = rope_cls.create(
        spec=rope_spec,
        dtype_or_type=jnp.uint8,
        # Read-modify-write: the row is fetched, patched by the body, stored.
        buffer_type=pltpu.BufferType.INPUT_OUTPUT,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    rope_args = (
        rope_cache_ref,
        kv_slot_mapping_ref,
        token_to_req_indices_ref,
        distribution_ref,
    )
  else:
    rope_alloc = rope_spec = rope_args = None

  return (
      (cos_sin_alloc, output_alloc, page_buffer_alloc, rope_alloc),
      (cos_sin_spec, output_spec, page_buffer_spec, rope_spec),
      (cos_sin_args, output_args, page_buffer_args, rope_args),
  )