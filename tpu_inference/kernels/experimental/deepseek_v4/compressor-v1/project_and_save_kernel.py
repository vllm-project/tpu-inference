import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class TileSizes:
  """Tile sizes for Pallas kernel."""

  tile_m: int  # output dimension
  tile_k: int  # hidden dimension
  tile_n: int  # sequence length


@dataclasses.dataclass(frozen=True)
class Dimensions:
  """Dimensions of the input and output tensors."""

  size_m: int  # output dimension
  size_k: int  # hidden dimension
  size_n: int  # sequence length

  # other values
  compress_ratio: int


@dataclasses.dataclass(frozen=True)
class Configs:
  """Configuration parameters for the projection and save state kernel."""

  tile_sizes: TileSizes
  dims: Dimensions
  token_page_size: int

  @property
  def num_m(self) -> int:
    return pl.cdiv(self.dims.size_m, self.tile_sizes.tile_m)

  @property
  def num_k(self) -> int:
    return pl.cdiv(self.dims.size_k, self.tile_sizes.tile_k)

  @property
  def num_n(self) -> int:
    return pl.cdiv(self.dims.size_n, self.tile_sizes.tile_n)


def load_from_acc_ref(ref, dtype=None):
  """Loads data from HBM with strided access, handling 128-lane alignment."""
  #  (tile_n, slots_per_m_tile, 1, 128) -> (tile_n, state_width)
  tile_n, slots_per_m_tile, _, _ = ref.shape
  num_lanes = pltpu.get_tpu_info().num_lanes
  ref_flat = ref.reshape(-1, num_lanes)  # tile_n * slots_per_m_tile, 128

  # Gather the chunks into the original head dimension.
  chunks = [
      ref_flat[pl.ds(m, tile_n, slots_per_m_tile)]
      for m in range(slots_per_m_tile)
  ]
  vec = jnp.concat(chunks, axis=1)
  if dtype is not None:
    return pltpu.bitcast(vec, dtype)
  return vec


def store_to_acc_ref(ref, val, dtype=None):
  # val should be (tile_n, state_width)
  num_lanes = pltpu.get_tpu_info().num_lanes

  tile_n, slots_per_m_tile, _, _ = ref.shape
  ref_flat = ref.reshape(-1, num_lanes)

  for i in range(slots_per_m_tile):
    val_slice = val[:, i * num_lanes : (i + 1) * num_lanes]
    ref_flat[pl.ds(i, tile_n, slots_per_m_tile)] = val_slice


def kernel(
    # scalar prefetch
    slot_mapping,
    positions,
    # input
    hidden_states,  # [num_tokens, hidden_size]
    wkv_wgate,  # [2 * state_width, hidden_size]
    ape,  # [compress_ratio, state_width]
    cache,  # [num_pages, page_size, 4, 128] uint8 (aliased to output)
    # output
    _,  # unused output ref (aliased to cache)
    # scratch_memory
    acc_ref,  # (2, tile_n, slots_per_m_tile, 1, 128)
    sem_ref,
    write_counts_ref,  # [2]
    *,
    cfgs: Configs,
):
  """Inner kernel executing matmul, APE addition, and paged writes."""
  tile_m = cfgs.tile_sizes.tile_m
  tile_n = cfgs.tile_sizes.tile_n
  slots_per_m_tile = tile_m // 128
  acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
  write_counts_ref[0] = 0
  write_counts_ref[1] = 0

  def _dma_copy_out(n_idx_val, m_idx_val, buf_idx_val):
    """Copys one tile from accumulator to the KV cache.

    We run the dma one tile after the previous tile is done computing, using a
    double buffer on the k dimension.

    The reason we do this in the following tile is that we hope to pipeline the
    DMA with the matmuls.

    (however, this is awaiting RAW no-hazard flag to land)
    """
    slot_offset = m_idx_val * slots_per_m_tile
    num_valid_writes = 0
    for i in range(tile_n):
      global_token_idx = n_idx_val * tile_n + i
      global_slot_idx = slot_mapping[global_token_idx]
      valid = global_slot_idx >= 0
      num_valid_writes = num_valid_writes + lax.select(valid, 1, 0)

      # Guard index calculations to avoid out-of-bounds even for size 0
      safe_slot_idx = lax.select(valid, global_slot_idx, 0)
      page_size_tokens = cfgs.token_page_size
      p = safe_slot_idx // page_size_tokens
      start_slot_token = safe_slot_idx % page_size_tokens
      start_slot = start_slot_token + slot_offset

      size = lax.select(valid, slots_per_m_tile, 0)
      dest = cache.at[p, pl.ds(start_slot, size), :, :]
      src = acc_ref_u8.at[buf_idx_val, i, pl.ds(0, size), :, :]
      pltpu.make_async_copy(src, dest, sem_ref.at[buf_idx_val]).start()

    write_counts_ref[buf_idx_val] = num_valid_writes

  def inner_kernel(
      tiled_wgate_ref,  # (tile_k, tile_m)
      tiled_hidden_states_ref,  # (tile_n, tile_k)
      tiled_positions_ref,  # (tile_n,)
  ):
    n_idx = pl.program_id(0)
    m_idx = pl.program_id(1)
    k_idx = pl.program_id(2)
    tile_idx = (
        n_idx * cfgs.num_m + m_idx
    )  # omit k_idx because we write to the same buffer, double buffer on k
    buf_idx = tile_idx % 2

    state_width = cfgs.dims.size_m // 2

    def _matmul():
      lhs = tiled_hidden_states_ref[...]
      rhs = tiled_wgate_ref[...]
      prod = lax.dot_general(
          lhs,
          rhs,
          (((1,), (0,)), ((), (()))),
          preferred_element_type=jnp.float32,
      )
      return prod

    def _add_ape():
      tile_positions = tiled_positions_ref[...]
      ape_rows = tile_positions % cfgs.dims.compress_ratio
      ape_val = ape[...].astype(jnp.float32)
      # Gather Path: doesn't work for compiler reasons?
      # gather_indices = jnp.broadcast_to(
      #     ape_rows[:, None], (tile_n, ape_val.shape[1])
      # )
      # ape_selected = jnp.take_along_axis(ape_val, gather_indices, axis=0)
      # matmul path
      one_hot = (
          ape_rows[:, None] == jnp.arange(cfgs.dims.compress_ratio)[None, :]
      ).astype(jnp.float32)
      ape_selected = jnp.matmul(
          one_hot, ape_val, precision=lax.Precision.HIGHEST
      )  #  (tile_n, state_width)
      existing_acc = load_from_acc_ref(
          acc_ref.at[buf_idx, ...], dtype=jnp.float32
      )
      new_acc = existing_acc + ape_selected  # (tile_n, state_width)
      store_to_acc_ref(acc_ref.at[buf_idx], new_acc)

    def process(copy_dma: bool, add_ape: bool):
      # 1. Semaphore wait (only on first step)
      @pl.when(k_idx == 0)
      def _wait_for_buffer():
        count = write_counts_ref[buf_idx][...]

        @pl.when(count > 0)
        def _do_wait():
          pl.semaphore_wait(sem_ref.at[buf_idx], count)

      # 2. Launch DMA for the PREVIOUS tile
      if copy_dma:
        prev_tile_idx = tile_idx - 1
        prev_n_idx = prev_tile_idx >> 1
        prev_m_idx = prev_tile_idx & 1
        prev_buf_idx = 1 - buf_idx
        _dma_copy_out(prev_n_idx, prev_m_idx, prev_buf_idx)

      # 3. Matmul
      prod = _matmul()

      # 4. Unified Accumulate/Assign
      acc_val = load_from_acc_ref(acc_ref.at[buf_idx, ...], dtype=jnp.float32)
      old_acc = lax.select(
          k_idx > 0,
          acc_val,
          jnp.zeros_like(acc_val),
      )
      new_acc = old_acc + prod
      store_to_acc_ref(acc_ref.at[buf_idx], new_acc)

      # 5. APE addition if last
      if add_ape:
        _add_ape()

    # Select and execute
    num_k = cfgs.num_k
    is_last_k = k_idx == (num_k - 1)
    is_score = m_idx * tile_m >= state_width

    copy_dma = (k_idx == 0) & (tile_idx > 0)
    add_ape = is_last_k & is_score

    @pl.when(jnp.logical_not(copy_dma) & jnp.logical_not(add_ape))
    @jax.named_scope("matmul")
    def matmul():
      process(False, False)

    @pl.when(copy_dma & add_ape)
    @jax.named_scope("matmul_ape_write_out")
    def matmul_ape_write_out():
      process(True, True)

    @pl.when(copy_dma & jnp.logical_not(add_ape))
    @jax.named_scope("matmul_write_out")
    def matmul_write_out():
      process(True, False)

    @pl.when(jnp.logical_not(copy_dma) & add_ape)
    @jax.named_scope("matmul_ape_only")
    def matmul_ape_only():
      process(False, True)

  wkv_spec = pl.BlockSpec(
      (cfgs.tile_sizes.tile_k, cfgs.tile_sizes.tile_m),
      lambda n, m, k: (k, m),
      memory_space=pltpu.VMEM,
  )
  hidden_states_spec = pl.BlockSpec(
      (cfgs.tile_sizes.tile_n, cfgs.tile_sizes.tile_k),
      lambda n, m, k: (n, k),
      memory_space=pltpu.VMEM,
  )
  positions_spec = pl.BlockSpec(
      (cfgs.tile_sizes.tile_n,),
      lambda n, m, k: (n,),
      memory_space=pltpu.VMEM,
  )

  pipeline_fn = pltpu.emit_pipeline(
      inner_kernel,
      grid=(cfgs.num_n, cfgs.num_m, cfgs.num_k),
      in_specs=(
          wkv_spec,
          hidden_states_spec,
          positions_spec,
      ),
  )

  pipeline_fn(
      wkv_wgate,
      hidden_states,
      positions,
  )

  # Epilogue: launch DMA for the last tile and wait for all outstanding DMAs
  last_tile_idx = cfgs.num_n * cfgs.num_m - 1
  last_n_idx = last_tile_idx // cfgs.num_m
  last_m_idx = last_tile_idx % cfgs.num_m
  last_buf_idx = last_tile_idx % 2

  if last_tile_idx >= 0:
    _dma_copy_out(last_n_idx, last_m_idx, last_buf_idx)

  for b in range(2):
    count = write_counts_ref[b][...]

    @pl.when(count > 0)
    def _wait_epilogue(b=b, count=count):
      pl.semaphore_wait(sem_ref.at[b], count)


@functools.partial(
    jax.jit,
    static_argnames=[
        "compress_ratio",
    ],
)
def proj_and_save_state(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    wkv_wgate: jax.Array,  # [hidden_size, 2 * state_width]
    ape: jax.Array,  # [compress_ratio, state_width]
    positions: jax.Array,  # [num_tokens]
    slot_mapping: jax.Array,  # [num_tokens]
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8
    compress_ratio: int,
) -> jax.Array:
  """Projects hidden states and saves them to the KV cache in HBM.

  Args:
    hidden_states: Input hidden states.
    wkv_wgate: Projection weights.
    ape: Absolute Position Embeddings.
    positions: Token positions.
    slot_mapping: Mapping from token index to cache slot.
    cache: KV cache tensor in HBM (updated in-place).
    compress_ratio: APE compression ratio.

  Returns:
    The updated KV cache tensor.
  """
  num_tokens, hidden_size = hidden_states.shape
  _, state_dim = wkv_wgate.shape

  # 1. Geometry & Cache Setup
  token_page_size = cache.shape[1]

  dims = Dimensions(
      size_m=state_dim,
      size_k=hidden_size,
      size_n=num_tokens,
      compress_ratio=compress_ratio,
  )

  # Static tile sizes for Pallas (tile_m is set to state_width)
  state_width = state_dim // 2
  tile_m = state_width
  tile_k = 3584  # default
  tile_n = 128

  tile_sizes = TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

  cfgs = Configs(
      tile_sizes=tile_sizes,
      dims=dims,
      token_page_size=token_page_size,
  )

  # 2. Scratch & Semaphores for Double Buffering
  slots_per_m_tile = tile_m // 128
  # Think of slots_per_m_tile as how many chunks of size (1, 128) fit into a single M tile?
  scratch_shapes = [
      pltpu.VMEM((2, tile_n, slots_per_m_tile, 1, 128), jnp.float32),
      pltpu.SemaphoreType.REGULAR((2,)),
      pltpu.SMEM((2,), jnp.int32),
  ]

  out_shape = jax.ShapeDtypeStruct(cache.shape, cache.dtype)

  return pl.pallas_call(
      functools.partial(kernel, cfgs=cfgs),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(),
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),  # positions
              pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states
              pl.BlockSpec(memory_space=pltpu.HBM),  # wkv_wgate
              pl.BlockSpec(memory_space=pltpu.VMEM),  # ape (prefetch to VMEM)
              pl.BlockSpec(memory_space=pltpu.HBM),  # cache
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          scratch_shapes=scratch_shapes,
      ),
      input_output_aliases={5: 0},  # Alias output to cache (index 5)
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
      ),
      name="proj_and_save_state",
  )(slot_mapping, positions, hidden_states, wkv_wgate, ape, cache)