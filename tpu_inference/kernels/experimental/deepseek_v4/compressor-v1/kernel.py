import functools
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
  sys.path.insert(0, _DIR)

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

import buffered_ref
import compute
import config


def _merge_rope_records(
    rope_out_vmem,
    rope_record,
    global_idx,
    kv_slot_mapping_ref,
    token_to_req_indices_ref,
    distribution_ref,
    cfgs: config.Configs,
):
  """Merges the tile's rope records into the staged packed row(s).

  `rope_out_vmem` holds the rows this tile's records belong to, already fetched
  from HBM by the rope BufferedRef: one row per token in "decode" mode, one row
  for the whole tile in "prefill" mode. Each record replaces its own sub-slot
  and the rest of the row is left as fetched, so copy_out can write whole rows
  back. Selecting beats a dynamic sub-slot store: the sub-slot index is the
  second-minor (tiled) dim.

  Args:
    rope_out_vmem: (rows, rope_pack, LANE) uint8 staged rows.
    rope_record: (tile_n, 1, LANE) uint8 packed record per token.
    global_idx: index of the tile's first token.
    kv_slot_mapping_ref: [size_n] destination slots.
    token_to_req_indices_ref: [size_n] request index per token.
    distribution_ref: i32[3] request split.
    cfgs: kernel configuration.
  """
  rope_pack = cfgs.rope_pack
  pack_iota = jnp.arange(rope_pack)[None, :, None]  # (1, rope_pack, 1)

  subs = []
  for i in range(cfgs.tile_sizes.tile_n):
    store, kv_slot = buffered_ref.rope_store_pred(
        cfgs,
        global_idx + i,
        kv_slot_mapping_ref,
        token_to_req_indices_ref,
        distribution_ref,
    )
    subs.append(jnp.where(store, kv_slot % rope_pack, -1))

  if cfgs.rope_store_mode == "decode":
    # Row per token: every token patches its own staged row independently.
    sub = jnp.stack(subs)  # (tile_n,)
    rope_out_vmem[...] = jnp.where(
        pack_iota == sub[:, None, None], rope_record, rope_out_vmem[...]
    )
  else:
    # One row for the tile: fold every record into it before writing back.
    row = rope_out_vmem[...]  # (1, rope_pack, LANE)
    for i, sub in enumerate(subs):
      row = jnp.where(pack_iota == sub, rope_record[i][None], row)
    rope_out_vmem[...] = row


def inner_kernel(
    cos_sin_vmem,
    out_vmem,
    page_buffer_vmem,
    rope_out_vmem,
    positions_ref,
    kv_slot_mapping_ref,
    token_to_req_indices_ref,
    distribution_ref,
    rms_weight_vmem_ref,
    window_vmem,
    *,
    cfgs: config.Configs,
):
  tile_n = cfgs.tile_sizes.tile_n
  window = cfgs.window
  head_tiles = cfgs.head_tiles

  pid = pl.program_id(0)
  global_idx = pid * tile_n

  # Per-token window start = position - window + 1
  start_list = []
  for i in range(tile_n):
    idx = global_idx + i
    safe_idx = jax.lax.select(idx < cfgs.dims.size_n, idx, 0)
    start_list.append(positions_ref[safe_idx] - window + 1)
  start = jnp.stack(start_list)  # (tile_n,)

  window_u8 = window_vmem.bitcast(jnp.uint8).reshape(2, tile_n, window, head_tiles, 4, 128)
  kv_window_u8 = window_u8.at[0]
  score_window_u8 = window_u8.at[1]

  compute.gather_from_page_buffer(
      page_buffer=page_buffer_vmem,
      positions_ref=positions_ref,
      kv_window_u8=kv_window_u8,
      score_window_u8=score_window_u8,
      global_idx=global_idx,
      num_tokens=cfgs.dims.size_n,
      window=window,
      block_size=cfgs.dims.block_size,
      pages_to_buffer_per_token=cfgs.pages_to_buffer_per_token,
      slots_per_part=cfgs.gather_slots,
      slots_per_token=cfgs.slots_per_token,
      overlap=cfgs.dims.overlap,
  )

  kv_val = window_vmem.at[0][...]      # (tile_n, window, head_tiles, 128)
  scores_val = window_vmem.at[1][...]  # (tile_n, window, head_tiles, 128)
  rms_weight_tiled = rms_weight_vmem_ref[...].astype(jnp.float32)  # (head_tiles, 128)

  # --- windowed softmax ---
  curr_pos = start[:, None] + jnp.arange(window)[None, :]  # (tile_n, window)
  mask = curr_pos >= 0  # (tile_n, window)
  mask_float = mask.astype(scores_val.dtype)
  mask_float_reshaped = mask_float[:, :, None, None]  # (tile_n, window, 1, 1)
  neg_inf = jnp.array(-jnp.inf, dtype=scores_val.dtype)
  masked_scores = jnp.where(mask_float_reshaped > 0.5, scores_val, neg_inf)
  weights = jax.nn.softmax(masked_scores, axis=1)
  compressed = jnp.sum(weights * kv_val, axis=1)  # (tile_n, head_tiles, 128)

  # --- rms norm ---
  variance = jnp.mean(jnp.square(compressed), axis=(1, 2), keepdims=True)
  normed = (
      compressed
      * jax.lax.rsqrt(variance + cfgs.dims.rms_eps)
      * rms_weight_tiled[None, :, :]
  )  # (tile_n, head_tiles, 128)

  # --- rope ---
  rope_ropped = None
  if cfgs.dims.has_rope:
    rope_slot = cfgs.rope_slot
    rope_val = normed[:, rope_slot : rope_slot + 1]

    cos_sin = cos_sin_vmem[...][:, None, :]
    cos_val = cos_sin[:, :, : cfgs.half_rope]
    sin_val = cos_sin[:, :, cfgs.half_rope :]

    rope_ropped = compute.interleaved_rope_vector(rope_val, cos_val, sin_val)
    if head_tiles > 1:
      normed = jnp.concatenate([normed[:, :rope_slot], rope_ropped], axis=1)
    else:
      normed = rope_ropped

  # --- pack + store ---
  if cfgs.dims.is_quantized:
    q, scale = compute.quantize_fp8_tiled(normed, cfgs.dims.quant_block)
    out_vmem[:, 0] = compute.pack_nope_tiled(
        q, scale, cfgs.nope_store_dim, cfgs.dims.quant_block,
        nope_width_bytes=cfgs.record_bytes,
    )
    if cfgs.dims.has_rope_cache:
      _merge_rope_records(
          rope_out_vmem,
          compute.pack_rope_tiled(
              rope_ropped, cfgs.dims.rope_head_dim, cfgs.dims.rope_width
          ),  # (tile_n, 1, 128)
          global_idx,
          kv_slot_mapping_ref,
          token_to_req_indices_ref,
          distribution_ref,
          cfgs,
      )
  else:
    # hca: bitcast bf16 -> uint8 and match the output block shape.
    out_vmem[...] = pltpu.bitcast(
        normed.astype(cfgs.dims.nope_dtype), jnp.uint8
    ).reshape(tile_n, cfgs.record_slots, cfgs.hbm_pack, 128)


def kernel_fn(
    # prefetched inputs (scalar)
    block_table_ref,
    positions_ref,
    token_to_req_indices_ref,
    kv_slot_mapping_ref,
    distribution_ref,
    grid_size_ref,
    rms_weight_ref,
    # HBM inputs (dynamic access)
    cos_sin_cache_ref,
    cache_ref,
    rope_cache_ref,
    # outputs (aliased)
    _out_cache_ref,
    _out_rope_cache_ref,
    window_scratch_ref,
    *,
    cfgs: config.Configs,
):
  """Pallas kernel entry point."""
  grid_size = grid_size_ref[...]
  allocs, in_specs, pipeline_args = buffered_ref.create_allocs_and_specs(
      cfgs=cfgs,
      cache_ref=cache_ref,
      rope_cache_ref=rope_cache_ref,
      cos_sin_cache_ref=cos_sin_cache_ref,
      positions_ref=positions_ref,
      block_table_ref=block_table_ref,
      token_to_req_indices_ref=token_to_req_indices_ref,
      kv_slot_mapping_ref=kv_slot_mapping_ref,
      distribution_ref=distribution_ref,
  )

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(inner_kernel, cfgs=cfgs),
      grid=(grid_size,),
      in_specs=in_specs,
      out_specs=[],
  )

  @pl.with_scoped(allocations=tuple(allocs))
  def _run(allocations):
    pipeline_func(
        *pipeline_args,
        scratches=(
            positions_ref,
            kv_slot_mapping_ref,
            token_to_req_indices_ref,
            distribution_ref,
            rms_weight_ref,
            window_scratch_ref,
        ),
        allocations=allocations,
    )

  _run()


def _select_mode(head_dim: int, overlap: bool) -> config.Mode:
  if head_dim == 128:
    return config.Mode.CSA_INDEXER
  return config.Mode.CSA if overlap else config.Mode.HCA


def derive_aliases(
    has_rope: bool, has_rope_cache: bool, num_scalar_prefetch: int
) -> dict[int, int]:
  cache_index = num_scalar_prefetch + 1 + int(has_rope)
  aliases = {cache_index: 0}
  if has_rope_cache:
    aliases[cache_index + 1] = 1
  return aliases


@functools.partial(
    jax.jit,
    static_argnames=(
        "compress_ratio",
        "overlap",
        "state_block_size",
        "quant_block",
        "rms_eps",
        "rope_store_mode",
        "interpret",
        "name",
    ),
    donate_argnames=("cache", "rope_cache"),
)
def compress_norm_rope_store(
    cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    rms_weight: jax.Array,
    *,
    rope_cache: jax.Array | None = None,
    distribution: jax.Array | None = None,
    cos_sin_cache: jax.Array | None = None,
    compress_ratio: int,
    overlap: bool,
    state_block_size: int,
    quant_block: int = 64,
    rms_eps: float = 1e-6,
    rope_store_mode: str = "decode",
    interpret: bool = False,
    name: str = "compress_norm_rope_store",
) -> tuple[jax.Array, jax.Array | None]:
  """Compresses, normalizes, applies RoPE and stores to cache.

  One invocation handles one class of token, selected by ``rope_store_mode``,
  because the two classes need different rope-cache write patterns; run it
  twice to cover a mixed batch (see compressor.compressor_forward).

  Args:
    cache: [num_pages, page_size, hbm_pack, 128] uint8 state / compressed-KV
      cache, updated in place.
    positions: [num_tokens] logical position of each boundary token.
    block_table: [num_reqs, max_blocks] state-cache page table.
    token_to_req_indices: [num_tokens] request index of each token.
    kv_slot_mapping: [num_tokens] destination slot, negative to skip.
    rms_weight: [head_dim] fp32 norm weight.
    rope_cache: [num_pages, page_size // rope_pack, rope_pack, 128] uint8,
      required when the mode has a separate rope cache (CSA).
    distribution: i32[3] (i, j, k): requests [0:i) are decode-only, [i:j) are
      chunked-prefill-only and [j:k) are mixed, matching the convention in
      mla.py. Required alongside rope_cache -- it says which side of the batch
      this invocation's rope store is for.
    cos_sin_cache: [max_pos, rope_head_dim] fp32 cos/sin table.
    compress_ratio: tokens compressed into one record.
    overlap: whether windows overlap (CSA path).
    state_block_size: tokens per state-cache page.
    quant_block: fp8 quantization block, 0 to disable.
    rms_eps: RMSNorm epsilon.
    rope_store_mode: which tokens' rope records this invocation stores, and how.
      "decode" stores tokens of requests below distribution[0], one packed row
      read-modify-written per token -- valid because decode tokens never share
      a row. "prefill" stores the rest, one packed row per grid step: the batch
      must be laid out so that the tile [4i, 4i+4) holds exactly the tokens of
      one row (compressor._row_tiled_boundary_batch does this), or records will
      overwrite each other.
    interpret: run the Pallas kernel in interpret mode.
    name: kernel name used in profiles.

  Returns:
    The updated (cache, rope_cache); rope_cache is None when unused.
  """
  # TODO(alynie): In HCA, we want to overlay HCA's compressor state cache
  # onto CSA's compressed KV cache (instead of HCA's compressed KV cache)
  # to save memory.
  head_dim = rms_weight.shape[0]
  rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

  # The grid runs whole tiles of tile_n tokens, so a batch that is not a
  # multiple of tile_n would have its last tile index past the end of the
  # per-token arrays -- and bounds checks are disabled, so that reads garbage
  # and can fabricate a store. Pad up to a tile boundary; kv_slot = -1 makes
  # the padding lanes inert. Reachable from small decode batches: 3 tokens
  # with 1 boundary still runs a full 4-wide tile.
  tile_n = 4
  num_tokens = pl.cdiv(positions.shape[0], tile_n) * tile_n
  num_pad = num_tokens - positions.shape[0]
  if num_pad:
    positions = jnp.pad(positions, (0, num_pad))
    token_to_req_indices = jnp.pad(token_to_req_indices, (0, num_pad))
    kv_slot_mapping = jnp.pad(
        kv_slot_mapping, (0, num_pad), constant_values=-1
    )

  cfgs = config.Configs.make(
      _select_mode(head_dim, overlap),
      size_n=num_tokens,
      block_size=state_block_size,
      rms_eps=rms_eps,
      tile_n=tile_n,
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      compress_ratio=compress_ratio,
      quant_block=quant_block,
      rope_store_mode=rope_store_mode,
  )

  if cfgs.dims.has_rope_cache:
    if rope_cache is None:
      raise ValueError(
          "rope_cache must be provided when has_rope_cache is True"
      )
    if distribution is None:
      raise ValueError(
          "distribution must be provided when has_rope_cache is True; the rope"
          " store needs it to tell decode tokens from prefill tokens"
      )
    expected = (cfgs.rope_page_size, cfgs.rope_pack, 128)
    if rope_cache.shape[1:] != expected:
      raise ValueError(
          f"rope_cache must be [num_pages, {expected[0]}, {expected[1]}, 128];"
          f" got {rope_cache.shape}"
      )
    # "prefill" makes one grid step own exactly one packed rope row: the caller
    # lays the batch out at stride rope_pack (compressor.row_tiled_boundary_batch)
    # and the staging buffer is a single row. A wider tile would span two rows
    # and merge the second into the first; a narrower one would split a row over
    # two read-modify-write cycles and lose the earlier records. Both are silent.
    if rope_store_mode == "prefill" and tile_n != cfgs.rope_pack:
      raise ValueError(
          f'rope_store_mode="prefill" requires tile_n == rope_pack; got'
          f" tile_n={tile_n}, rope_pack={cfgs.rope_pack}"
      )

  assert distribution is not None

  rms_weight_reshaped = rms_weight.reshape(cfgs.head_tiles, 128)

  # Tiles up to and including the last token with a destination. Counting the
  # valid tokens instead would be wrong in "prefill" mode, where the row-tiled
  # layout leaves holes wherever a row is only partly covered.
  last_token = jnp.max(
      jnp.where(kv_slot_mapping >= 0, jnp.arange(num_tokens), -1)
  )
  grid_size = pl.cdiv(last_token + 1, cfgs.tile_sizes.tile_n)

  # Outer pallas_call operands, in call order. Optional operands are passed as
  # None to keep kernel_fn's argument positions fixed; pallas drops the Nones
  # when indexing, so alias indices count only the operands actually present.
  scalar_prefetch = (
      block_table,
      positions,
      token_to_req_indices,
      kv_slot_mapping,
      distribution,
      grid_size,
  )
  cos_sin_operand = cos_sin_cache if cfgs.dims.has_rope else None
  rope_operand = rope_cache if cfgs.dims.has_rope_cache else None

  in_specs = (
      pl.BlockSpec(memory_space=pltpu.VMEM),  # rms_weight
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope
          else None
      ),  # cos_sin
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope_cache
          else None
      ),  # rope_cache
  )
  out_specs = (
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope_cache
          else None
      ),  # rope_cache
  )
  out_shapes = (
      jax.ShapeDtypeStruct(cache.shape, cache.dtype),
      jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype)
      if cfgs.dims.has_rope_cache
      else None,
  )

  aliases = derive_aliases(
      cfgs.dims.has_rope, cfgs.dims.has_rope_cache, len(scalar_prefetch)
  )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=len(scalar_prefetch),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=[pltpu.VMEM(cfgs.window_shape(), jnp.float32)],
  )

  out_cache, out_rope_cache = pl.pallas_call(
      functools.partial(kernel_fn, cfgs=cfgs),
      out_shape=out_shapes,
      grid_spec=grid_spec,
      input_output_aliases=aliases,
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
      interpret=interpret,
      name=name,
  )(
      *scalar_prefetch,
      rms_weight_reshaped,
      cos_sin_operand,
      cache,
      rope_operand,
  )

  return out_cache, out_rope_cache