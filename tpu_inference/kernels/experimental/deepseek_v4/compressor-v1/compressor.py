"""DeepSeek V4 Compressor Layer forward pass implementation."""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
  sys.path.insert(0, _DIR)

import jax
import jax.numpy as jnp

import kernel as compress_kernel
import project_and_save_kernel as proj_kernel


def derive_metadata(
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    compress_ratio: int,
    state_block_size: int,
    head_dim: int,
    overlap: bool,
    cos_sin_cache: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Derives token_to_req_indices, slot_mapping_slots, and kv_slot_mapping.

  Args:
    positions: [num_tokens]. Logical position of each token in its request.
    block_table: [num_reqs, max_blocks]. Page table for the state cache.
    query_start_loc: [num_reqs + 1]. Cumulative sum of query lengths.
    kv_block_table: [num_reqs, max_kv_blocks]. Page table for the compressed KV
      cache.
    compress_ratio: Compression ratio (e.g. 4 for CSA, 128 for HCA).
    state_block_size: Block size of the state cache.
    head_dim: Dimensionality of attention heads.
    overlap: Whether to use overlap (CSA path).
    cos_sin_cache: [max_pos, rope_head_dim] or None. RoPE cos/sin cache.

  Returns:
    token_to_req_indices: [num_tokens]. Request index for each token.
    slot_mapping_slots: [num_tokens]. Physical slot index in state cache.
    kv_slot_mapping: [num_tokens]. Physical slot index in compressed KV cache.
  """
  num_tokens = positions.shape[0]
  rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

  # 1. Inline Layout Calculations (Derived from geometry)
  coff = 1 + int(overlap)
  state_width = coff * head_dim
  state_dim = 2 * state_width

  # Determine if quantized (CSA/Indexer use FP8, HCA uses BF16)
  is_quantized = (rope_head_dim > 0 and overlap) or (rope_head_dim == 0)
  if not is_quantized:
    total_bytes_out = head_dim * 2  # HCA (bf16)
  else:
    total_bytes_out = 256 if head_dim == 128 else 512  # CSA (fp8)

  total_sub_slots = total_bytes_out // 128
  slots_per_part_hbm = min(total_sub_slots, 4)
  slots_per_part_out = (total_sub_slots + 3) // 4

  # Calculate slots per token and page size in slots
  slots_per_token = (state_dim * 4) // (slots_per_part_hbm * 128)
  page_size = state_block_size * slots_per_token

  # 2. Map tokens to request indices (Handles Ragged Batch)
  query_lens = jnp.diff(query_start_loc)
  batch_size = query_start_loc.shape[0] - 1
  token_to_req_indices = jnp.repeat(
      jnp.arange(batch_size), query_lens, total_repeat_length=num_tokens
  )
  req = token_to_req_indices

  # 3. State Cache Slot Mapping (Virtual -> Physical)
  state_page_idx = positions // state_block_size
  state_page_offset = positions % state_block_size
  state_page_numbers = block_table[req, state_page_idx]
  slot_mapping_slots = (
      state_page_numbers * page_size + state_page_offset * slots_per_token
  )

  # 4. Compressed KV Cache Slot Mapping (Virtual -> Physical)
  kv_idx = positions // compress_ratio
  kv_page_size = page_size // slots_per_part_out
  kv_page_idx = kv_idx // kv_page_size
  kv_page_offset = kv_idx % kv_page_size

  kv_page_number = kv_block_table[req, kv_page_idx]

  kv_slot_mapping = (
      kv_page_number * page_size + kv_page_offset * slots_per_part_out
  )

  is_boundary = ((positions + 1) % compress_ratio) == 0
  kv_slot_mapping = jnp.where(is_boundary, kv_slot_mapping, -1)

  return token_to_req_indices, slot_mapping_slots, kv_slot_mapping


def pack_boundary_batch(
    positions: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    keep: jax.Array,
    size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Moves the kept tokens to the front of a `size`-long batch.

  The kernel's grid covers whole tiles up to the last token with a destination,
  so anything left out here is simply never visited.

  Args:
    positions: [num_tokens] logical position of each token.
    token_to_req_indices: [num_tokens] request index of each token.
    kv_slot_mapping: [num_tokens] destination slot, negative for "skip".
    keep: [num_tokens] bool, which tokens this batch is for.
    size: length of the returned arrays; must be >= the number kept.

  Returns:
    The (positions, token_to_req_indices, kv_slot_mapping) triple, compacted.
  """
  order = jnp.nonzero(keep, size=size, fill_value=-1)[0]
  valid = order >= 0
  safe = jnp.where(valid, order, 0)
  return (
      jnp.where(valid, positions[safe], 0),
      jnp.where(valid, token_to_req_indices[safe], 0),
      jnp.where(valid, kv_slot_mapping[safe], -1),
  )


def row_tiled_boundary_batch(
    positions: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    keep: jax.Array,
    max_boundary: int,
    rope_pack: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Lays the kept tokens out one packed rope row per `rope_pack`-wide tile.

  The rope cache packs `rope_pack` records into each row, and a row is the
  smallest thing that can be exchanged with HBM, so tokens sharing a row have
  to be merged by a single grid step -- otherwise each of them writes its own
  copy of the row back and only the last one survives.

  Tokens sharing a row are always adjacent in token order (kv slots increase
  within a request, pages hold a whole number of rows, and a page belongs to
  one request), so numbering the runs of equal rows gives each row its own
  tile. Within the tile, a token sits at its own sub-slot, leaving holes
  wherever a row is only partly covered by this batch.

  Args:
    positions: [num_tokens] logical position of each token.
    token_to_req_indices: [num_tokens] request index of each token.
    kv_slot_mapping: [num_tokens] destination slot, negative for "skip".
    keep: [num_tokens] bool, which tokens this batch is for.
    max_boundary: upper bound on the number of kept tokens.
    rope_pack: records packed into one rope-cache row (also the tile width).

  Returns:
    The (positions, token_to_req_indices, kv_slot_mapping) triple, row-tiled.
  """
  pos, req, kv = pack_boundary_batch(
      positions, token_to_req_indices, kv_slot_mapping, keep, max_boundary
  )

  valid = kv >= 0
  row = jnp.where(valid, kv // rope_pack, -1)
  starts_row = jnp.concatenate([valid[:1], valid[1:] & (row[1:] != row[:-1])])
  tile_index = jnp.cumsum(starts_row.astype(jnp.int32)) - 1

  # Position within the tile: how far into its run of equal rows a token is.
  # The kernel reads each lane's sub-slot out of kv_slot_mapping, so the lane
  # only has to be distinct within the tile, not equal to the sub-slot.
  index = jnp.arange(max_boundary)
  run_start = jax.lax.cummax(jnp.where(starts_row, index, 0))
  pos_within_tile = index - run_start

  size = rope_pack * max_boundary
  # Invalid entries are scattered out of range, i.e. dropped.
  dest = jnp.where(valid, tile_index * rope_pack + pos_within_tile, size)
  scatter = lambda base, src: base.at[dest].set(src, mode="drop")
  return (
      scatter(jnp.zeros((size,), pos.dtype), pos),
      scatter(jnp.zeros((size,), req.dtype), req),
      scatter(jnp.full((size,), -1, kv.dtype), kv),
  )


def compressor_forward(
    hidden_states: jax.Array,  # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,  # [2*coff*head_dim, hidden_size] fp32
    ape: jax.Array,  # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    positions: jax.Array,  # [num_tokens] int
    block_table: jax.Array,  # [num_reqs, max_blocks] int
    query_start_loc: jax.Array,  # [num_reqs + 1] int
    kv_block_table: jax.Array,  # [num_reqs, max_kv_blocks] int
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8
    rope_cache: jax.Array,  # [num_pages, page_size // 4, 4, 128] uint8
    distribution: jax.Array,  # i32[3]
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> tuple[jax.Array, jax.Array]:
  """Projects, saves state, then compresses and stores the boundary records.

  `distribution` is i32[3] (i, j, k): requests [0:i) are decode-only, [i:j) are
  chunked-prefill-only and [j:k) are mixed, matching mla.py.

  The compress-and-store half runs twice, once per class of token, because the
  packed rope cache wants a different write pattern for each: decode tokens are
  sparse and get a read-modify-write per token, prefill tokens are dense and
  get one merged write per shared row.
  """
  num_tokens = positions.shape[0]
  num_reqs = query_start_loc.shape[0] - 1

  token_to_req_indices, slot_mapping_slots, kv_slot_mapping = derive_metadata(
      positions=positions,
      block_table=block_table,
      query_start_loc=query_start_loc,
      kv_block_table=kv_block_table,
      compress_ratio=compress_ratio,
      state_block_size=state_block_size,
      head_dim=head_dim,
      overlap=overlap,
      cos_sin_cache=cos_sin_cache,
  )

  # Requests [distribution[2], num_reqs) carry no tokens, so only
  # [0, query_start_loc[distribution[2]]) of the batch is real; past that the
  # caller's buffers hold whatever the previous step left there. derive_metadata
  # maps every token unconditionally, so those stale positions still resolve to
  # a page (of the last, padded request) and a slot -- both kernels take a
  # negative slot as "skip", so mask them out here rather than trusting the
  # padding to be benign.
  token_index = jnp.arange(num_tokens)
  is_real_token = token_index < query_start_loc[distribution[2]]
  slot_mapping_slots = jnp.where(is_real_token, slot_mapping_slots, -1)

  cache = proj_kernel.proj_and_save_state(
      hidden_states=hidden_states,
      wkv_wgate=wkv_wgate,
      ape=ape,
      positions=positions,
      slot_mapping=slot_mapping_slots,
      cache=cache,
      compress_ratio=compress_ratio,
  )

  # Split the boundary tokens by class. `distribution` says requests
  # [0, distribution[0]) are decode-only, and tokens are grouped by request in
  # request order, so the decode tokens are exactly the prefix
  # [0, query_start_loc[distribution[0]]). `kv_slot_mapping >= 0` is what
  # derive_metadata leaves set on boundary tokens.
  #
  # The decode half is bounded above by query_start_loc[distribution[0]] <=
  # query_start_loc[distribution[2]], but the prefill half is everything else,
  # so without is_real_token a stale position satisfying
  # (pos + 1) % compress_ratio == 0 would enter it as a boundary token.
  is_decode_token = token_index < query_start_loc[distribution[0]]
  is_boundary = (kv_slot_mapping >= 0) & is_real_token

  # Each request contributes at most ceil(query_len / compress_ratio) boundary
  # tokens, so this bounds their total. The row-tiled layout spreads them over
  # at most that many rows, hence rope_pack times as many slots.
  rope_pack = 4
  max_boundary = num_tokens // compress_ratio + 1 + num_reqs

  decode_batch = pack_boundary_batch(
      positions,
      token_to_req_indices,
      kv_slot_mapping,
      keep=is_boundary & is_decode_token,
      size=num_tokens,
  )
  prefill_batch = row_tiled_boundary_batch(
      positions,
      token_to_req_indices,
      kv_slot_mapping,
      keep=is_boundary & ~is_decode_token,
      max_boundary=max_boundary,
      rope_pack=rope_pack,
  )

  for (b_positions, b_req, b_kv_slot), rope_store_mode in (
      (decode_batch, "decode"),
      (prefill_batch, "prefill"),
  ):
    cache, rope_cache = compress_kernel.compress_norm_rope_store(
        cache,
        b_positions,
        block_table,
        b_req,
        b_kv_slot,
        norm_weight,
        rope_cache=rope_cache,
        distribution=distribution,
        cos_sin_cache=cos_sin_cache,
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_block_size=state_block_size,
        quant_block=quant_block,
        rms_eps=rms_eps,
        rope_store_mode=rope_store_mode,
    )

  return cache, rope_cache