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
"""DeepSeek V4 Compressor Layer forward pass implementation."""

import jax
import jax.numpy as jnp

import tpu_inference.kernels.experimental.deepseek_v4.compress_and_store.config as config
import tpu_inference.kernels.experimental.deepseek_v4.compress_and_store.kernel as compress_kernel
import tpu_inference.kernels.experimental.deepseek_v4.compress_and_store.proj_and_save_state as proj_kernel


def block_table_row_stride(block_table: jax.Array, num_reqs: int) -> int:
    """Per-request row stride of a flattened [num_reqs * max_blocks] table."""
    assert block_table.ndim == 1
    assert block_table.shape[0] % num_reqs == 0
    return block_table.shape[0] // num_reqs


def derive_metadata(
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    cache: jax.Array,
    compress_ratio: int,
    state_block_size: int,
    head_dim: int,
    overlap: bool,
    cos_sin_cache: jax.Array | None,
    state_cache: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Derives token_to_req_indices, slot_mapping_slots, and kv_slot_mapping.

  The page geometry is taken from ``config.Configs`` keyed on
  ``cache.shape[1]`` (compressed KV) and ``state_cache.shape[1]`` (state),
  which is the same derivation ``compress_norm_rope_store`` and
  ``proj_and_save_state`` use. It must not be recomputed from a separate model
  here: the three only agree for CSA, and diverge by 2x (HCA) and 4x (indexer)
  otherwise, which silently scatters state and compressed-KV writes across the
  wrong pages.

  Args:
    positions: [num_tokens]. Logical position of each token in its request.
    block_table: [num_reqs * max_blocks]. Flattened row-major page table for the
      state cache.
    query_start_loc: [num_reqs + 1]. Cumulative sum of query lengths.
    kv_block_table: [num_reqs * max_kv_blocks]. Flattened row-major page table
      for the compressed KV cache.
    cache: [num_pages, physical_page_size, 4, lanes] uint8. The compressed-KV
      buffer; only its shape is read here.
    compress_ratio: Compression ratio (e.g. 4 for CSA, 128 for HCA).
    state_block_size: Block size vLLM used to build ``block_table``. Must match
      the geometry of ``state_cache``; asserted below.
    head_dim: Dimensionality of attention heads.
    overlap: Whether to use overlap (CSA path).
    cos_sin_cache: [max_pos, rope_head_dim] or None. RoPE cos/sin cache.
    state_cache: [num_pages, state_page_size, 4, lanes] uint8, or None when the
      state shares ``cache``'s buffer. Only its shape is read here.
  Returns:
    token_to_req_indices: [num_tokens]. Request index for each token.
    slot_mapping_slots: [num_tokens]. Physical row index in the state cache.
    kv_slot_mapping: [num_tokens]. Sub-slot index in the compressed KV cache
      (``// tokens_in_second_minor`` gives the physical row).
  """
    num_tokens = positions.shape[0]
    rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0
    state_source = cache if state_cache is None else state_cache

    cfgs = config.Configs.make(
        config.select_mode(head_dim, overlap),
        size_n=num_tokens,
        physical_page_size=cache.shape[1],
        state_physical_page_size=state_source.shape[1],
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
    )
    # Rows per page, rows per token's state, and the KV sub-slot packing.
    page_size = cfgs.state_physical_page_size
    state_rows_per_token = cfgs.state_rows_per_token
    kv_block_size = cfgs.kv_block_size
    kv_stride = cfgs.kv_stride
    kv_page_stride = kv_block_size * kv_stride

    # The block tables are paged at vLLM's state-cache block size, so it has to
    # be the number of token states that actually fit in one physical page.
    assert state_block_size == cfgs.state_block_size, (
        f"state cache block_size {state_block_size} does not match the "
        f"{cfgs.dims.mode.value} state cache geometry {state_source.shape}, "
        f"which holds {cfgs.state_block_size} token states per page ")

    # 2. Map tokens to request indices (Handles Ragged Batch)
    query_lens = jnp.diff(query_start_loc)
    batch_size = query_start_loc.shape[0] - 1
    token_to_req_indices = jnp.repeat(jnp.arange(batch_size),
                                      query_lens,
                                      total_repeat_length=num_tokens)
    req = token_to_req_indices

    block_table_stride = block_table_row_stride(block_table, batch_size)
    kv_block_table_stride = block_table_row_stride(kv_block_table, batch_size)

    # 3. State Cache Slot Mapping (Virtual -> Physical)
    state_page_idx = positions // state_block_size
    state_page_offset = positions % state_block_size
    state_page_numbers = block_table[req * block_table_stride + state_page_idx]
    slot_mapping_slots = (state_page_numbers * page_size +
                          state_page_offset * state_rows_per_token)

    # 4. Compressed KV Cache Slot Mapping (Virtual -> Physical)
    kv_idx = positions // compress_ratio
    kv_page_idx = kv_idx // kv_block_size
    kv_page_offset = kv_idx % kv_block_size

    kv_page_number = kv_block_table[req * kv_block_table_stride + kv_page_idx]

    kv_slot_mapping = (kv_page_number * kv_page_stride +
                       kv_page_offset * kv_stride)

    is_boundary = ((positions + 1) % compress_ratio) == 0
    kv_slot_mapping = jnp.where(is_boundary, kv_slot_mapping, -1)

    return token_to_req_indices, slot_mapping_slots, kv_slot_mapping


def _get_packing_indices(keep, size):
    order = jnp.nonzero(keep, size=size, fill_value=-1)[0]
    valid = order >= 0
    safe = jnp.where(valid, order, 0)
    return safe, valid


def _pack_array(x, safe, valid, default_value):
    return jnp.where(valid, x[safe], default_value)


def prepare_boundary_batch(
    positions: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    is_boundary: jax.Array,
    is_decode_token: jax.Array,
    compress_ratio: int,
    num_reqs: int,
    rope_pack: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generates a combined batch of boundary tokens in a single scatter.

  Layout: [ decode_packed (0..K-1) | pad (K..K_aligned-1) |
  prefill_row_tiled (K_aligned..) | pad ]
  """
    num_tokens = positions.shape[0]
    max_prefill_boundary = num_tokens // compress_ratio + 1 + num_reqs
    packed_size = num_tokens + max_prefill_boundary

    # 1. Get packing indices for all boundary tokens
    safe, valid = _get_packing_indices(is_boundary, packed_size)

    # 2. Pack inputs and the decode mask
    pos = _pack_array(positions, safe, valid, 0)
    req = _pack_array(token_to_req_indices, safe, valid, 0)
    kv = _pack_array(kv_slot_mapping, safe, valid, -1)
    is_decode = _pack_array(is_decode_token, safe, valid, False)

    valid_token = kv >= 0
    is_prefill = valid_token & ~is_decode

    # 3. Calculate row-tiling ONLY for prefill tokens
    row = jnp.where(is_prefill, kv // rope_pack, -1)
    row_shifted = jnp.roll(row, 1)
    starts_row = (row != row_shifted) & is_prefill
    starts_row = starts_row.at[0].set(is_prefill[0])

    tile_index = jnp.cumsum(starts_row.astype(jnp.int32)) - 1

    index = jnp.arange(packed_size)
    run_start = jax.lax.cummax(jnp.where(starts_row, index, 0))
    pos_within_tile = index - run_start

    # 4. Calculate alignment offset for prefill block
    K = jnp.sum(is_decode)
    K_aligned = ((K + 3) // 4) * 4

    # 5. Combine destination indices
    prefill_size = rope_pack * max_prefill_boundary
    output_size = num_tokens + prefill_size

    dest_prefill = K_aligned + tile_index * rope_pack + pos_within_tile
    dest = jnp.where(is_decode, index, dest_prefill)
    dest = jnp.where(valid_token, dest, output_size)

    # 6. Scatter to output once
    def scatter(default_val, src):
        return jnp.full((output_size, ), default_val,
                        dtype=src.dtype).at[dest].set(src, mode="drop")

    return (
        scatter(0, pos),
        scatter(0, req),
        scatter(-1, kv),
    )


def compressor_forward(
    hidden_states: jax.Array,  # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,  # [2*coff*head_dim, hidden_size] fp32
    ape: jax.Array,  # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    positions: jax.Array,  # [num_tokens] int
    block_table: jax.Array,  # [num_reqs * max_blocks] int
    query_start_loc: jax.Array,  # [num_reqs + 1] int
    kv_block_table: jax.Array,  # [num_reqs * max_kv_blocks] int
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8
    rope_cache: jax.Array,  # [num_pages, page_size // 4, 4, 128] uint8
    distribution: jax.Array,  # i32[3]
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    state_cache: jax.Array
    | None = None,  # [num_pages, state_page_size, 4, 128] uint8
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projects, saves state, then compresses and stores the boundary records.
    """
    num_tokens = positions.shape[0]
    num_reqs = query_start_loc.shape[0] - 1
    shares_buffer = state_cache is None

    token_to_req_indices, slot_mapping_slots, kv_slot_mapping = derive_metadata(
        positions=positions,
        block_table=block_table,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table,
        cache=cache,
        compress_ratio=compress_ratio,
        state_block_size=state_block_size,
        head_dim=head_dim,
        overlap=overlap,
        cos_sin_cache=cos_sin_cache,
        state_cache=state_cache,
    )

    token_index = jnp.arange(num_tokens)
    is_real_token = token_index < query_start_loc[distribution[2]]
    slot_mapping_slots = jnp.where(is_real_token, slot_mapping_slots, -1)

    updated_state = proj_kernel.proj_and_save_state(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=positions,
        slot_mapping=slot_mapping_slots,
        cache=cache if shares_buffer else state_cache,
        compress_ratio=compress_ratio,
    )
    if shares_buffer:
        cache = updated_state
    else:
        state_cache = updated_state

    is_decode_token = token_index < query_start_loc[distribution[0]]
    is_boundary = (kv_slot_mapping >= 0) & is_real_token

    boundary_positions, boundary_req, boundary_kv_slot = prepare_boundary_batch(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        is_boundary=is_boundary,
        is_decode_token=is_decode_token,
        compress_ratio=compress_ratio,
        num_reqs=num_reqs,
    )

    cache, rope_cache = compress_kernel.compress_norm_rope_store(
        cache,
        boundary_positions,
        block_table,
        boundary_req,
        boundary_kv_slot,
        norm_weight,
        block_table_stride=block_table_row_stride(block_table, num_reqs),
        state_cache=state_cache,
        rope_cache=rope_cache,
        cos_sin_cache=cos_sin_cache,
        compress_ratio=compress_ratio,
        overlap=overlap,
        quant_block=quant_block,
        rms_eps=rms_eps,
    )

    if shares_buffer:
        return cache, rope_cache, cache

    # `compress_norm_rope_store` READS `state_cache` (its read-only state
    # operand) but emits no new version of it, so the value published here is
    # the pre-read one straight off `proj_and_save_state`. The DSv4 KV-cache
    # overlay hosts another layer's SWA cache on this same physical array, and
    # that kernel takes whole-buffer ownership via `input_output_aliases`.
    # Nothing orders the read against that write, so XLA's copy-insertion
    # AddCopiesToResolveInterference cannot prove the read completes first and
    # insert the cache copy. Publishing a version that is downstream of
    # the read supplies the missing ordering edge.
    state_cache, cache = jax.lax.optimization_barrier((state_cache, cache))
    return cache, rope_cache, state_cache
