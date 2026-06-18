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
"""Forward pass for the DeepSeek-V4 KV compressor."""

import jax
import jax.numpy as jnp

from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    compress_norm_rope_store, compress_norm_rope_store_indexer,
    pack_state_cache, unpack_state_cache)
from tpu_inference.kernels.experimental.deepseek_v4.compress_store import (
    save_partial_states)


def _project_and_save(
    hidden_states: jax.Array,  # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,      # [2 * coff * head_dim, hidden_size] fp32
    ape: jax.Array,              # [compress_ratio, coff * head_dim] fp32
    positions: jax.Array,        # [num_tokens] int
    state_cache: jax.Array,      # [num_blocks, block_size, 2*coff*head_dim] fp32
    slot_mapping: jax.Array,     # [num_tokens] int
    head_dim: int,
    overlap: bool,
    compress_ratio: int,
) -> jax.Array:
    coff = 1 + int(overlap)
    state_width = coff * head_dim

    # Keep at full precision for testing precision for now.
    kv_score = jnp.matmul(
        hidden_states.astype(jnp.float32),
        wkv_wgate.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    # [num_tokens, 2 * coff * head_dim]
    kv = kv_score[:, :state_width] # [num_tokens, coff * head_dim]
    score = kv_score[:, state_width:2 * state_width] # [num_tokens, coff * head_dim]

    return save_partial_states(
        kv=kv,
        score=score,
        ape=ape,
        positions=positions,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        compress_ratio=compress_ratio,
    )


def compressor_forward(
    hidden_states: jax.Array,       # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,           # [2*coff*head_dim, hidden_size] fp32
    ape: jax.Array,                 # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array,         # [head_dim] fp32 RMSNorm gamma
    cos_sin_cache: jax.Array,       # [max_pos, rope_head_dim] fp32 RoPE table
    positions: jax.Array,           # [num_tokens] int logical pos per token
    slot_mapping: jax.Array,        # [num_tokens] int flat state-cache slot
    block_table: jax.Array,         # [num_reqs, max_blocks] int state pages
    token_to_req_indices: jax.Array,  # [num_tokens] int req id per token
    kv_slot_mapping: jax.Array,     # [num_tokens] int flat compressed-KV slot
    cache: jax.Array,               # [num_pages, page_size//4, 4, width] uint8
    state_block_size: int,          # state tokens per page (4=C4, 8=C128)
    head_dim: int,                  # 512 for sparse CSA/HCA main path
    rope_head_dim: int,             # 64; trailing dims get interleaved RoPE
    compress_ratio: int,            # 4 (CSA) or 128 (HCA); boundary stride
    overlap: bool,                  # True for C4 (two head slices per state row)
    rms_eps: float,
    quant_block: int,                # fp8 absmax block along nope (64)
):
    """head_dim=512 path: project, save state, compress, store into one buffer.

    State cache and compressed KV cache share the same underlying ``cache``
    buffer (see ``compress_norm_rope`` for layout).
    """
    coff = 1 + int(overlap)
    state_dim = 2 * coff * head_dim

    state_view = unpack_state_cache(cache, state_block_size, state_dim)
    state_view = _project_and_save(
        hidden_states, wkv_wgate, ape, positions, state_view, slot_mapping,
        head_dim, overlap, compress_ratio)
    cache = pack_state_cache(cache, state_view)

    cache = compress_norm_rope_store(
        cache=cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )

    return cache


def compressor_forward_indexer(
    hidden_states: jax.Array,       # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,           # [2*coff*head_dim, hidden_size] fp32
    ape: jax.Array,                 # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array,         # [head_dim] fp32 RMSNorm gamma
    cos_sin_cache: jax.Array,       # [max_pos, rope_head_dim] fp32 RoPE table
    positions: jax.Array,           # [num_tokens] int logical pos per token
    slot_mapping: jax.Array,        # [num_tokens] int flat state-cache slot
    block_table: jax.Array,         # [num_reqs, max_blocks] int state pages
    token_to_req_indices: jax.Array,  # [num_tokens] int req id per token
    kv_slot_mapping: jax.Array,     # [num_tokens] int flat indexer-KV slot
    cache: jax.Array,               # [num_pages, page_size//4, 4, width] uint8
    state_block_size: int,          # indexer state tokens per page
    head_dim: int,                  # 128 for the indexer path
    rope_head_dim: int,             # 64; trailing dims get interleaved RoPE
    compress_ratio: int,            # 4 (CSA) or 128 (HCA); boundary stride
    overlap: bool,                  # True for C4 (two head slices per state row)
    rms_eps: float,
    quant_block: int,                # whole-head fp8 absmax block (128)
):
    """head_dim=128 indexer path: same as ``compressor_forward``, head_dim=128."""
    coff = 1 + int(overlap)
    state_dim = 2 * coff * head_dim

    state_view = unpack_state_cache(cache, state_block_size, state_dim)
    state_view = _project_and_save(
        hidden_states, wkv_wgate, ape, positions, state_view, slot_mapping,
        head_dim, overlap, compress_ratio)
    cache = pack_state_cache(cache, state_view)

    cache = compress_norm_rope_store_indexer(
        cache=cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )

    return cache
