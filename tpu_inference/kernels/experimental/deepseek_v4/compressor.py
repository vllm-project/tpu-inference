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
    compress_norm_rope_store, compress_norm_rope_store_indexer)
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

    kv_score = hidden_states.astype(jnp.float32) @ wkv_wgate.T
    kv = kv_score[:, :state_width]
    score = kv_score[:, state_width:2 * state_width]

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
    wkv_wgate: jax.Array,           # [2 * coff * head_dim, hidden_size] fp32
    ape: jax.Array,                 # [compress_ratio, coff * head_dim] fp32
    norm_weight: jax.Array,         # [head_dim] fp32
    cos_sin_cache: jax.Array,       # [max_pos, rope_head_dim] fp32
    positions: jax.Array,           # [num_tokens] int
    slot_mapping: jax.Array,        # [num_tokens] int
    block_table: jax.Array,         # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,     # [num_tokens] int
    state_cache: jax.Array,         # [num_blocks, block_size, 2*coff*head_dim] fp32
    kv_cache: jax.Array,            # [kv_blocks, kv_block_size, packed_width] uint8
    block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """head_dim=512 path: fp8 nope + bf16 rope packed into one uint8 KV cache."""
    state_cache = _project_and_save(
        hidden_states, wkv_wgate, ape, positions, state_cache, slot_mapping,
        head_dim, overlap, compress_ratio)

    kv_cache = compress_norm_rope_store(
        state_cache=state_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache=kv_cache,
        rms_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        block_size=block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )

    return state_cache, kv_cache


def compressor_forward_indexer(
    hidden_states: jax.Array,       # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,           # [2 * coff * head_dim, hidden_size] fp32
    ape: jax.Array,                 # [compress_ratio, coff * head_dim] fp32
    norm_weight: jax.Array,         # [head_dim] fp32
    cos_sin_cache: jax.Array,       # [max_pos, rope_head_dim] fp32
    positions: jax.Array,           # [num_tokens] int
    slot_mapping: jax.Array,        # [num_tokens] int
    block_table: jax.Array,         # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,     # [num_tokens] int
    state_cache: jax.Array,         # [num_blocks, block_size, 2*coff*head_dim] fp32
    kv_cache: jax.Array,            # [kv_blocks, kv_block_size, packed_width] uint8
    block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """head_dim=128 indexer path: whole-head fp8 + one scale, one uint8 cache."""
    state_cache = _project_and_save(
        hidden_states, wkv_wgate, ape, positions, state_cache, slot_mapping,
        head_dim, overlap, compress_ratio)

    kv_cache = compress_norm_rope_store_indexer(
        state_cache=state_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache=kv_cache,
        rms_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        block_size=block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )

    return state_cache, kv_cache
