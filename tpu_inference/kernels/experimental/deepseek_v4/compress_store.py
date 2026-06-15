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
"""Saves partial DeepSeek-V4 compressor states into the paged state cache."""

import jax
import jax.numpy as jnp


def save_partial_states(
    kv: jax.Array,            # [num_tokens, state_width] fp32
    score: jax.Array,         # [num_tokens, state_width] fp32
    ape: jax.Array,           # [compress_ratio, state_width] fp32
    positions: jax.Array,     # [num_tokens] int
    state_cache: jax.Array,   # [num_blocks, block_size, 2 * state_width] fp32
    slot_mapping: jax.Array,  # [num_tokens] int
    compress_ratio: int,
) -> jax.Array:
    """Scatter ``[kv | score + ape]`` into ``state_cache``; skip ``slot < 0``."""
    num_blocks, block_size, two_state_width = state_cache.shape
    state_width = two_state_width // 2

    if kv.shape[-1] != state_width or score.shape[-1] != state_width:
        raise ValueError(
            f"kv/score last dim must equal state_width={state_width}; got "
            f"kv={kv.shape}, score={score.shape}, state_cache={state_cache.shape}"
        )

    cache_dtype = state_cache.dtype
    kv = kv.astype(cache_dtype)
    score = score.astype(cache_dtype)
    ape = ape.astype(cache_dtype)

    ape_rows = jnp.mod(positions, compress_ratio)
    score_state = score + ape[ape_rows]
    packed = jnp.concatenate([kv, score_state], axis=-1)

    num_slots = num_blocks * block_size
    flat = state_cache.reshape(num_slots, two_state_width)
    valid = slot_mapping >= 0
    slots = jnp.where(valid, slot_mapping, num_slots)  # OOB sentinel for pads
    flat = flat.at[slots].set(packed, mode="drop")
    return flat.reshape(num_blocks, block_size, two_state_width)