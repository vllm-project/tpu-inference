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

import jax
import jax.numpy as jnp


def _pack_kv_for_cache(
    keys: jax.Array,
    values: jax.Array,
    cache_shape: tuple[int, ...],
) -> jax.Array:
    if keys.shape != values.shape:
        raise ValueError("keys and values must have identical shapes")
    if keys.ndim != 3 or len(cache_shape) != 5:
        raise ValueError("expected token-major K/V and a five-dimensional cache")

    num_tokens, num_kv_heads, actual_head_dim = keys.shape
    cache_head_groups, packing, cache_head_dim = cache_shape[-3:]
    padded_kv_heads_x2 = cache_head_groups * packing
    actual_kv_heads_x2 = num_kv_heads * 2
    if actual_kv_heads_x2 > padded_kv_heads_x2:
        raise ValueError("K/V heads do not fit the cache layout")
    if actual_head_dim > cache_head_dim:
        raise ValueError("K/V head dimension does not fit the cache layout")

    merged = jnp.concatenate([keys, values], axis=-1).reshape(
        num_tokens, actual_kv_heads_x2, actual_head_dim)
    merged = jnp.pad(
        merged,
        (
            (0, 0),
            (0, padded_kv_heads_x2 - actual_kv_heads_x2),
            (0, cache_head_dim - actual_head_dim),
        ),
    )
    return merged.reshape(num_tokens, cache_head_groups, packing,
                          cache_head_dim)


def replace_cached_kv(
    kv_cache: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    input_positions: jax.Array,
    block_tables: jax.Array,
    active_rows: jax.Array,
) -> jax.Array:
    """Overwrite arbitrary logical token positions in a paged K/V cache."""
    if active_rows.ndim != 1:
        raise ValueError("active_rows must be one-dimensional")
    batch_size = active_rows.shape[0]
    if keys.shape[0] % batch_size:
        raise ValueError("K/V token count must be divisible by the batch size")
    if input_positions.shape != (keys.shape[0], ):
        raise ValueError("input_positions must contain one entry per K/V token")
    if block_tables.size % batch_size:
        raise ValueError("block_tables must contain equally sized request rows")

    tokens_per_row = keys.shape[0] // batch_size
    pages_per_row = block_tables.size // batch_size
    page_size = kv_cache.shape[1]
    logical_tables = block_tables.reshape(batch_size, pages_per_row)
    row_indices = jnp.repeat(jnp.arange(batch_size), tokens_per_row)
    logical_pages = input_positions // page_size
    bounded_logical_pages = jnp.clip(logical_pages, 0, pages_per_row - 1)
    physical_pages = logical_tables[row_indices, bounded_logical_pages]
    token_is_active = jnp.repeat(active_rows.astype(bool), tokens_per_row)
    token_is_valid = (
        token_is_active
        & (input_positions >= 0)
        & (logical_pages >= 0)
        & (logical_pages < pages_per_row)
        & (physical_pages >= 0)
        & (physical_pages < kv_cache.shape[0])
    )
    dropped_page = jnp.asarray(kv_cache.shape[0], dtype=physical_pages.dtype)
    physical_pages = jnp.where(token_is_valid, physical_pages, dropped_page)
    page_offsets = jnp.mod(input_positions, page_size)
    packed_kv = _pack_kv_for_cache(keys, values, kv_cache.shape)
    return kv_cache.at[physical_pages, page_offsets].set(packed_kv,
                                                         mode="drop")
