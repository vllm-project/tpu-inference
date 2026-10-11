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
"""Copy-on-Write (CoW) block copying utilities for TPU KV caches.

When Decode Context Parallelism (DCP) prefix caching is enabled with fine-grained
cache hits (e.g. 256-token granularity inside 1024-token logical attention blocks),
the vLLM scheduler emits `kv_cache_block_copies = [(src_block_id, dst_block_id), ...]`.
These utilities apply the block copies in device memory using JAX JIT-compiled
tensor updates with buffer donation and power-of-2 bucket padding to avoid
recompilation overhead.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Sequence

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Power-of-2 bucket sizes to avoid repeated JAX JIT recompilation across steps
BUCKET_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


@functools.partial(jax.jit, donate_argnums=(0, ))
def jitted_copy_kv_cache_blocks(
    cache: jax.Array,
    src_indices: jax.Array,
    dst_indices: jax.Array,
) -> jax.Array:
    """Copies cache blocks from src_indices to dst_indices in-place using buffer donation."""
    return cache.at[dst_indices].set(cache[src_indices])


def apply_dense_kv_cache_block_copies(
    kv_caches: list[Any],
    block_copies: Sequence[Any],
) -> int:
    """Applies Copy-on-Write (CoW) block copies for partial prefix-cache hits.

    When Decode Context Parallelism (DCP) prefix caching produces sub-block hits
    (e.g. 256-token granularity inside 1024-token attention blocks), the scheduler
    allocates a new private block and emits kv_cache_block_copies = [(src_block_id, dst_block_id)].
    This function copies the existing KV content from src into dst across all attention layers
    on TPU device memory before the forward pass executes.

    Args:
        kv_caches: List of layer KV cache objects. Attention layers are jax.Array,
            while hybrid/Mamba layers are tuples of state tensors.
        block_copies: Sequence of KVCacheBlockCopy objects or (src, dst) tuples.

    Returns:
        The number of applied copies.
    """
    if not block_copies or not kv_caches:
        return 0

    # Find all attention cache layer indices (Mamba layers are tuples of states)
    attn_indices = [
        i for i, c in enumerate(kv_caches) if isinstance(c, jax.Array)
    ]
    if not attn_indices:
        return 0

    num_copies = len(block_copies)
    bucket_size = next((b for b in BUCKET_SIZES if b >= num_copies),
                       num_copies)

    src_list = [int(getattr(c, "src_block_id", c[0])) for c in block_copies]
    dst_list = [int(getattr(c, "dst_block_id", c[1])) for c in block_copies]

    pad_len = bucket_size - num_copies
    if pad_len > 0:
        # Pad with idempotent self-copies (0 -> 0)
        src_list.extend([0] * pad_len)
        dst_list.extend([0] * pad_len)

    src_arr = jnp.array(src_list, dtype=jnp.int32)
    dst_arr = jnp.array(dst_list, dtype=jnp.int32)

    for idx in attn_indices:
        kv_caches[idx] = jitted_copy_kv_cache_blocks(kv_caches[idx], src_arr,
                                                     dst_arr)

    return num_copies
