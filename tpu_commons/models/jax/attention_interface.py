from typing import Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.attention import (sharded_flash_attention,
                                                     sharded_paged_attention,
                                                     update_cache)
from tpu_commons.models.jax.layers.chunked_prefill_attention import (
    sharded_chunked_prefill_attention, sharded_chunked_prefill_update_cache)

KVCache = Tuple[jax.Array, jax.Array]


def attention(
    is_prefill: bool,
    kv_cache: KVCache,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    num_heads: int,
    num_kv_heads: int,
) -> Tuple[KVCache, jax.Array]:
    # B: batch_size
    # T: seq_len
    # N: num_heads
    # K: num_kv_heads
    # D: hidden_size
    # H: head_dim
    # L: num_blocks
    # S: block_size

    # (K, L, S, H)
    k_cache, v_cache = kv_cache
    md = attention_metadata
    if md.chunked_prefill_enabled:
        k_cache = sharded_chunked_prefill_update_cache(mesh)(
            k_cache, md.kv_cache_write_indices, k, md.num_decode_seqs)
        v_cache = sharded_chunked_prefill_update_cache(mesh)(
            v_cache, md.kv_cache_write_indices, v, md.num_decode_seqs)
        outputs = sharded_chunked_prefill_attention(mesh)(
            q,
            k_cache,
            v_cache,
            attention_metadata.decode_lengths,
            attention_metadata.decode_page_indices,
            attention_metadata.num_decode_seqs,
            attention_metadata.prefill_lengths,
            attention_metadata.prefill_page_indices,
            attention_metadata.prefill_query_start_offsets,
            attention_metadata.num_prefill_seqs,
        )
    else:
        k_cache = update_cache(is_prefill, k_cache, md.kv_cache_write_indices,
                               k)
        v_cache = update_cache(is_prefill, v_cache, md.kv_cache_write_indices,
                               v)
        if is_prefill:
            # (B, N, T, H)
            # TODO(xiang): support MQA and GQA
            if num_kv_heads != num_heads:
                k = jnp.repeat(k, num_heads // num_kv_heads, axis=1)
                v = jnp.repeat(v, num_heads // num_kv_heads, axis=1)
            outputs = sharded_flash_attention(mesh)(q, k, v)
        else:
            # (B, N, H)
            q = jnp.squeeze(q, 2)
            outputs = sharded_paged_attention(mesh)(q, k_cache, v_cache,
                                                    md.seq_lens,
                                                    md.block_indices)
            # (B, N, 1, H)
            outputs = jnp.expand_dims(outputs, 2)

    new_kv_cache = (k_cache, v_cache)

    return new_kv_cache, outputs
