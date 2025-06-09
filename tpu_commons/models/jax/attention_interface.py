from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.attention import (sharded_flash_attention,
                                                     sharded_paged_attention,
                                                     update_cache)
from tpu_commons.models.jax.layers.quantized_kvcache import \
    update_cache_quantized

KVCache = Tuple[jax.Array, jax.Array]
QuantizedKVCache = Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]


def attention(
    is_prefill: bool,
    kv_cache: KVCache | QuantizedKVCache,
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
    # k_cache, v_cache = kv_cache
    # TODO (jacobplatin): might be better to make quantized KV cache an actual type
    # to make this a nicer check
    is_quantized = (
        isinstance(kv_cache, tuple)
        and len(kv_cache) == 2
        and isinstance(kv_cache[0], tuple)
    )

    md = attention_metadata
    new_kv_cache = None
    if md.chunked_prefill_enabled:
        raise NotImplementedError("TODO for quantized kv cache")
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
        # TODO (jacobplatin): update kv_cache
    else:
        if is_quantized:
            (k_quant_data, k_scales), (v_quant_data, v_scales) = kv_cache
            quant_dtype = k_quant_data.dtype
            k_quant_data, k_scales = update_cache_quantized(
                is_prefill,
                k_quant_data,
                k_scales,
                quant_dtype,
                md.kv_cache_write_indices,
                k,
            )
            v_quant_data, v_scales = update_cache_quantized(
                is_prefill,
                v_quant_data,
                v_scales,
                quant_dtype,
                md.kv_cache_write_indices,
                v,
            )
            k_cache = quantization_utils.QuantizedTensor(
                k_quant_data, k_scales)
            v_cache = quantization_utils.QuantizedTensor(
                v_quant_data, v_scales)
            new_kv_cache = ((k_quant_data, k_scales), (v_quant_data, v_scales)) if is_quantized else (k_cache, v_cache)
        else:
            k_cache, v_cache = kv_cache
            k_cache = update_cache(is_prefill, k_cache, md.kv_cache_write_indices,
                                k)
            v_cache = update_cache(is_prefill, v_cache, md.kv_cache_write_indices,
                                v)
            new_kv_cache = (k_cache, v_cache)

        if is_prefill:
            # (B, N, T, H)
            # TODO(xiang): support MQA and GQA
            if num_kv_heads != num_heads:
                k = jnp.repeat(k, num_heads // num_kv_heads, axis=1)
                v = jnp.repeat(v, num_heads // num_kv_heads, axis=1)
            outputs = sharded_flash_attention(mesh)(q, k, v)
        else:
            # (B, N, H)
            # NOTE: we can optionally manually dequantize the cache here, but
            # the paged attention kernel nicely takes in quantized tensors
            # k_cache_dequant = dequantize(k_quant_data, quant_dtype, k_scales,
            #                              q.dtype)
            # v_cache_dequant = dequantize(v_quant_data, quant_dtype, v_scales,
            #                              q.dtype)
            q = jnp.squeeze(q, 2)

            outputs = sharded_paged_attention(mesh)(q, k_cache,
                                                    v_cache,
                                                    md.seq_lens,
                                                    md.block_indices)
            # (B, N, 1, H)
            outputs = jnp.expand_dims(outputs, 2)

    return new_kv_cache, outputs
