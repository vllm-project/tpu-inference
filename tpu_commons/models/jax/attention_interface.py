from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.attention import (sharded_flash_attention,
                                                     sharded_paged_attention,
                                                     update_cache)
from tpu_commons.models.jax.layers.chunked_prefill_attention import (
    sharded_chunked_prefill_attention, sharded_chunked_prefill_update_cache)
from tpu_commons.models.jax.layers.quantized_kvcache import (
    chunked_prefill_update_cache_quantized, dequantize, update_cache_quantized)

KVCache = Tuple[jax.Array, jax.Array]
QuantizedKVCache = Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array,
                                                            jax.Array]]


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

    # TODO (jacobplatin): might be better to make quantized KV cache an actual type
    # to make this a nicer check
    is_quantized = (isinstance(kv_cache, tuple) and len(kv_cache) == 2
                    and isinstance(kv_cache[0], tuple))

    md = attention_metadata
    new_kv_cache = None
    if md.chunked_prefill_enabled:
        if attention_metadata.page_aligned_update:
            # The e2e performance improvement of this was ~5% back in 2024/03 per gxd@. This might have changed now.
            # k_cache = sharded_chunked_prefill_update_cache(mesh)(
            #     k_cache, md.kv_cache_write_indices, k, md.num_decode_seqs)
            # v_cache = sharded_chunked_prefill_update_cache(mesh)(
            #     v_cache, md.kv_cache_write_indices, v, md.num_decode_seqs)
            if is_quantized:
                (k_quant_data, k_scales), (v_quant_data, v_scales) = kv_cache
                quant_dtype = k_quant_data.dtype
                k_quant_data, k_scales = chunked_prefill_update_cache_quantized(
                    k_quant_data, k_scales, quant_dtype,
                    md.kv_cache_write_indices, k, md.num_decode_seqs)
                v_quant_data, v_scales = chunked_prefill_update_cache_quantized(
                    v_quant_data, v_scales, quant_dtype,
                    md.kv_cache_write_indices, v, md.num_decode_seqs)
                # TODO
                # k_cache = quantization_utils.QuantizedTensor(
                #     k_quant_data, k_scales)
                # v_cache = quantization_utils.QuantizedTensor(
                #     v_quant_data, v_scales)
                new_kv_cache = ((k_quant_data, k_scales), (v_quant_data,
                                                           v_scales))
                k_cache = dequantize(k_quant_data, quant_dtype, k_scales,
                                     jnp.bfloat16)
                v_cache = dequantize(v_quant_data, quant_dtype, v_scales,
                                     jnp.bfloat16)
            else:
                k_cache = sharded_chunked_prefill_update_cache(mesh)(
                    k_cache, md.kv_cache_write_indices, k, md.num_decode_seqs)
                v_cache = sharded_chunked_prefill_update_cache(mesh)(
                    v_cache, md.kv_cache_write_indices, v, md.num_decode_seqs)
        else:
            # TODO(pooyam): Try the following ideas to optimize this.
            # Idea 1: use lax loop similar to to chunked_prefill_update_cache above.
            # Idea 2: Convert a prefill of size m*PAGE_SIZE + n, to a prefill of size m*PAGE_SIZE AND n decodes (But in fact they are all the same seq.)
            # This will however make the code much more complicated.
            k = k.swapaxes(0, 2)
            v = v.swapaxes(0, 2)
            if is_quantized:
                (k_quant_data, k_scales), (v_quant_data, v_scales) = kv_cache
                quant_dtype = k_quant_data.dtype
                k_quant_data, k_scales = update_cache_quantized(
                    False,
                    k_quant_data,
                    k_scales,
                    quant_dtype,
                    md.kv_cache_write_indices,
                    k,
                )
                v_quant_data, v_scales = update_cache_quantized(
                    False,
                    v_quant_data,
                    v_scales,
                    quant_dtype,
                    md.kv_cache_write_indices,
                    v,
                )
                # k_cache = quantization_utils.QuantizedTensor(
                #     k_quant_data, k_scales)
                # v_cache = quantization_utils.QuantizedTensor(
                #     v_quant_data, v_scales)
                k_cache = dequantize(k_quant_data, quant_dtype, k_scales,
                                     jnp.bfloat16)
                v_cache = dequantize(v_quant_data, quant_dtype, v_scales,
                                     jnp.bfloat16)
                new_kv_cache = ((k_quant_data, k_scales),
                                (v_quant_data,
                                 v_scales)) if is_quantized else (k_cache,
                                                                  v_cache)
            else:
                k_cache, v_cache = kv_cache
                k_cache = update_cache(False, k_cache,
                                       md.kv_cache_write_indices, k)
                v_cache = update_cache(False, v_cache,
                                       md.kv_cache_write_indices, v)
            new_kv_cache = (k_cache, v_cache)

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
            new_kv_cache = ((k_quant_data, k_scales),
                            (v_quant_data,
                             v_scales)) if is_quantized else (k_cache, v_cache)
        else:
            k_cache, v_cache = kv_cache
            k_cache = update_cache(is_prefill, k_cache,
                                   md.kv_cache_write_indices, k)
            v_cache = update_cache(is_prefill, v_cache,
                                   md.kv_cache_write_indices, v)
            new_kv_cache = (k_cache, v_cache)

        if is_prefill:
            # (B, N, T, H)
            # NOTE(pooyam): Based on my benchmarks, We should not use splash kernel for normal settings (e.g., 32 heads, 8 kv heads) as flash attention is faster.
            # I think splash kernel is faster only in presence of sparsity otherwise flash attention is faster.
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

    return new_kv_cache, outputs
