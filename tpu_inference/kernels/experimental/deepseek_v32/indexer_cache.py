# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek V3.2 indexer K-cache quantization and insertion."""

import jax
import jax.numpy as jnp


def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dimension as bytes."""
    byte_view = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if byte_view.ndim > x.ndim:
        byte_view = byte_view.reshape(*x.shape[:-1], -1)
    return byte_view


def quantize_indexer_k(
    k: jax.Array,
    *,
    quant_block_size: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Quantize indexer keys using V3.2's FP8/UE8M0-scale contract.

    The returned scales are the power-of-two values used for quantization,
    stored as float32 by the V3.2 cache format.
    """
    if k.ndim != 2:
        raise ValueError(
            f"k must have shape [num_tokens, head_dim], got {k.shape}")
    if quant_block_size <= 0 or k.shape[-1] % quant_block_size:
        raise ValueError(
            f"head_dim {k.shape[-1]} must be divisible by quant_block_size "
            f"{quant_block_size}")

    blocked = k.astype(jnp.float32).reshape(k.shape[0], -1, quant_block_size)
    amax = jnp.maximum(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    scales = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    quantized = (blocked / scales).astype(jnp.float8_e4m3fn).reshape(k.shape)
    return quantized, jnp.squeeze(scales, axis=-1).astype(jnp.float32)


def pack_indexer_k_records(
    k: jax.Array,
    *,
    record_width: int,
    quant_block_size: int = 128,
) -> jax.Array:
    """Pack V3.2 indexer keys as FP8 values followed by float32 scales."""
    quantized, scales = quantize_indexer_k(k,
                                           quant_block_size=quant_block_size)
    num_scale_bytes = scales.shape[-1] * 4
    packed_width = k.shape[-1] + num_scale_bytes
    if record_width < packed_width:
        raise ValueError(
            f"record_width {record_width} cannot hold {k.shape[-1]} FP8 "
            f"values and {num_scale_bytes} scale bytes")

    value_bytes = _to_byte_lane(quantized)
    scale_bytes = _to_byte_lane(scales)
    packed = jnp.concatenate((value_bytes, scale_bytes), axis=-1)
    records = jnp.zeros((k.shape[0], record_width), dtype=jnp.uint8)
    return records.at[:, :packed_width].set(packed)


def insert_indexer_k_cache(
    cache: jax.Array,
    k: jax.Array,
    slot_mapping: jax.Array,
    *,
    quant_block_size: int = 128,
) -> jax.Array:
    """Quantize and scatter indexer keys into a paged V3.2 cache.

    ``cache`` may be either ``[pages, page_size, width]`` or TPU-packed
    ``[pages, page_size / packing, packing, width]``. ``slot_mapping`` uses
    physical flattened token slots; negative or out-of-range slots are
    ignored, matching vLLM's padding sentinel behavior.
    """
    if cache.dtype != jnp.uint8 or cache.ndim not in (3, 4):
        raise ValueError(
            "cache must be a rank-3 or rank-4 uint8 paged cache, got "
            f"shape={cache.shape}, dtype={cache.dtype}")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != k.shape[0]:
        raise ValueError("slot_mapping must have one entry per key token, got "
                         f"slot_mapping={slot_mapping.shape}, k={k.shape}")

    record_width = cache.shape[-1]
    records = pack_indexer_k_records(
        k,
        record_width=record_width,
        quant_block_size=quant_block_size,
    )
    flat_cache = cache.reshape(-1, record_width)
    flat_cache = flat_cache.at[slot_mapping].set(
        records,
        mode="drop",
        wrap_negative_indices=False,
    )
    return flat_cache.reshape(cache.shape)
