from typing import Optional, Tuple

import jax
import jax.numpy as jnp

MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)

AxisNames = tuple[str, ...]


# TODO: max_axis
def quantize(x: jax.Array, quant_dtype: jnp.dtype):
    """Quantize key/values stored in kvcache."""
    # assert self.axis_cfg, "KV quant axis cannot be None"
    # max_axis = _get_max_axis(axis_names)
    # TODO
    max_axis = -1
    scale = jnp.max(jnp.abs(x), axis=max_axis, keepdims=True)
    if quant_dtype == jnp.int8:
        value = jnp.int8(jnp.rint(x * (MAX_INT8 / scale)))
        return value, scale
    if quant_dtype == jnp.int4:
        value = jnp.int4(jnp.rint(x * (MAX_INT4 / scale)))
        return value, scale
    if quant_dtype == jnp.float8_e4m3fn:
        value = jnp.float8_e4m3fn(x * (E4M3_MAX / scale))
        return value, scale
    raise ValueError(f"Invalid KV quant dtype:{quant_dtype}.")


def dequantize(quantized_value: jax.Array, quant_dtype: jnp.dtype,
               scale: jax.Array, target_float_dtype: jnp.dtype) -> jax.Array:
    """Dequantizes x using per-token scales."""
    q_val_float = quantized_value.astype(target_float_dtype)
    scale_float = scale.astype(target_float_dtype)

    if quant_dtype == jnp.int8:
        # Inverse: (quant_val / MAX_INT8) * scale
        original_value_approx = (q_val_float / MAX_INT8) * scale_float
        return original_value_approx
    # TODO
    # if self.dtype == "int4": # Using string to denote conceptual int4
    #     # Assuming quantized_value is int8 containing values in int4 range
    #     original_value_approx = (q_val_float / MAX_INT4) * scale_float
    #     return original_value_approx

    # if self.dtype == _F8E4M3FN_DTYPE and _FLOAT8_AVAILABLE:
    #     original_value_approx = (q_val_float / E4M3_MAX) * scale_float
    #     return original_value_approx

    raise ValueError(
        f"Invalid KV quant dtype for dequantization: {quant_dtype}.")


def update_cache_quantized(
        is_prefill: bool,
        quantized_data_cache: jax.Array,  # int8 cache, e.g., (K, L, S, H)
        scale_cache: jax.Array,  # float scale cache, e.g., (K, L, S, 1)
        quant_dtype: jnp.dtype,
        indices: jax.Array,  # Indices for writing
        operand: jax.
    Array,  # float operand to be written, e.g., (B, K, T, H) or (B, K, 1, H)
        prefill_seq_len: Optional[
            int] = None,  # Only for sliding window in prefill
        sliding_window: Optional[
            int] = None,  # Only for sliding window in prefill
) -> Tuple[jax.Array, jax.Array]:
    """
    Updates the quantized KV cache.
    - operand is float. It will be quantized before writing.
    - quantized_data_cache stores QUANT_DTYPE (e.g., int8).
    - scale_cache stores scales (float/bf16), matching operand's original precision.
      Shape assumed to be (K, L, S, 1) to store one scale per token.
    """
    B, K, T, H = operand.shape  # Original operand shape, e.g., (batch, num_kv_heads, seq_len, head_dim)
    K_c, L, S, H_c = quantized_data_cache.shape  # Cache shape, e.g., (num_kv_heads, num_blocks, block_size, head_dim)

    assert K == K_c and H == H_c
    assert scale_cache.shape == (K_c, L, S, 1)  # Expecting per-token scales

    if is_prefill:
        # Prefill: operand (B, K, T, H), indices (B, T // S)
        if sliding_window and T > sliding_window:
            assert B == 1  # Sliding window typically for single sequence prefill
            start_index = jax.lax.max(
                0, prefill_seq_len -
                sliding_window)  # prefill_seq_len is actual length
            operand = jax.lax.dynamic_slice_in_dim(operand,
                                                   start_index,
                                                   sliding_window,
                                                   axis=2)
            _T = sliding_window

        # Reshape operand to match cache block structure: (K, B*T//S, S, H)
        # This is (num_kv_heads, num_blocks_to_write, block_size, head_dim)
        # num_blocks_to_write = _B * _T // _S
        # operand_reshaped_for_quant = operand.transpose(1, 0, 2, 3).reshape(
        #     _K, num_blocks_to_write, _S, _H)  # (K, I, S, H)

        # ruff: noqa: E741
        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)

        # Quantize the operand blocks
        # Each (S,H) slice is a "token group" for quantization here if S > 1
        # If S=1 (each token is a block), then it's per-token quantization
        # quantize_per_token_symmetric expects (..., feature_dim)
        # Here, for each (K, I) we have an (S, H) block.
        # If we want one scale per (S,H) block:
        #   Pass operand_reshaped_for_quant.reshape(_K * num_blocks_to_write, _S * _H)
        #   This makes scales shape (_K * num_blocks_to_write, 1)
        # If we want one scale per token (per H vector inside S,H):
        #   Pass operand_reshaped_for_quant.reshape(_K * num_blocks_to_write * _S, _H)
        #   This makes scales shape (_K * num_blocks_to_write * _S, 1) -> reshape to (K, I, S, 1)

        # Let's stick to per-token (innermost H-dim vector) quantization:
        # Shape before quant: (K, I, S, H)
        # TODO: make quant dtype configurable
        quant_operand_blocks, operand_scales = quantize(operand, quant_dtype)
        indices = indices.reshape(I)

        # Example quantized_data_cache shape: (8, 3752, 32, 128)
        # Example quant_operand_blocks shape: (8, 4, 32, 128)
        quantized_data_cache = quantized_data_cache.at[:, indices, :, :].set(
            quant_operand_blocks)
        scale_cache = scale_cache.at[:, indices, :, :].set(operand_scales)

    else:  # Decode
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        quantized_data_cache = quantized_data_cache.reshape(K, L * S, H)
        scale_cache = scale_cache.reshape(K, L * S, 1)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        quant_operand_tokens, operand_token_scales = quantize(
            operand, quant_dtype)
        # quant_operand_tokens shape: (K, B, H)
        # operand_token_scales shape: (K, B, 1)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        # Example quantized_data_cache shape: (8, 120064, 128)
        # Example quant_operand_tokens shape: (8, 1, 128)
        quantized_data_cache = quantized_data_cache.at[jnp.arange(K)[
            ..., None], indices, :].set(quant_operand_tokens)
        quantized_data_cache = quantized_data_cache.reshape(K, L, S, H)

        scale_cache = scale_cache.at[jnp.arange(K)[..., None],
                                     indices, :].set(operand_token_scales)
        scale_cache = scale_cache.reshape(K, L, S, 1)

    return quantized_data_cache, scale_cache


def chunked_prefill_update_cache_quantized(
        quantized_data_cache: jax.Array,  # int8 cache, e.g., (K, L, S, H)
        scale_cache: jax.Array,  # float scale cache, e.g., (K, L, S, 1)
        quant_dtype: jnp.dtype,  # The quantization dtype (e.g., jnp.int8)
        indices: jax.Array,  # Indices for writing
        operand: jax.Array,  # float operand (B, K, T, H)
        num_decode_seqs: jax.Array,  # (1,) array with number of decode tokens
) -> Tuple[jax.Array, jax.Array]:
    """
    Updates the quantized KV cache for chunked prefill, where an operand contains
    both decode and prefill tokens.
    """
    B, K, T, H = operand.shape
    K_c, L, S, H_c = quantized_data_cache.shape
    assert K == K_c and H == H_c, "Cache and operand dimensions must match"
    assert scale_cache.shape == (K, L, S, 1), "Scale cache shape is incorrect"
    assert B == 1, "Chunked prefill assumes batch size of 1"

    operand = jnp.squeeze(operand, 0)
    # operand now: (K, T, H)

    # Handle Decode tokens kv cache update.
    decode_indices = jax.lax.slice(indices, (0, ), (T, ))

    # Number of valid indice update could be much smaller
    # than T. We improve performance by skipping the
    # update for those padded indices as much as possible.
    # TODO(b/396129273): tune the value of DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE
    # base on benchmarking
    DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE = 16
    assert T % DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE == 0

    decode_indices = decode_indices.reshape((
        T // DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
    ))
    decode_update_operand = operand.reshape(
        K,
        T // DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        H,
    )

    # Quantize the part of the operand for decode updates
    quant_decode_operand, decode_scales = quantize(decode_update_operand,
                                                   quant_dtype)
    # quant_decode_operand shape: (K, T/block, block, H)
    # decode_scales shape: (K, T/block, block, 1)

    quantized_data_cache = quantized_data_cache.reshape(K, L * S, H)
    scale_cache = scale_cache.reshape(K, L * S, 1)

    def decode_update_body(i, caches):
        q_cache, s_cache = caches
        q_cache_updated = q_cache.at[jnp.arange(K)[..., None],
                                     decode_indices[i], :].set(
                                         quant_decode_operand[:, i, :, :])
        s_cache_updated = s_cache.at[jnp.arange(K)[..., None],
                                     decode_indices[i], :].set(
                                         decode_scales[:, i, :, :])
        return q_cache_updated, s_cache_updated

    # Loop over the decode blocks and update both caches
    quantized_data_cache, scale_cache = jax.lax.fori_loop(
        0,
        jnp.ceil(num_decode_seqs[0] /
                 DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE).astype(jnp.int32),
        body_fun=decode_update_body,
        init_val=(quantized_data_cache, scale_cache),
    )

    quantized_data_cache = quantized_data_cache.reshape(K, L, S, H)
    scale_cache = scale_cache.reshape(K, L, S, 1)

    # Handle Prefill tokens kv cache update.
    # ruff: noqa: E741
    I = T // S
    prefill_indices = jax.lax.slice(indices, (T, ), (T + I, ))
    # cache: (K, L, S, H)
    # prefill_operand: (K, T, H) -> (K, I, S, H)
    # prefill_indices: (I,)
    prefill_operand = operand.reshape(K, I, S, H)

    # Quantize the part of the operand for prefill updates
    quant_prefill_operand, prefill_scales = quantize(prefill_operand,
                                                     quant_dtype)

    # Update both caches with the prefill blocks
    quantized_data_cache = quantized_data_cache.at[:,
                                                   prefill_indices, :, :].set(
                                                       quant_prefill_operand)
    scale_cache = scale_cache.at[:, prefill_indices, :, :].set(prefill_scales)

    return quantized_data_cache, scale_cache
