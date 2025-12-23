# Copyright 2025 Google LLC
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

import itertools
from typing import Tuple

import jax
import jax.numpy as jnp

MXFP4_BLOCK_SIZE = 32


def quantize_tensor_to_mxfp4_packed(
    tensor: jax.Array,
    axis: int | tuple = -1,
) -> Tuple[jax.Array, jax.Array]:
    """Quantize a tensor to mxfp4 and pack it into uint8."""

    # Perform regular block quantization.
    tensor_q, scale = quantize_tensor(
        jnp.float4_e2m1fn,
        tensor,
        axis,
        MXFP4_BLOCK_SIZE,
    )

    # last two e2m1 elements will be packed into a single uint8 element.
    bitcast_shape = tensor_q.shape[:-1] + (-1, 2)
    tensor_q = tensor_q.reshape(bitcast_shape)
    tensor_q_packed = jax.lax.bitcast_convert_type(tensor_q, jnp.uint8)

    # Since TPU does not have native support for e8m0, we convert scale into
    # e8m0 manually and store it as uint8.
    e8m0_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    _, scale_exp = jnp.frexp(scale)
    # Subtract exponents by one since e8m0 has no decimal.
    scale_exp -= 1
    scale_exp = (scale_exp - e8m0_finfo.minexp).astype(jnp.uint8)

    return tensor_q_packed, scale_exp


def u8_unpack_e2m1(u8_packed_e2m1: jax.Array) -> jax.Array:
    """Unpack e2m1 tensor that was packed into u8."""
    assert u8_packed_e2m1.dtype == jnp.uint8
    e2m1 = jax.lax.bitcast_convert_type(u8_packed_e2m1, jnp.float4_e2m1fn)
    # bitcast creates one more dimension that splits 8 bits into two e2m1.
    # we flatten them with the last dim.
    return jnp.reshape(e2m1, e2m1.shape[:-2] + (-1, ))


def e8m0_to_fp32(u8: jax.Array) -> jax.Array:
    """Convert e8m0 (that was bitcasted to u8) into fp32."""
    assert u8.dtype == jnp.uint8

    e8_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    exponents = u8.astype(jnp.int32) + e8_finfo.minexp
    ones = jnp.ones_like(u8, dtype=jnp.float32)
    return jnp.ldexp(ones, exponents)


def awq_u32_unpack_u4(awq_u32_packed: jax.Array) -> jax.Array:
    """Unpack u4 tensor that was packed into u32 in awq ordering."""

    awq_u4 = jax.lax.bitcast_convert_type(awq_u32_packed, jnp.uint4)

    # AWQ packs 8 uint4 into 32-bits in this order: (0, 2, 4, 6, 1, 3, 5, 7).
    # Following list maps the order used by AWQ into an ascending order.
    reverse_awq_order = (0, 4, 1, 5, 2, 6, 3, 7)
    u4 = awq_u4[..., reverse_awq_order]
    return jnp.reshape(u4, u4.shape[:-2] + (-1, ))


def dequantize_tensor(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | None | tuple = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize a quantized tensor

    Args:
        tensor_q: Quantized tensor.
        scale: Quantization scale.
        axis: The axis tensor was quantized. None denotes per-tensor.
        out_dtype: Dtype of the output.

    Returns:
        Dequantized tensor_q.
    """
    if axis is None:
        # Perform per-tensor quantization.
        axis = [i for i in range(tensor_q.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q.shape
    if tensor_q.ndim == scale.ndim:
        # Indicates the tensor was block quantized.
        blocked_shape = [[i] for i in orig_shape]
        for i in axis:
            num_blocks = scale.shape[i]
            if tensor_q.shape[i] % num_blocks:
                raise ValueError(
                    f"Unable to perform block dequantization. axis={i} of "
                    f"{tensor_q.shape=} is not divisible by {num_blocks=}", )
            block_size = tensor_q.shape[i] // num_blocks

            blocked_shape[i] = (num_blocks, block_size)

        # Convert all axis into positive values.
        axis = sorted([(i + tensor_q.ndim) % tensor_q.ndim for i in axis])
        # Shift axis by 1 since its original position is now occupied by
        # num_blocks dim. Also, if n axes before an axis was also quantized,
        # shift its position by n.
        axis = [1 + n + i for n, i in enumerate(axis)]

        # Flatten list of lists that contains (num_blocks, block).
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor_q = tensor_q.reshape(blocked_shape)

    scale = jnp.expand_dims(scale, axis)

    tensor = (tensor_q.astype(jnp.float32) * scale).astype(out_dtype)

    return tensor.reshape(orig_shape)


def dequantize_tensor_from_mxfp4_packed(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | tuple = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize packed mxfp4 tensor.

    Args:
        tensor_q: fp4 tensor packed into uint8.
        scale: e8m0 scale packed into uint8.
        axis: The axis tensor was quantized.
        out_dtype: Dtype of the output.

    Returns:
        Dequantized tensor_q.
    """
    tensor_e2m1 = u8_unpack_e2m1(tensor_q)
    scale_fp32 = e8m0_to_fp32(scale)

    return dequantize_tensor(
        tensor_e2m1,
        scale_fp32,
        axis,
        out_dtype,
    )


def quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    axis: int | tuple | None = -1,
    block_size: int | None = None,
    pad_tensor: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Quantize tensor.

    Args:
        dtype: dtype to perform quantization.
        tensor: Unquantized tensor
        axis: Axis to perform quantization. None denotes per-tensor.
        block_size: Specify block quantization size.
        pad_tensor: Whether to pad the axis along block size.

    Returns:
        Tensor quantized to dtype.
    """
    if axis is None:
        # Perform per-tensor quantization.
        axis = [i for i in range(tensor.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor.shape
    mask = jnp.ones_like(tensor, jnp.int32)

    if block_size is not None:
        if isinstance(block_size, int):
            block_size = [block_size] * len(axis)

        blocked_shape = [[i] for i in orig_shape]
        pad_width = [[0, 0] for _ in range(tensor.ndim)]
        for i, block in zip(axis, block_size):
            num_blocks = (tensor.shape[i] + block - 1) // block
            padding_size = num_blocks * block - tensor.shape[i]
            if padding_size and not pad_tensor:
                raise ValueError(
                    f"Unable to perform block quantization. axis={i} of "
                    f"{tensor.shape=} is not divisible by {block=}")

            # Pad the tensor to align with block size.
            pad_width[i][1] = padding_size

            blocked_shape[i] = (num_blocks, block)

        # In order to avoid padded values affecting scale value, we pad it
        # using edge value of the tensor.
        tensor = jnp.pad(tensor, pad_width, "edge")
        mask = jnp.pad(mask, pad_width)

        orig_shape = tensor.shape
        # Convert all axis into positive values.
        axis = sorted([i % tensor.ndim for i in axis])
        # Shift axis by 1 since its original position is now occupied by
        # num_blocks dim. Also, if n axes before an axis was also quantized,
        # shift its position by n.
        axis = [1 + n + i for n, i in enumerate(axis)]

        # Flatten list of lists that contains (num_blocks, block).
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor = tensor.reshape(blocked_shape)

    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
    else:
        dtype_info = jnp.finfo(dtype)

    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
    scale = abs_max / dtype_max

    tensor_q = jnp.clip(tensor / scale, dtype_min, dtype_max)
    tensor_q = tensor_q.reshape(orig_shape)
    tensor_q = tensor_q.astype(dtype)

    # To avoid padded values affecting output of quantized matmul, we mask them
    # out with 0s.
    tensor_q = jnp.where(mask, tensor_q, 0)

    scale = jnp.squeeze(scale, axis).astype(jnp.float32)

    return tensor_q, scale


def static_per_tensor_quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    scale: float,
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
    else:
        dtype_info = jnp.finfo(dtype)

    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    return jnp.clip(tensor / scale, dtype_min, dtype_max).astype(dtype)


def quantize_kv(
    dtype: jnp.dtype,
    key: jax.Array,
    value: jax.Array,
    k_scale: float,
    v_scale: float,
) -> Tuple[jax.Array, jax.Array]:
    """Static quantize key and value tensors."""
    key = static_per_tensor_quantize_tensor(dtype, key, k_scale)
    value = static_per_tensor_quantize_tensor(dtype, value, v_scale)
    return key, value
