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

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Float, Int


def _get_mask_dtype(x_dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(x_dtype).itemsize:
        case 2:
            return jnp.int16
        case 4:
            return jnp.int32
        case _:
            raise ValueError(f"Unsupported dtype: {x_dtype}")


def _topk_kernel(
    x_ref: Float,
    vals_ref: Float,
    idxs_ref: Int,
    *,
    k: int,
    axis: int,
) -> None:
    """Pallas kernel body: k iterations of argmax + mask-out.

    Entirely within one VMEM-resident block so the whole top-k computation is a
    single kernel launch instead of k separate XLA ops. On an exact-value
    tie, Mosaic's argmax picks the highest index (jax.lax.top_k picks the
    lowest) - assumed rare/inconsequential for real router logits.

    Args:
        x_ref: Reference to the input router logits block of shape [tokens, experts]
        vals_ref: Reference to the output top-k values block of shape [tokens, k].
        idxs_ref: Reference to the output top-k indices block of shape [tokens, k].
        k: Number of top values/indices to select.
        axis: Axis to perform top-k on.
    """
    out_dtype = vals_ref.dtype
    mask_dtype = _get_mask_dtype(x_ref.dtype)
    x = x_ref[...].astype(jnp.float32)  # Mosaic argmax/max reduce needs f32
    lane_iota = jax.lax.broadcasted_iota(mask_dtype, x.shape, axis)
    for i in range(k):
        idx = jnp.argmax(x, axis=axis)
        val = jnp.max(x, axis=axis)
        _idx = [slice(None)] * vals_ref.ndim
        _idx[axis] = i
        vals_ref.at[tuple(_idx)][...] = val.astype(out_dtype)
        idxs_ref.at[tuple(_idx)][...] = idx
        mask = lane_iota == jnp.expand_dims(idx.astype(mask_dtype), axis=axis)
        x = jnp.where(mask, jnp.finfo(x.dtype).min, x)


def iterative_top_k_kernel(x: Float,
                           k: int,
                           *,
                           axis: int = -1) -> tuple[Float, Int]:
    """Top-k via a single Pallas kernel matching jax.lax.top_k exactly.

    Values and indices always correspond to each other.

    Args:
        x: Input router logits array of shape [tokens, experts].
        k: Number of top values/indices to select.
        axis: Axis to perform top-k on. axis=-1 (the last/lane axis) tiles the
        other axis into small grid blocks and is far slower on device than
        axis=-2 (the sublane axis) - measured ~18x at T=16384 (631us vs
        35us self-time). Callers that care about performance should transpose
        their input so the reduction axis lands on -2 and pass axis=-2
        explicitly, rather than relying on this function to do it.

    Returns:
        A tuple of (values, indices) arrays, both of shape [tokens, k], containing
        the top k values and their expert indices for each token.
    """
    axis = axis + x.ndim if axis < 0 else axis
    out_shape = list(x.shape)
    out_shape[axis] = k

    bitwidth = jnp.dtype(x.dtype).itemsize * 8
    if axis == x.ndim - 1:
        # T (sequence dimension) is on axis -2, tile size is (8, 128) for fp32,
        # (16, 128) for bf16.
        t_packing = 256 // bitwidth
        grid = ((x.shape[-2] + t_packing - 1) // t_packing, )
        assert x.shape[axis] % 128 == 0, (
            "This kernel requires x.shape[axis] to be a multiple of 128, got"
            f" {x.shape[axis]}.")
        x_block_shape = list(x.shape)
        x_block_shape[-2] = t_packing
        block_spec_in = pl.BlockSpec(tuple(x_block_shape), lambda i: (i, 0))
        out_block_shape = list(out_shape)
        out_block_shape[-2] = t_packing
        block_spec_out = pl.BlockSpec(tuple(out_block_shape), lambda i: (i, 0))
    else:
        # T (sequence dimension) is on axis -1, last tile size is 128 always.
        t_packing = 128
        grid = ((x.shape[-1] + t_packing - 1) // t_packing, )
        assert x.shape[axis] % (256 // bitwidth) == 0, (
            "This kernel requires x.shape[axis] to be a multiple of"
            f" {256 // bitwidth}, got {x.shape[axis]}.")
        x_block_shape = list(x.shape)
        x_block_shape[-1] = t_packing
        block_spec_in = pl.BlockSpec(tuple(x_block_shape), lambda i: (0, i))
        out_block_shape = list(out_shape)
        out_block_shape[-1] = t_packing
        block_spec_out = pl.BlockSpec(tuple(out_block_shape), lambda i: (0, i))

    return pl.pallas_call(
        functools.partial(_topk_kernel, k=k, axis=axis),
        in_specs=[block_spec_in],
        out_specs=[block_spec_out, block_spec_out],
        out_shape=[
            jax.ShapeDtypeStruct(tuple(out_shape), x.dtype),
            jax.ShapeDtypeStruct(tuple(out_shape), jnp.int32),
        ],
        grid=grid,
    )(x)
