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

from collections.abc import Sequence

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from sympy import divisors

from tpu_inference.kernels.ragged_paged_attention.v3.util import \
    get_dtype_packing
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@jax.jit(static_argnames=[
    'transpose_axes',
])
def xpose_full(input, *, transpose_axes):

    def xpose_kernel(input_ref, output_ref):
        output_ref[...] = input_ref[...].transpose(*transpose_axes)

    input_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
    output_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
    transposed_shape = tuple(input.shape[i] for i in transpose_axes)

    output_shape = [
        jax.ShapeDtypeStruct(shape=transposed_shape, dtype=input.dtype)
    ]
    shape_str = "x".join([str(i) for i in input.shape])
    transpose_str = "x".join([str(i) for i in transpose_axes])
    scope_name = f"xpose_full_shape_{shape_str}_xpose_{transpose_str}"
    return pl.pallas_call(xpose_kernel,
                          in_specs=input_specs,
                          out_specs=output_specs,
                          out_shape=output_shape,
                          name=scope_name)(input)


def prev_closest_valid_divisor(number: int,
                               divider: int,
                               multiple_of: int = 1) -> int:
    """
    Finds the largest divisor of 'number' that is <= 'divider' and divisible
    by 'multiple_of'.

    Raises ValueError if no divisor of 'number' satisfies both constraints.
    The exception is if min(number, divider) < multiple_of and number <= divider: return
      'number' itself.  
    This is because Pallas accepts a sublane tile equal to the full array
    dimension.
    """
    if divider < 1:
        return 1

    bound = min(number, divider)
    if bound < multiple_of:
        if divider < number:
            raise ValueError(
                f"divider={divider} < number={number} and both are < "
                f"multiple_of={multiple_of}: no valid tile size exists.")
        # number <= divider and number < multiple_of: tile equals full dim.
        return number

    all_divisors = divisors(number)
    valid = [d for d in all_divisors if d <= divider and d % multiple_of == 0]
    if valid:
        return valid[-1]

    raise ValueError(
        f"No divisor of {number} is both <= {divider} and divisible by "
        f"{multiple_of}. A non-divisor tile would produce incorrect results.")


def get_reshape_dimension(shape, reshape_axes, dtype=jnp.float32):
    input_shape_struct = jax.ShapeDtypeStruct(shape, dtype)

    def _reshape(inp):
        return inp.reshape(*reshape_axes)

    return jax.eval_shape(_reshape, input_shape_struct).shape


def identity_fn_generator(num_scalars: int = 0):
    """Method to copy input content into outputs."""

    def identity(*arg):
        n = len(arg)
        d = n // 2  # first half of args are inputs; second half are outputs
        for i in range(d):
            # Copy over kernel scalars directly into output
            if i < num_scalars:
                if arg[i].ndim == 0:
                    arg[i + d].set(arg[i].get())
                else:  # ndim == 1
                    for j in range(arg[i].shape[0]):
                        arg[i + d][j] = arg[i][j]
            # Write the input VMEM contents into the output VMEM buffer.
            else:
                arg[i + d][...] = arg[i][...]

    return identity


@jax.jit(static_argnames=['num_scalars'])
def pin_vmem_custom_call(input_tensor: jax.Array, num_scalars: int = 0):
    """Prefetches buffers to VMEM."""
    return jax.named_scope("prefetch")(pl.pallas_call(
        identity_fn_generator(num_scalars),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(input_tensor.shape, input_tensor.dtype),
        ],
        name="prefetch",
    ))(input_tensor)


@jax.jit(static_argnames=[
    'transpose_axes', 'n_tile', 'm_tile', 'parallel_axis', 'pipeline_axis'
])
def xpose_pipeline(input: jax.Array,
                   *,
                   transpose_axes: Sequence[int],
                   n_tile: int = 128,
                   m_tile: int = 128,
                   parallel_axis: int = 0,
                   pipeline_axis: int = 1):
    """
    Double buffer transpose custom call implementation.
    n_tile is used to tile the parallel dimension while m_tile is used to tile the pipeline dimension.
    Args:
      input: input array to be transposed
      tranpose_axes: transpose ordering
      n_tile: tile amount for the parallelizable axis
      m_tile: tile amount for the pipelined axis
      parallel_axis: index of the parallel axis
      pipeline_axis: index of the pipeline axis
    """

    def xpose_kernel(input_ref, output_ref):
        output_ref[...] = input_ref[...].transpose(*transpose_axes)

    n_tile = n_tile if n_tile <= input.shape[parallel_axis] else input.shape[
        parallel_axis]
    m_tile = m_tile if m_tile <= input.shape[pipeline_axis] else input.shape[
        pipeline_axis]
    # Find the best tile that (a) divides the axis inclusively
    # and (b) satisfies Pallas's sublane alignment
    # requirement: block dims must be divisible by
    # get_dtype_packing(dtype) * 8. If no such tiling exists,
    # then throw a ValueError
    sublane_multiple = get_dtype_packing(input.dtype) * 8
    n_tile_new = prev_closest_valid_divisor(input.shape[parallel_axis],
                                            n_tile,
                                            multiple_of=sublane_multiple)
    if input.shape[parallel_axis] % n_tile_new != 0:
        raise ValueError(
            f"No divisor of parallel axis size {input.shape[parallel_axis]} "
            f"is both <= {n_tile} and divisible by {sublane_multiple} "
            f"(dtype={input.dtype}). Consider increasing n_tile and/or padding your input to be "
            f"suble-aligned (i.e. a multiple of {sublane_multiple}).")
    m_tile_new = prev_closest_valid_divisor(input.shape[pipeline_axis],
                                            m_tile,
                                            multiple_of=sublane_multiple)
    if input.shape[pipeline_axis] % m_tile_new != 0:
        raise ValueError(
            f"No divisor of pipeline axis size {input.shape[pipeline_axis]} "
            f"is both <= {m_tile} and divisible by {sublane_multiple} "
            f"(dtype={input.dtype}). Consider increasing n_tile and/or padding your input to be "
            f"suble-aligned (i.e. a multiple of {sublane_multiple}).")
    if n_tile_new != n_tile:
        logger.warning(
            f"Adjusting n_tile={n_tile} to new valid tiling={n_tile_new} "
            f"which is <= n_tile={n_tile} and sublane-aligned (i.e a multiple of "
            f"{sublane_multiple}).")
    if m_tile_new != m_tile:
        logger.warning(
            f"Adjusting m_tile={m_tile} to new valid tiling={m_tile_new} "
            f"which is <= m_tile={m_tile} and sublane-aligned (i.e a multiple of "
            f"{sublane_multiple}).")
    n_tile, m_tile = n_tile_new, m_tile_new
    grid = (input.shape[parallel_axis] // n_tile,
            input.shape[pipeline_axis] // m_tile)

    # Define the input and ouptut shapes and block shapes.
    full_block_shape = list(input.shape)
    full_block_shape[parallel_axis] = n_tile
    full_block_shape[pipeline_axis] = m_tile
    full_block_shape = tuple(full_block_shape)
    transposed_block_shape = tuple(full_block_shape[i] for i in transpose_axes)
    transposed_input_shape = tuple(input.shape[i] for i in transpose_axes)
    output_shape = transposed_input_shape

    # The transposition settings will influence the input and ouptut index maps.
    def get_grid_index(i: int, j: int, input_grid: bool):
        grid_idx = [
            0,
        ] * input.ndim
        if input_grid:
            grid_idx[parallel_axis] = i
            grid_idx[pipeline_axis] = j
        else:
            grid_idx[pipeline_axis] = i
            grid_idx[parallel_axis] = j
        return grid_idx

    out_index_map = lambda i, j: get_grid_index(  # noqa: E731
        i, j, input_grid=False)

    input_specs = [
        pl.BlockSpec(
            index_map=lambda i, j: get_grid_index(i, j, input_grid=True),
            block_shape=full_block_shape,
            memory_space=pltpu.VMEM,
        )
    ]
    output_specs = [
        pl.BlockSpec(
            index_map=out_index_map,
            block_shape=transposed_block_shape,
            memory_space=pltpu.VMEM,
        )
    ]
    shape_str = "x".join([str(i) for i in input.shape])
    transpose_str = "x".join([str(i) for i in transpose_axes])
    scope_name = f"xpose_pipeline_shape_{shape_str}_xpose_{transpose_str}_n_tile_{n_tile}_m_tile_{m_tile}_pa_{parallel_axis}_pi_{pipeline_axis}"
    return pl.pallas_call(xpose_kernel,
                          grid=grid,
                          compiler_params=pltpu.CompilerParams(
                              dimension_semantics=("parallel", "arbitrary")),
                          in_specs=input_specs,
                          out_specs=output_specs,
                          out_shape=[
                              jax.ShapeDtypeStruct(shape=output_shape,
                                                   dtype=input.dtype)
                          ],
                          name=scope_name)(input)
