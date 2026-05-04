import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
from typing import Tuple

min_pipeline = 3


@jax.jit(
    static_argnames=['transpose_axes',
                     'reshape_axes']
)
def xpose_full(input, *, transpose_axes, reshape_axes=None):
    def xpose_kernel(input_ref, output_ref):
        if reshape_axes is not None:
            output_ref[...] = input_ref[...]\
                .transpose(*transpose_axes)\
                .reshape(*reshape_axes)
        else:
            output_ref[...] = input_ref[...].transpose(*transpose_axes)
    input_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
    ]
    output_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
    ]
    transposed_shape = tuple(input.shape[i] for i in transpose_axes)
    if reshape_axes is not None:
        final_shape = get_reshape_dimension(transposed_shape, reshape_axes, dtype=input.dtype)
    else:
        final_shape = transposed_shape
        
    output_shape = [
        jax.ShapeDtypeStruct(
            shape=final_shape,
            dtype=input.dtype
            )
    ]
    shape_str = "x".join([str(i) for i in input.shape])
    transpose_str = "x".join([str(i) for i in transpose_axes])
    scope_name = f"xpose_full_shape_{shape_str}_xpose_{transpose_str}"
    return pl.pallas_call( 
        xpose_kernel,
        in_specs=input_specs,
        out_specs=output_specs,
        out_shape=output_shape,
        name=scope_name
    )(input)


def get_reshape_dimension(shape, reshape_axes, dtype=jnp.float32):
    input_shape_struct = jax.ShapeDtypeStruct(shape, dtype)
    def _reshape(inp):
        return inp.reshape(*reshape_axes)
    return jax.eval_shape(_reshape, input_shape_struct).shape

@jax.jit(
    static_argnames=['transpose_axes',
                     'reshape_axes', 
                     'n_tile', 
                     'm_tile']
)
def x_pose_pipeline(input, *, transpose_axes, reshape_axes=None, n_tile=128, m_tile=128):
    def x_pose_kernel(input_ref, output_ref):
        if reshape_axes is not None:
            output_ref[...] = input_ref[...]\
                .transpose(*transpose_axes)\
                .reshape(*reshape_axes)
        else:
            output_ref[...] = input_ref[...].transpose(*transpose_axes)
    n_tile = n_tile if n_tile <= input.shape[0] else input.shape[0]
    m_tile = m_tile if m_tile <= input.shape[1] else input.shape[1]
    assert input.shape[0] % n_tile == 0, f"input.shape[0]={input.shape[0]} must be divisible by n_tile={n_tile}."
    assert input.shape[1] % m_tile == 0, f"input.shape[1]={input.shape[1]} must be divisible by m_tile={m_tile}."

    grid = (input.shape[0] // n_tile, input.shape[1] // m_tile)

    full_block_shape = (n_tile, m_tile) + input.shape[2:]
    transposed_block_shape  = tuple(full_block_shape[i] for i in transpose_axes)
    transposed_input_shape = tuple(input.shape[i] for i in transpose_axes)
    if reshape_axes is not None:
        reshaped_block_shape = get_reshape_dimension(transposed_block_shape, reshape_axes, dtype=input.dtype)
        output_shape = get_reshape_dimension(transposed_input_shape, reshape_axes, dtype=input.dtype)
        _n_tiles = input.shape[0] // n_tile
        _trailing = len(reshaped_block_shape) - 1
        out_index_map = lambda i, j, _nt=_n_tiles, _tr=_trailing: (j * _nt + i,) + (0,) * _tr
    else:
        reshaped_block_shape = transposed_block_shape
        output_shape = transposed_input_shape
        out_index_map = lambda i, j: (j, i) + (0,) * (input.ndim - 2)

    input_specs = [
        pl.BlockSpec(
            index_map=lambda i, j: (i, j) + (0,) * (input.ndim - 2),
            block_shape=full_block_shape,
            memory_space=pltpu.VMEM,
        )
    ]
    output_specs = [
        pl.BlockSpec(
            index_map=out_index_map,
            block_shape=reshaped_block_shape,
            memory_space=pltpu.VMEM,
        )
    ]
    shape_str = "x".join([str(i) for i in input.shape])
    transpose_str = "x".join([str(i) for i in transpose_axes])
    scope_name = f"xpose_full_shape_{shape_str}_xpose_{transpose_str}_n_tile_{n_tile}_m_tile_{m_tile}"
    return pl.pallas_call(
        x_pose_kernel,
        grid=grid,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        in_specs=input_specs,
        out_specs=output_specs,
        out_shape=[
            jax.ShapeDtypeStruct(
                shape=output_shape,
                dtype=input.dtype
            )
        ],
        name=scope_name
    )(input)