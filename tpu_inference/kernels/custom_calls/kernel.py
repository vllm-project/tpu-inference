import bisect
from sympy import divisors
import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
from typing import Tuple

min_pipeline = 3


@jax.jit(
    static_argnames=['transpose_axes',]
)
def xpose_full(input, *, transpose_axes):
    def xpose_kernel(input_ref, output_ref):
        output_ref[...] = input_ref[...].transpose(*transpose_axes)
    input_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
    ]
    output_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
    ]
    transposed_shape = tuple(input.shape[i] for i in transpose_axes)
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

def prev_closest_divisor(number: int, divider: int) -> int:
    """
    Finds the smallest divisor of 'number' that is <= 'divider'.
    """
    if divider < 1:
        return 1
        
    all_divisors = divisors(number)
    idx = bisect.bisect_right(all_divisors, divider)
    if idx > 0:
        return all_divisors[idx - 1]
        
    raise ValueError(f"Could not find a next closest divisor for number={number} and divider={divider}.")


def get_reshape_dimension(shape, reshape_axes, dtype=jnp.float32):
    input_shape_struct = jax.ShapeDtypeStruct(shape, dtype)
    def _reshape(inp):
        return inp.reshape(*reshape_axes)
    return jax.eval_shape(_reshape, input_shape_struct).shape

def identity_fn_generator(num_scalars: int = 0):
  def identity(*arg):
    n = len(arg)
    d = n // 2
    for i in range(d):
      if i < num_scalars:
        if arg[i].ndim == 0:
          arg[i+d].set(arg[i].get())
        else:  # ndim == 1
          for j in range(arg[i].shape[0]):
            arg[i+d][j] = arg[i][j]
      else:
        arg[i+d][...] = arg[i][...]
  return identity

@jax.jit(static_argnames=['num_scalars'])
def pin_vmem_custom_call(input_tensor, num_scalars=0):
    return jax.named_scope("prefetch")(
    pl.pallas_call(
        identity_fn_generator(num_scalars),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(
                input_tensor.shape, input_tensor.dtype),
        ],
        name="prefetch",
    ))(input_tensor)

@jax.jit(
    static_argnames=['transpose_axes',
                     'n_tile',
                     'm_tile',
                     'parallel_axis',
                     'pipeline_axis']
)
def xpose_pipeline(input, *, transpose_axes, n_tile=128, m_tile=128, parallel_axis=0, pipeline_axis=1):
    def xpose_kernel(input_ref, output_ref):
        output_ref[...] = input_ref[...].transpose(*transpose_axes)
    n_tile = n_tile if n_tile <= input.shape[parallel_axis] else input.shape[parallel_axis]
    m_tile = m_tile if m_tile <= input.shape[pipeline_axis] else input.shape[pipeline_axis]
    assert input.shape[parallel_axis] % n_tile == 0, f"input.shape[{parallel_axis}]={input.shape[parallel_axis]} must be divisible by n_tile={n_tile}."
    assert input.shape[pipeline_axis] % m_tile == 0, f"input.shape[{pipeline_axis}]={input.shape[pipeline_axis]} must be divisible by m_tile={m_tile}."

    grid = (input.shape[parallel_axis] // n_tile, input.shape[pipeline_axis] // m_tile)

    full_block_shape = list(input.shape)
    full_block_shape[parallel_axis] = n_tile
    full_block_shape[pipeline_axis] = m_tile
    full_block_shape = tuple(full_block_shape)
    transposed_block_shape  = tuple(full_block_shape[i] for i in transpose_axes)
    transposed_input_shape = tuple(input.shape[i] for i in transpose_axes)
    output_shape = transposed_input_shape
    def get_grid_index(i: int, j: int, input_grid: bool):
        grid_idx = [0,] * input.ndim
        if input_grid:
            grid_idx[parallel_axis] = i
            grid_idx[pipeline_axis] = j
        else:
            grid_idx[pipeline_axis] = i
            grid_idx[parallel_axis] = j
        return grid_idx


    out_index_map = lambda i, j: get_grid_index(i, j, input_grid=False)

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
    scope_name = f"xpose_full_shape_{shape_str}_xpose_{transpose_str}_n_tile_{n_tile}_m_tile_{m_tile}_pa_{parallel_axis}_pi_{pipeline_axis}"
    return pl.pallas_call(
        xpose_kernel,
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