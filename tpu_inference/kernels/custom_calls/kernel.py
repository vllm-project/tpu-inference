import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

@jax.jit(
    static_argnames=['transpose_axes']
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
    transposed_shape = [input.shape[i] for i in transpose_axes]
    output_shape = [
        jax.ShapeDtypeStruct(
            shape=tuple(transposed_shape),
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
