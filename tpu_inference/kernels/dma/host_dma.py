# SPDX-License-Identifier: Apache-2.0
""" Host <-> HBM DMA kernel"""
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def host_hbm_dma(x_ref, y_ref):
    """
    DMA a jax array between host and hbm
    Input jax array ref: x_ref
    Output jax array ref: y_ref
    """

    def body(sem):
        pltpu.async_copy(x_ref, y_ref, sem).wait()

    pl.run_scoped(body, pltpu.SemaphoreType.DMA)


# NOTE(jcgu): input / out arrays should have the same sharding, but different memory_kind
# NOTE(jcgu): only support NamedSharding, does not support SingleDeviceSharding
def d2h_dma(
    input_array: jax.Array,
    input_sharding: jax.sharding.NamedSharding,
    out_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
    """ DMA a device jax array to host memory.
    Args:
        input_array: input jax array on device hbm
        input_sharding: input's device sharding
        out_sharding: output's host sharding
    Returns:
        jax array on host memory with the same sharding
    """

    @jax.jit
    def _d2h_dma_call(x):
        return pl.pallas_call(
            host_hbm_dma,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.HOST),
            out_shape=pltpu.HOST(shape=x.shape, dtype=x.dtype),
            name="d2h_dma_kernel",
        )(x)

    d2h_dma_kernel = jax.jit(
        jax.shard_map(
            _d2h_dma_call,
            mesh=input_sharding.mesh,
            in_specs=input_sharding.spec,
            out_specs=out_sharding.spec,
            check_vma=False,
        ),
        out_shardings=out_sharding,
    )

    return d2h_dma_kernel(input_array)


# NOTE(jcgu): input / out arrays should have the same sharding, but different memory_kind
# NOTE(jcgu): only support NamedSharding, does not support SingleDeviceSharding
def h2d_dma(
    input_array: jax.Array,
    input_sharding: jax.sharding.NamedSharding,
    out_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
    """ DMA a host jax array to device hbm.
    Args:
        input_array: input jax array on host memory
        input_sharding: the host sharding for input
        out_sharding: the device sharding for output
    Returns:
        jax array on device hbm with the assigned sharding
    """

    @jax.jit
    def _h2d_dma_call(x):
        return pl.pallas_call(
            host_hbm_dma,
            in_specs=[
                pl.BlockSpec(memory_space=pl.HOST),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
            name="h2d_dma_kernel",
        )(x)

    h2d_dma_kernel = jax.jit(
        jax.shard_map(
            _h2d_dma_call,
            mesh=input_sharding.mesh,
            in_specs=input_sharding.spec,
            out_specs=out_sharding.spec,
            check_vma=False,
        ),
        out_shardings=out_sharding,
    )

    return h2d_dma_kernel(input_array)
