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


# TODO(jcgu): unit tests
# TODO(jcgu): verify single device sharding case
def d2h_dma(
    input_array: jax.Array,
    out_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
    """ DMA a device jax array to host memory.
    Args:
        input_array: input jax array on device hbm
        out_sharding: output's host sharding
    Returns:
        jax array on host memory with the same sharding
    """
    device_sharding = input_array.sharding
    assert isinstance(device_sharding, jax.sharding.NamedSharding)
    assert isinstance(out_sharding, jax.sharding.NamedSharding)
    assert device_sharding.memory_kind == "device"
    assert out_sharding.memory_kind == "pinned_host"

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
            mesh=device_sharding.mesh,
            in_specs=device_sharding.spec,
            out_specs=out_sharding.spec,
        ),
        out_shardings=out_sharding,
    )

    return d2h_dma_kernel(input_array)


# TODO(jcgu): unit tests
# TODO(jcgu): verify single device sharding case
def h2d_dma(
    input_array: jax.Array,
    out_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
    """ DMA a host jax array to device hbm.
    Args:
        input_array: input jax array on host memory
        out_sharding: the device sharding for output
    Returns:
        jax array on device hbm with the assigned sharding
    """

    host_sharding = input_array.sharding

    # TODO(jcgu): support SingleDeviceSharding
    assert isinstance(host_sharding, jax.sharding.NamedSharding)
    assert isinstance(out_sharding, jax.sharding.NamedSharding)
    assert host_sharding.memory_kind == "pinned_host"
    assert out_sharding.memory_kind == "device"

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
            mesh=host_sharding.mesh,
            in_specs=host_sharding.spec,
            out_specs=out_sharding.spec,
            check_vma=False,
        ),
        out_shardings=out_sharding,
    )
    return h2d_dma_kernel(input_array)
