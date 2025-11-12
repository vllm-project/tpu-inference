import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


# Dummy Pallas kernel - takes 2D weights
def dummy_matmul_kernel(x_ref, w_q_ref, w_s_ref, out_ref):
    print(
        f"[right before matmul_kernel after vmap] shapes: x={x_ref.shape}, w_q={w_q_ref.shape}, w_s={w_s_ref.shape}"
    )
    out_ref[...] = jnp.matmul(x_ref[...], w_q_ref[...])


def dummy_quantized_matmul_kernel(x, w_q, w_s):
    out_shape = jax.ShapeDtypeStruct((x.shape[0], w_q.shape[1]), x.dtype)

    # Simplified BlockSpecs for the dummy kernel.
    # We use full block sizes, meaning no tiling within the kernel.
    x_spec = pl.BlockSpec(x.shape, lambda i, j: (0, 0))
    w_q_spec = pl.BlockSpec(w_q.shape, lambda i, j: (0, 0))
    w_s_spec = pl.BlockSpec(w_s.shape, lambda i, j: (0, 0))
    out_spec = pl.BlockSpec(out_shape.shape, lambda i, j: (0, 0))

    return pl.pallas_call(
        dummy_matmul_kernel,
        in_specs=[x_spec, w_q_spec, w_s_spec],
        out_specs=out_spec,
        grid=(1, 1),  # Single grid for simplicity
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(),
        name="dummy_kernel")(x, w_q, w_s)


def repro_vmap_pallas_error():
    if jax.device_count() < 2:
        print("This test requires at least 2 TPU devices.")
        return

    devices = jax.devices()[:2]
    mesh = Mesh(np.array(devices), ('model', ))
    print(f"Using mesh: {mesh}")

    key = jax.random.PRNGKey(0)
    x_shape = (16, 5120)
    w_q_shape = (5120, 64, 128)
    w_s_shape = (1, 64, 128)

    x = jax.random.normal(key, x_shape, dtype=jnp.bfloat16)
    w_q = jax.random.normal(key, w_q_shape, dtype=jnp.bfloat16)
    w_s = jax.random.normal(key, w_s_shape, dtype=jnp.bfloat16)

    # Sharding
    x_sharding = NamedSharding(mesh, P(None, None))
    w_q_sharding = NamedSharding(mesh, P(None, "model", None))
    w_s_sharding = NamedSharding(mesh, P(None, "model", None))
    out_sharding = NamedSharding(mesh, P(None, "model",
                                         None))  # Expecting 3D output

    x = jax.device_put(x, x_sharding)
    w_q = jax.device_put(w_q, w_q_sharding)
    w_s = jax.device_put(w_s, w_s_sharding)

    def wrapper(x_local, w_q_local, w_s_local):
        print(
            f"[ORIGINAL before vmap] Shapes: x={x_local.shape}, w_q={w_q_local.shape}, w_s={w_s_local.shape}"
        )

        vmapped_kernel = jax.vmap(
            dummy_quantized_matmul_kernel,
            in_axes=(None, 1, 1),  # Map over axis 1 of w_q and w_s
            out_axes=1  # Place the mapped axis at index 1 in the output
        )

        output = vmapped_kernel(x_local, w_q_local, w_s_local)
        # Expected output shape: (16, 32, 128)
        return output

    shard_map(wrapper,
              mesh=mesh,
              in_specs=(x_sharding.spec, w_q_sharding.spec, w_s_sharding.spec),
              out_specs=out_sharding.spec,
              check_rep=False)(x, w_q, w_s)


if __name__ == '__main__':
    # This example needs to be run on a machine with TPU devices
    try:
        print(f"JAX device count: {jax.device_count()}")
        print(f"JAX devices: {jax.devices()}")
        repro_vmap_pallas_error()
    except Exception as main_e:
        print(f"An error occurred: {main_e}")
