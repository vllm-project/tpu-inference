from typing import Tuple

import jax
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Jax default random generator is memory inefficient. See https://github.com/google/jax/issues/21045
jax.config.update("jax_threefry_partitionable", True)


def sharding_init(
    named_axes: Tuple[str, ...],
    mesh: Mesh,
    use_constant: bool = False,
):
    # This sharding_init runs fast, but it creates the tensor
    # on the single device first, then shards it across the mesh.
    # It will easily lead to OOM if the tensor is too large.
    # The normal initializer cannot be used with int8 tensors.
    # The constant initializer should be used with int8 tensors.
    def sharded_init(*args, **kwargs):
        if use_constant:
            state = nn.initializers.constant(0.5)(*args, **kwargs)
        else:
            state = nn.initializers.normal()(*args, **kwargs)
        sharding = NamedSharding(mesh, P(*named_axes))
        return jax.lax.with_sharding_constraint(state, sharding)

    return sharded_init


def slow_sharding_init(
    named_axes: Tuple[str, ...],
    mesh: Mesh,
    use_constant: bool = False,
):
    # This sharding_init runs slow, but it creates the tensor
    # on the sharding mesh directly, which requires less memory.
    # It is useful for really large tensor creation, like embedding.
    # The normal initializer cannot be used with int8 tensors.
    # The constant initializer should be used with int8 tensors.
    if use_constant:
        init_fn = nn.initializers.constant(0.5)
    else:
        init_fn = nn.initializers.normal()
    sharding = NamedSharding(mesh, P(*named_axes))
    sharded_init = jax.jit(init_fn,
                           static_argnums=(1, 2),
                           out_shardings=sharding)
    return sharded_init
