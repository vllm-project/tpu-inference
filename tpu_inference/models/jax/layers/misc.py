import math
from typing import Tuple

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


# TODO(xiang): move this to weight_utils.py
def shard_put(x: jax.Array, sharding_names: Tuple[str, ...] | P,
              mesh: jax.sharding.Mesh) -> jax.Array:
    # Single device sharding requires this special handling
    # to avoid the recursive jit error.
    if math.prod(mesh.axis_sizes) == 1:
        return jax.device_put(x, mesh.devices.flatten()[0])
    return jax.device_put(x, NamedSharding(mesh, P(*sharding_names)))
