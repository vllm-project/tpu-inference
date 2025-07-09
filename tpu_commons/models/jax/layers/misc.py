from typing import Tuple

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


# TODO(xiang): move this to weight_utils.py
def shard_put(x: jax.Array, sharding_names: Tuple[str, ...] | P,
              mesh: jax.sharding.Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P(*sharding_names)))
