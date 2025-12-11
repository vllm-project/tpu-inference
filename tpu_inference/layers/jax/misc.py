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

def round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without
    exceeding the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater
    than or equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest
    multiple of 128 less than or equal to `limit` (down to 512) that divides `x`
    evenly, and returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128).

    Returns:
        int: The rounded value according to the rules above.

    Raises:
        AssertionError: If `limit` is less than 128 or not a multiple of 128.
    """
    assert limit >= 128 and limit % 128 == 0
    if x <= 128:
        return 128
    if x < limit:
        return (x + 127) // 128 * 128
    for candidate in range(limit, 511, -128):
        if x % candidate == 0:
            return candidate
    return limit