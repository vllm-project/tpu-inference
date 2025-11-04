import functools

import jax

from .pooler import Pooler, PoolerOutput
from .pooling_metadata import TPUSupportedPoolingMetadata

@jax.jit
def pool(
    hidden_states: jax.Array,
    pooling_metadata: TPUSupportedPoolingMetadata,
    pooler: Pooler,
) -> PoolerOutput:
    return pooler(hidden_states, pooling_metadata)
