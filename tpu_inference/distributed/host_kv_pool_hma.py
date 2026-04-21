# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import time
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.distributed.host_kv_pool import HostKVPool
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class HostKVPoolHMA(HostKVPool):
    """Host KV Pool supporting HMA.
    
    Each buffer holds a flat list of kv cache jax arrays across all
    layers. A Mamba layer's tuple (ssm + conv state) will be expanded
    into individual entries in the flat list.
    """

    def __init__(self, pool_size: int, per_array_max_blocks: List[int],
                 per_array_inner_shape: List[Tuple],
                 per_array_dtype: List[jnp.dtype],
                 per_array_host_sharding: List[jax.sharding.NamedSharding]):
        # Skip HostKVPool.__init__: it allocates exactly one kv cache jax
        # array per layer. HMA allocates a variable number per layer (e.g.
        # two arrays for a Mamba layer, ssm and conv state, and one for an
        # attention layer).

        # Assert the number of kv cache arrays are the same.
        assert (len(per_array_max_blocks) == len(per_array_inner_shape) ==
                len(per_array_dtype) == len(per_array_host_sharding))

        # Allocate the pool
        self.pool_size = pool_size
        self.available_indices = queue.Queue(maxsize=pool_size)
        self.buffers: List[List[jax.Array]] = []
        per_array_shapes = [
            (n, ) + tuple(inner)
            for n, inner in zip(per_array_max_blocks, per_array_inner_shape)
        ]
        num_kv_arrays = len(per_array_shapes)
        bytes_per_buffer = sum(
            int(np.prod(shape)) * jnp.dtype(dt).itemsize
            for shape, dt in zip(per_array_shapes, per_array_dtype))
        start = time.perf_counter()
        for i in range(pool_size):
            buffer = self._create_one_buffer(per_array_shapes, per_array_dtype,
                                             per_array_host_sharding)
            self.buffers.append(buffer)
            self.available_indices.put(i)
        end = time.perf_counter()
        logger.info(f"HostKVPoolHMA --> allocated | pool_size={pool_size} | "
                    f"num_kv_arrays={num_kv_arrays} | "
                    f"bytes_per_buffer={bytes_per_buffer} | "
                    f"total_bytes={pool_size * bytes_per_buffer} | "
                    f"elapsed_s={end - start:.2f}")

    def _create_one_buffer(
        self,
        per_array_shapes: List[Tuple[int, ...]],
        per_array_dtype: List[jnp.dtype],
        per_array_host_sharding: List[jax.sharding.NamedSharding],
    ) -> List[jax.Array]:
        host_zeros = [
            np.zeros(shape, dtype=dtype)
            for shape, dtype in zip(per_array_shapes, per_array_dtype)
        ]
        return jax.device_put(host_zeros, per_array_host_sharding)
