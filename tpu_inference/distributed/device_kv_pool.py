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

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class DeviceKVPool:

    def __init__(self, pool_size: int, num_layers: int,
                 max_blocks_per_req: int, cache_inner_shape: tuple,
                 dtype: jnp.dtype,
                 device_sharding: jax.sharding.NamedSharding):
        """
        Pre-allocates a pool of TPU HBM buffers to eliminate per-request
        device array allocation overhead during H2D KV transfer.
        """
        logger.info(
            f"Initializing DeviceKVPool with pool_size={pool_size}, num_layers={num_layers}, "
            f"max_blocks_per_req={max_blocks_per_req}, cache_inner_shape={cache_inner_shape}, "
            f"dtype={dtype}, device_sharding={device_sharding}")
        self.pool_size = pool_size
        self.max_blocks_per_req = max_blocks_per_req
        self.buffers: List[List[jax.Array]] = []

        self.available_indices = queue.Queue(maxsize=pool_size)

        # e.g., (1024, 16, 128) -> (max_blocks, num_heads, head_size)
        layer_buffer_shape = (max_blocks_per_req, ) + cache_inner_shape

        logger.info(f"Allocating {pool_size} HBM buffers for device KV pool.")
        start_time = time.perf_counter()

        for i in range(pool_size):
            layer_buffers = [
                jax.device_put(
                    jnp.zeros(layer_buffer_shape, dtype=dtype),
                    device_sharding)
                for _ in range(num_layers)
            ]
            self.buffers.append(layer_buffers)
            self.available_indices.put(i)

        end_time = time.perf_counter()
        logger.info(
            f"Device HBM KV pool allocation complete. Time taken: {end_time - start_time:.2f} seconds."
        )

    def get_buffer(self,
                   block: bool = True,
                   timeout: float = None) -> Tuple[int, List[jax.Array]]:
        """
        Checks out an exclusive buffer from the pool.
        Blocks if the pool is empty until another thread calls return_buffer().
        """
        is_empty = self.available_indices.empty()
        if is_empty:
            logger.warning(
                "DeviceKVPool available_indices is empty. Waiting for a buffer to be returned..."
            )
            start_wait = time.perf_counter()

        idx = self.available_indices.get(block=block, timeout=timeout)

        if is_empty:
            wait_time = time.perf_counter() - start_wait
            logger.info(
                f"DeviceKVPool acquired buffer after waiting for {wait_time * 1000:.2f} ms."
            )

        return idx, self.buffers[idx]

    def return_buffer(self, idx: int,
                      updated_buffer: List[jax.Array] = None):
        """
        Returns the buffer to the pool so other requests can use it.
        Pass updated_buffer if JAX donated/replaced the array reference.
        """
        logger.info(f"Returning buffer to DeviceKVPool (idx={idx})")

        if updated_buffer is not None:
            self.buffers[idx] = updated_buffer

        self.available_indices.put(idx)
