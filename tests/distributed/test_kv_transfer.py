# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Performance tests for async copy on TPU."""

import gc
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import compilation_cache as cc
from jax._src import test_util as jtu

from tpu_inference.distributed.kv_transfer import multi_layer_copy

P = jax.sharding.PartitionSpec


def create_mesh(axis_shapes, axis_names, explicit_axis: bool = False):
    """Creates a JAX device mesh with the default device order."""
    try:
        num_required_devices = np.prod(axis_shapes)
        devices = np.array(jax.devices())
        if len(devices) < num_required_devices:
            print(
                f'Expected at least {num_required_devices} devices, but only found'
                f' {len(devices)}. This script requires more devices.')
            return None

        device_array = devices[:num_required_devices].reshape(axis_shapes)
        axis_types = (tuple([jax.sharding.AxisType.Explicit] *
                            len(axis_shapes)) if explicit_axis else None)
        return jax.sharding.Mesh(device_array,
                                 axis_names,
                                 axis_types=axis_types)
    except RuntimeError:
        print('No TPU devices found. This script must be run on a TPU node.')
        return None


def create_single_layer_kv_cache(
    cache_shape: Tuple,
    cache_dtype: jnp.dtype,
    cache_sharding: jax.sharding.NamedSharding,
    init_zeros: bool = False,
) -> jax.Array:
    """Creates a single layer KV cache.

  Args:
    cache_shape: The shape of the cache.
    cache_dtype: The dtype of the cache.
    cache_sharding: The sharding specification for the cache.
    init_zeros: If True, initialize the cache with zeros. Otherwise, use random
      uniform values.

  Returns:
    A sharded jax.Array representing the KV cache.
  """

    def _allocate() -> jax.Array:
        if init_zeros:
            return jnp.zeros(shape=cache_shape, dtype=cache_dtype)
        return jax.random.uniform(jax.random.key(1),
                                  shape=cache_shape,
                                  dtype=cache_dtype)

    sharded_allocate = jax.jit(_allocate, out_shardings=cache_sharding)
    return sharded_allocate()


class KVTransferTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        cc.reset_cache()
        jax.clear_caches()

        # Force Python GC
        gc.collect()

    @parameterized.named_parameters(
        dict(
            testcase_name='single_layer_single_block',
            kv_shape=(1024, 128, 8, 2, 64),
            num_layers=1,
            src_offsets_list=[0],
            dest_offsets_list=[0],
            chunk_sizes_list=[1],
            dtype=jnp.bfloat16,
        ),
        dict(
            testcase_name='padded_offsets_array',
            kv_shape=(1024, 128, 8, 2, 64),
            num_layers=4,
            src_offsets_list=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17],
            dest_offsets_list=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17],
            chunk_sizes_list=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dtype=jnp.float32,
            num_chunks=2,
        ),
        dict(
            testcase_name='padded_offsets_array_middle_blocks',
            kv_shape=(1024, 128, 8, 2, 64),
            num_layers=4,
            src_offsets_list=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17],
            dest_offsets_list=[567, 32, 10, 11, 12, 13, 14, 15, 16, 17],
            chunk_sizes_list=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dtype=jnp.bfloat16,
            num_chunks=2,
        ),
        dict(
            testcase_name='multi_layer_variable_block',
            kv_shape=(1024, 128, 8, 2, 128),
            num_layers=10,
            src_offsets_list=[0, 1, 3, 6],
            dest_offsets_list=[0, 10, 20, 30],
            chunk_sizes_list=[1, 2, 3, 4],
            dtype=jnp.bfloat16,
        ),
    )
    def test_async_copy(
        self,
        kv_shape,
        num_layers,
        src_offsets_list,
        dest_offsets_list,
        chunk_sizes_list,
        dtype,
        num_chunks=None,
    ):
        num_devices = jax.device_count()
        axis_shapes = (1, num_devices)
        axis_names = ('data', 'model')
        explicit_sharding = False
        mesh = create_mesh(axis_shapes,
                           axis_names,
                           explicit_axis=explicit_sharding)
        partition_spec = jax.sharding.PartitionSpec(None, None, 'model')
        device_sharding = jax.sharding.NamedSharding(mesh,
                                                     partition_spec,
                                                     memory_kind='device')
        replicated_sharding = jax.sharding.NamedSharding(mesh,
                                                         P(),
                                                         memory_kind='device')

        _, bs, nh, kv_div, hd = kv_shape
        src_offsets = jnp.array(src_offsets_list, dtype=jnp.int32)
        dest_offsets = jnp.array(dest_offsets_list, dtype=jnp.int32)
        chunk_sizes = jnp.array(chunk_sizes_list, dtype=jnp.int32)
        if num_chunks is None:
            num_chunks = len(chunk_sizes_list)
        num_copy_elements = sum(chunk_sizes_list)

        poison_dest_offsets = []
        if num_chunks < len(dest_offsets_list):
            poison_dest_offsets = dest_offsets_list[num_chunks:]

        src_offsets = jax.device_put(src_offsets, replicated_sharding)
        dest_offsets = jax.device_put(dest_offsets, replicated_sharding)
        chunk_sizes = jax.device_put(chunk_sizes, replicated_sharding)
        num_chunks_array = jax.device_put(
            jnp.array([num_chunks], dtype=jnp.int32), replicated_sharding)

        kv_caches = [
            create_single_layer_kv_cache(
                kv_shape,
                dtype,
                device_sharding,
                init_zeros=True,
            ) for _ in range(num_layers)
        ]
        jax.block_until_ready(kv_caches)

        num_copy_elements = int(num_copy_elements)
        src_kv_shape = (num_copy_elements, bs, nh, kv_div, hd)
        src_kv_blocks = [
            create_single_layer_kv_cache(
                src_kv_shape,
                dtype,
                device_sharding,
                init_zeros=False,
            ) for _ in range(num_layers)
        ]
        jax.block_until_ready(src_kv_blocks)

        y = multi_layer_copy(
            src_array=src_kv_blocks,
            dest_array=kv_caches,
            src_offsets=src_offsets,
            dest_offsets=dest_offsets,
            chunk_sizes=chunk_sizes,
            num_chunks=num_chunks_array,
            mesh=mesh,
            src_sharding_spec=partition_spec,
            dest_sharding_spec=partition_spec,
            replicated_sharding_spec=P(),
        )
        for _y in y:
            _y.block_until_ready()

        self.assertEqual(y[0].sharding.memory_kind, 'device')

        # Verify correctness.
        for layer_idx in range(num_layers):
            for i in range(num_chunks):
                s_off = int(src_offsets[i])
                d_off = int(dest_offsets[i])
                c_size = int(chunk_sizes[i])
                actual = np.asarray(y[layer_idx])[d_off:d_off + c_size]
                desired = np.asarray(src_kv_blocks[layer_idx])[s_off:s_off +
                                                               c_size]
                np.testing.assert_array_equal(actual, desired)
            if poison_dest_offsets:
                for p_d_off in poison_dest_offsets:
                    np.testing.assert_array_equal(
                        np.asarray(y[layer_idx])[p_d_off], 0)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())
