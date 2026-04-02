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

import os
import gc
import unittest
import jax
import time
import jax.numpy as jnp
import numpy as np
import random


opts = jax.config.jax_pjrt_client_create_options or ''

if isinstance(opts, str):
    # Append the recycle mode to the semicolon-separated string
    opts = f'{opts};pinned_host_allocation_mode:recycle' if opts else 'pinned_host_allocation_mode:recycle'
elif isinstance(opts, dict):
    opts['pinned_host_allocation_mode'] = 'recycle'

# Update the configuration to instruct the TPU plugin to recycle premapped buffers
jax.config.update('jax_pjrt_client_create_options', opts)

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src import compilation_cache as cc
import jax.numpy as jnp
import numpy as np
from tpu_inference.offload.kv_copy import kv_copy

from jax.sharding import NamedSharding, PartitionSpec
from absl.testing import parameterized

P = jax.sharding.PartitionSpec

class AsyncCopyPerfTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        # Force JAX to release memory
        cc.reset_cache()
        jax.clear_caches()
        # Force Python GC
        gc.collect()

    def _create_mesh(self, axis_shapes, axis_names):
        """Creates a JAX device mesh with the default device order."""
        try:
            num_required_devices = np.prod(axis_shapes)
            devices = np.array(jax.devices())
            if len(devices) < num_required_devices:
                self.skipTest(f"Not enough devices to create mesh of shape {axis_shapes}.")
                return None
            device_array = devices[:num_required_devices].reshape(axis_shapes)
            return jax.sharding.Mesh(device_array, axis_names)
        except RuntimeError:
            return None

    def _create_single_layer_kv_cache(self, cache_shape, cache_dtype, cache_sharding):
        def _allocate():
            return jax.random.uniform(jax.random.key(1), shape=cache_shape, dtype=cache_dtype)
        sharded_allocate = jax.jit(_allocate, out_shardings=cache_sharding)
        return sharded_allocate()

    def _create_single_layer_kv_cache_zeros(self, cache_shape, cache_dtype, cache_sharding):
        def _allocate():
            return jnp.zeros(shape=cache_shape, dtype=cache_dtype)
        sharded_allocate = jax.jit(_allocate, out_shardings=cache_sharding)
        return sharded_allocate()

    @parameterized.named_parameters(
        dict(
            testcase_name="2",
            kv_slices_shape=(16, 128, 8, 2, 128),
            num_layers=64, 
            dtype=jnp.bfloat16,
        ),
    )
    def test_d2h(self, kv_slices_shape, num_layers, dtype):
        num_blocks, block_size, num_heads, kv_div, head_dim = kv_slices_shape
        num_devices = jax.device_count()
        mesh = self._create_mesh((1, num_devices), ('data', 'model'))
        if mesh is None:
            self.skipTest("Cannot create mesh. Must be run on a TPU node.")
            return

        partition_spec = PartitionSpec(None, None, 'model')
        device_sharding = NamedSharding(mesh, partition_spec)
        host_sharding = NamedSharding(mesh, partition_spec, memory_kind="pinned_host")


        device_kv_slices = [self._create_single_layer_kv_cache(kv_slices_shape, dtype, device_sharding) for i in range(num_layers)]
        for x in device_kv_slices:
            x.block_until_ready()

        host_kv_slices = [self._create_single_layer_kv_cache_zeros(kv_slices_shape, dtype, host_sharding) for i in range(num_layers)]
        for x in host_kv_slices:
            x.block_until_ready()

        _host_kv_slices = kv_copy.copy_to_dest(device_kv_slices, host_kv_slices, mesh, partition_spec, host_sharding, "pinned_host", dtype)
        for x in _host_kv_slices:
            x.block_until_ready()
        for x, y in zip(device_kv_slices, _host_kv_slices):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(y.sharding.memory_kind, "pinned_host")

        num_warmup = 5
        host_allocated_data = []
        for i in range(num_warmup):
            host_kv_slices = [self._create_single_layer_kv_cache_zeros(kv_slices_shape, dtype, host_sharding) for i in range(num_layers)]
            for x in host_kv_slices:
                x.block_until_ready()
            host_allocated_data.append(host_kv_slices)

        for i in range(num_warmup):
            _host_kv_slices = kv_copy.copy_to_dest(device_kv_slices, host_allocated_data[i], mesh, partition_spec, host_sharding, "pinned_host", dtype)
            jax.block_until_ready(_host_kv_slices)

        num_profile = 5
        host_allocated_data = []
        for i in range(num_profile):
            host_kv_slices = [self._create_single_layer_kv_cache_zeros(kv_slices_shape, dtype, host_sharding) for i in range(num_layers)]
            for x in host_kv_slices:
                x.block_until_ready()
            host_allocated_data.append(host_kv_slices)

        print('FHZ: Starting profiled run...', flush=True)

        exp_name = f"d2h-copy-test-5/"
        profile_dir = f"/mnt/disks/jcgu/code/ullm/rebase/offload_tests/{exp_name}"
        options = jax.profiler.ProfileOptions()
        # default: https://docs.jax.dev/en/latest/profiling.html#general-options
        options.python_tracer_level = 1
        options.host_tracer_level = os.getenv("HOST_TRACER_LEVEL", 3)
        jax.profiler.start_trace(profile_dir, profiler_options=options)
        for i in range(num_profile):
            _host_kv_slices = kv_copy.copy_to_dest(device_kv_slices, host_allocated_data[i], mesh, partition_spec, host_sharding, "pinned_host", dtype)
            jax.block_until_ready(_host_kv_slices)

        jax.profiler.stop_trace()


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())
