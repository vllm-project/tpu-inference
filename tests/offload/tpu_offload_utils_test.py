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

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.offload.utils import (pre_update_kv_caches,
                                         pure_jax_stack_kv_cache_cross_layers,
                                         pure_jax_update_kv_caches_one,
                                         stack_kv_cache_cross_layers,
                                         update_kv_caches)


class TestTPUOffloadUtilsFn(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the tests."""
        self.num_layers = 64
        self.num_tokens = 8192
        num_devices = len(list(jax.devices()))
        self.num_kv_heads = num_devices
        self.head_dim = 128
        self.block_size = 128
        self.num_blocks = self.num_tokens // self.block_size
        self.cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            2,
            self.head_dim,
        )
        self.block_shape = (
            self.block_size,
            self.num_kv_heads,
            2,
            self.head_dim,
        )

        self.cache_dtype = jnp.bfloat16
        self.mesh = self.create_mesh((1, num_devices), ("data", "model"))
        self.partition_spec = PartitionSpec(None, None, "model")
        self.device_sharding = NamedSharding(self.mesh,
                                             self.partition_spec,
                                             memory_kind="device")
        self.host_sharding = NamedSharding(self.mesh,
                                           self.partition_spec,
                                           memory_kind="pinned_host")
        flatten_partition_spec = PartitionSpec(None, "model")
        self.flatten_device_sharding = NamedSharding(self.mesh,
                                                     flatten_partition_spec,
                                                     memory_kind="device")

        expand_partition_spec = PartitionSpec(None, None, None, "model")
        self.expand_device_sharding = NamedSharding(self.mesh,
                                                    expand_partition_spec,
                                                    memory_kind="device")
        self.replicated_device_sharding = NamedSharding(self.mesh,
                                                        PartitionSpec(),
                                                        memory_kind="device")

    def create_mesh(self, axis_shapes, axis_names):
        """Creates a JAX device mesh with the default device order."""
        try:
            num_required_devices = np.prod(axis_shapes)
            devices = np.array(jax.devices())
            if len(devices) < num_required_devices:
                self.skipTest(
                    f"Not enough devices to create mesh of shape {axis_shapes}."
                )
            device_array = devices[:num_required_devices].reshape(axis_shapes)
            return jax.sharding.Mesh(device_array, axis_names)
        except RuntimeError:
            return None

    def test_stack_kv_cache_cross_layers(self):
        """
        Verify stacking KV blocks across all the layers.
        """
        num_blocks_to_gather = 16
        src_blocks = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        ]

        # Create initial KV caches with unique data per block to verify gathering
        initial_kv_caches = []
        for i in range(self.num_layers):
            # Create data such that we can identify blocks.
            layer_data = jax.random.normal(jax.random.key(i),
                                           shape=self.cache_shape,
                                           dtype=self.cache_dtype)
            initial_kv_caches.append(
                jax.device_put(layer_data, self.device_sharding))

        jax.block_until_ready(initial_kv_caches)

        # Copy to host early because the operation will donate via stack_kv_cache_cross_layers
        initial_kv_caches_baseline = [
            np.array(cache) for cache in initial_kv_caches
        ]

        src_blocks_array = jnp.array(src_blocks)

        _, output = stack_kv_cache_cross_layers(initial_kv_caches,
                                                src_blocks_array,
                                                num_blocks_to_gather)
        jax.block_until_ready(output)

        # --- Verification ---
        self.assertEqual(len(output), num_blocks_to_gather)
        # Check that the stacked blocks equal to the original ones.

        for i in range(num_blocks_to_gather):
            # Shape of one element in stacked_blocks: (1, num_layers, block_size, num_kv_heads, 2, head_dim)
            self.assertEqual(output[i].shape[1], self.num_layers)
            self.assertEqual(output[i].shape[2:], self.cache_shape[1:])

            block_id = src_blocks[i]
            for j in range(self.num_layers):
                output_np = np.array(output[i])
                initial_np = np.array(initial_kv_caches_baseline[j])
                np.testing.assert_array_equal(output_np[0, j],
                                              initial_np[block_id])

        print(
            "\nTest passed: the gathered kv blocks equal to the original ones."
        )

    def test_update_kv_caches(self):
        """
        Verify update_kv_caches function.
        """
        num_blocks_to_update = 2
        update_indices = [1, 3]

        # Initial KV caches (zeros)
        initial_kv_caches = [
            jax.device_put(jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                           self.device_sharding)
            for _ in range(self.num_layers)
        ]

        # Create update data
        # stacked_blocks is a list of arrays. Each array corresponds to one block index being updated.
        # Shape of one element in stacked_blocks: (1, num_layers, block_size, num_kv_heads, 2, head_dim)
        stacked_blocks = [
            jax.device_put(
                jax.random.uniform(jax.random.PRNGKey(1),
                                   shape=(1, self.num_layers,
                                          *self.block_shape),
                                   dtype=self.cache_dtype),
                self.expand_device_sharding)
            for i in range(num_blocks_to_update)
        ]
        stacked_blocks_backup = [
            jnp.copy(stacked_block) for stacked_block in stacked_blocks
        ]
        jax.block_until_ready(stacked_blocks)
        jax.block_until_ready(stacked_blocks_backup)

        # Call the function
        src_offsets, dest_offsets, chunk_sizes, num_chunks = pre_update_kv_caches(
            update_indices, self.mesh, self.replicated_device_sharding)

        num_warmup = 2
        for _ in range(num_warmup):
            updated_caches = update_kv_caches(
                initial_kv_caches, stacked_blocks, src_offsets, dest_offsets,
                chunk_sizes, num_chunks, self.mesh, self.partition_spec,
                self.partition_spec, self.replicated_device_sharding.spec)

            jax.block_until_ready(updated_caches)
            initial_kv_caches = updated_caches

        updated_caches = update_kv_caches(initial_kv_caches, stacked_blocks,
                                          src_offsets, dest_offsets,
                                          chunk_sizes, num_chunks, self.mesh,
                                          self.partition_spec,
                                          self.partition_spec,
                                          self.replicated_device_sharding.spec)

        jax.block_until_ready(updated_caches)

        # Verification
        for layer_idx in range(self.num_layers):
            layer_cache = np.array(updated_caches[layer_idx])

            # Check updated blocks
            for i, block_idx in enumerate(update_indices):
                expected_block = stacked_blocks_backup[i][0, layer_idx]
                actual_block = layer_cache[block_idx]
                np.testing.assert_allclose(actual_block,
                                           expected_block,
                                           rtol=1e-2,
                                           atol=1e-2)

            # Check non-updated blocks (should be zero)
            for block_idx in range(self.num_blocks):
                if block_idx not in update_indices:
                    np.testing.assert_allclose(layer_cache[block_idx],
                                               0,
                                               rtol=1e-2,
                                               atol=1e-2)

        print("\nTest passed: update_kv_caches correctly updated the cache.")

    def test_pure_jax_stack_kv_cache_cross_layers(self):
        """
        Verify gathering and stacking KV blocks using pure_jax_stack_kv_cache_cross_layers.
        """
        num_blocks_to_gather = 3
        src_blocks = [2, 4, 6]

        initial_kv_caches = []
        for i in range(self.num_layers):
            layer_data = jax.random.normal(jax.random.key(i + 100),
                                           shape=self.cache_shape,
                                           dtype=self.cache_dtype)
            initial_kv_caches.append(
                jax.device_put(layer_data, self.device_sharding))

        jax.block_until_ready(initial_kv_caches)

        initial_kv_caches_baseline = [
            np.array(cache) for cache in initial_kv_caches
        ]
        src_blocks_array = jnp.array(src_blocks)

        kv_caches_out, output = pure_jax_stack_kv_cache_cross_layers(
            initial_kv_caches, src_blocks_array, num_blocks_to_gather)
        jax.block_until_ready(output)
        jax.block_until_ready(kv_caches_out)

        self.assertEqual(len(output), num_blocks_to_gather)
        for i in range(num_blocks_to_gather):
            self.assertEqual(output[i].shape[1], self.num_layers)
            self.assertEqual(output[i].shape[2:], self.cache_shape[1:])

            block_id = src_blocks[i]
            for j in range(self.num_layers):
                output_np = np.array(output[i])
                initial_np = np.array(initial_kv_caches_baseline[j])
                np.testing.assert_array_equal(output_np[0, j],
                                              initial_np[block_id])

        for j in range(self.num_layers):
            np.testing.assert_array_equal(np.array(kv_caches_out[j]),
                                          initial_kv_caches_baseline[j])

        print("\nTest passed: pure_jax_stack_kv_cache_cross_layers "
              "correctly gathered blocks and passed through kv_caches.")

    def test_pure_jax_update_kv_caches_one(self):
        """
        Verify pure_jax_update_kv_caches_one correctly scatters the blocks back into the KV cache.
        """
        num_blocks_to_update = 2
        update_indices = [2, 5]

        initial_kv_caches = [
            jax.device_put(jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                           self.device_sharding)
            for _ in range(self.num_layers)
        ]

        stacked_blocks = [
            jax.device_put(
                jax.random.uniform(jax.random.PRNGKey(i + 200),
                                   shape=(1, self.num_layers,
                                          *self.block_shape),
                                   dtype=self.cache_dtype),
                self.expand_device_sharding)
            for i in range(num_blocks_to_update)
        ]
        stacked_blocks_backup = [jnp.copy(sb) for sb in stacked_blocks]

        jax.block_until_ready(initial_kv_caches)
        jax.block_until_ready(stacked_blocks)
        jax.block_until_ready(stacked_blocks_backup)

        updated_caches = pure_jax_update_kv_caches_one(
            initial_kv_caches,
            stacked_blocks,
            update_indices,
            self.mesh,
            indices_sharding=jax.sharding.PartitionSpec())

        jax.block_until_ready(updated_caches)

        for layer_idx in range(self.num_layers):
            layer_cache = np.array(updated_caches[layer_idx])

            for i, block_idx in enumerate(update_indices):
                expected_block = np.array(stacked_blocks_backup[i])[0,
                                                                    layer_idx]
                actual_block = layer_cache[block_idx]
                np.testing.assert_allclose(actual_block,
                                           expected_block,
                                           rtol=1e-2,
                                           atol=1e-2)

            for block_idx in range(self.num_blocks):
                if block_idx not in update_indices:
                    np.testing.assert_allclose(layer_cache[block_idx],
                                               0,
                                               rtol=1e-2,
                                               atol=1e-2)

        print("\nTest passed: pure_jax_update_kv_caches_one "
              "correctly scattered the blocks.")

    def test_pure_jax_stack_kv_cache_cross_layers_single_block(self):
        """
        Edge Case: Verify gathering exactly 1 block.
        """
        num_blocks_to_gather = 1
        src_blocks = [5]

        initial_kv_caches = []
        for i in range(self.num_layers):
            layer_data = jax.random.normal(jax.random.key(i + 300),
                                           shape=self.cache_shape,
                                           dtype=self.cache_dtype)
            initial_kv_caches.append(
                jax.device_put(layer_data, self.device_sharding))

        jax.block_until_ready(initial_kv_caches)
        initial_kv_caches_baseline = [
            np.array(cache) for cache in initial_kv_caches
        ]
        src_blocks_array = jnp.array(src_blocks)

        kv_caches_out, output = pure_jax_stack_kv_cache_cross_layers(
            initial_kv_caches, src_blocks_array, num_blocks_to_gather)
        jax.block_until_ready(output)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape[1], self.num_layers)

        for j in range(self.num_layers):
            output_np = np.array(output[0])
            initial_np = np.array(initial_kv_caches_baseline[j])
            np.testing.assert_array_equal(output_np[0, j], initial_np[5])

    def test_pure_jax_update_kv_caches_one_out_of_order(self):
        """
        Edge Case: Verify pure_jax_update_kv_caches_one scatters correctly when indices are out of order.
        """
        num_blocks_to_update = 2
        update_indices = [5, 2]  # Out of order!

        initial_kv_caches = [
            jax.device_put(jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                           self.device_sharding)
            for _ in range(self.num_layers)
        ]

        stacked_blocks = [
            jax.device_put(
                jax.random.uniform(jax.random.PRNGKey(i + 400),
                                   shape=(1, self.num_layers,
                                          *self.block_shape),
                                   dtype=self.cache_dtype),
                self.expand_device_sharding)
            for i in range(num_blocks_to_update)
        ]
        stacked_blocks_backup = [jnp.copy(sb) for sb in stacked_blocks]
        jax.block_until_ready(stacked_blocks)

        updated_caches = pure_jax_update_kv_caches_one(
            initial_kv_caches,
            stacked_blocks,
            update_indices,
            self.mesh,
            indices_sharding=jax.sharding.PartitionSpec())
        jax.block_until_ready(updated_caches)

        for layer_idx in range(self.num_layers):
            layer_cache = np.array(updated_caches[layer_idx])
            # Check block 5 is stacked_blocks[0]
            np.testing.assert_allclose(
                layer_cache[5],
                np.array(stacked_blocks_backup[0])[0, layer_idx],
                rtol=1e-2,
                atol=1e-2)
            # Check block 2 is stacked_blocks[1]
            np.testing.assert_allclose(
                layer_cache[2],
                np.array(stacked_blocks_backup[1])[0, layer_idx],
                rtol=1e-2,
                atol=1e-2)

    def test_pure_jax_update_kv_caches_one_sharding_types(self):
        """
        Edge Case: Verify pure_jax_update_kv_caches_one accepts None, PartitionSpec, and NamedSharding for indices_sharding.
        """
        update_indices = [1]

        def get_fresh_inputs():
            caches = [
                jax.device_put(
                    jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                    self.device_sharding) for _ in range(self.num_layers)
            ]
            blocks = [
                jax.device_put(
                    jnp.ones((1, self.num_layers, *self.block_shape),
                             dtype=self.cache_dtype),
                    self.expand_device_sharding)
            ]
            return caches, blocks

        # Test None
        c1, b1 = get_fresh_inputs()
        out_none = pure_jax_update_kv_caches_one(c1,
                                                 b1,
                                                 update_indices,
                                                 self.mesh,
                                                 indices_sharding=None)

        # Test PartitionSpec
        c2, b2 = get_fresh_inputs()
        out_ps = pure_jax_update_kv_caches_one(
            c2,
            b2,
            update_indices,
            self.mesh,
            indices_sharding=jax.sharding.PartitionSpec())

        # Test NamedSharding
        c3, b3 = get_fresh_inputs()
        named_sharding = NamedSharding(self.mesh,
                                       jax.sharding.PartitionSpec(),
                                       memory_kind='device')
        out_ns = pure_jax_update_kv_caches_one(c3,
                                               b3,
                                               update_indices,
                                               self.mesh,
                                               indices_sharding=named_sharding)

        jax.block_until_ready([out_none, out_ps, out_ns])

        # All should have successfully scattered 1 into index 1
        np.testing.assert_allclose(np.array(out_none[0])[1],
                                   1.0,
                                   rtol=1e-2,
                                   atol=1e-2)
        np.testing.assert_allclose(np.array(out_ps[0])[1],
                                   1.0,
                                   rtol=1e-2,
                                   atol=1e-2)
        np.testing.assert_allclose(np.array(out_ns[0])[1],
                                   1.0,
                                   rtol=1e-2,
                                   atol=1e-2)


if __name__ == "__main__":
    unittest.main()
