import functools
import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.offload.utils import (get_kv_cache_swap_fn,
                                         jitted_insert_kv_cache_slices)


class TestTPUOffloadUtilsFn(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the tests."""
        self.num_layers = 2
        self.num_tokens = 256
        num_devices = len(list(jax.devices()))
        self.num_kv_heads = num_devices
        self.head_dim = 128
        self.block_size = 16
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
        partition_spec = PartitionSpec(None, None, "model")
        self.device_sharding = NamedSharding(self.mesh,
                                             partition_spec,
                                             memory_kind="device")
        self.host_sharding = NamedSharding(self.mesh,
                                           partition_spec,
                                           memory_kind="pinned_host")
        flatten_partition_spec = PartitionSpec(None, "model")
        self.flatten_device_sharding = NamedSharding(self.mesh,
                                                     flatten_partition_spec,
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

    def test_jitted_insert_kv_cache_slices_equivalence(self):
        """
        Verify inserting scattered kv slices / pages into the large kv cache.
        """
        num_blocks_to_insert = 3
        dst_blocks = [3, 5, 7]
        dst_blocks_array = jnp.array(dst_blocks)

        initial_kv_caches = [
            jax.device_put(jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                           self.device_sharding)
            for _ in range(self.num_layers)
        ]

        # The raw, chunked KV data (input for the new method)
        # This is a list of lists: List[layer -> List[block]]
        raw_chunked_kv = []
        for i in range(self.num_layers):
            layer_chunks = [
                jax.device_put(
                    jax.random.normal(jax.random.key(i),
                                      shape=self.block_shape,
                                      dtype=self.cache_dtype),
                    self.flatten_device_sharding)
                for _ in range(num_blocks_to_insert)
            ]
            raw_chunked_kv.append(layer_chunks)

        output = jitted_insert_kv_cache_slices(self.block_size,
                                               initial_kv_caches,
                                               raw_chunked_kv,
                                               dst_blocks_array)

        # --- Verification ---
        # Check that the selected pages for each layer equal to the original ones.
        for i in range(self.num_layers):
            for j in range(num_blocks_to_insert):
                block_id = dst_blocks[j]
                np.testing.assert_array_equal(np.array(output[i][block_id]),
                                              raw_chunked_kv[i][j])
            print("\nTest passed: the inserted kv equals to the original one.")

    def test_swap_fn_correctness(self):
        """
        Verify that swap-out and swap-in functions work correctly for different
        swap_op_types and jitted options.
        """
        swap_op_types = ["jax", "pallas"]
        jitted_options = [True, False]

        # NOTE(jcgu): we are using the entire kv cache [n_b, bs, nh, 2, hd],
        # actually, we will operate on concatenated blocks [nt, nh, 2, hd];
        @functools.partial(jax.jit, out_shardings=self.device_sharding)
        def create_on_device(key):
            return jax.random.uniform(key,
                                      shape=self.cache_shape,
                                      dtype=self.cache_dtype)

        initial_kv_caches = [
            create_on_device(jax.random.key(i)) for i in range(self.num_layers)
        ]
        jax.block_until_ready(initial_kv_caches)

        for swap_op_type, jitted in itertools.product(swap_op_types,
                                                      jitted_options):
            with self.subTest(swap_op_type=swap_op_type, jitted=jitted):
                swap_in_fn, swap_out_fn = get_kv_cache_swap_fn(
                    swap_op_type, self.host_sharding, self.device_sharding,
                    jitted)

                # Put initial data on device
                device_kv_caches = jax.device_put(initial_kv_caches,
                                                  self.device_sharding)
                jax.block_until_ready(device_kv_caches)

                # Swap out to host
                host_kv_caches = swap_out_fn(device_kv_caches)

                # Swap back in to device
                final_device_kv_caches = swap_in_fn(host_kv_caches)
                jax.block_until_ready(final_device_kv_caches)

                # Verify correctness
                for i in range(self.num_layers):
                    np.testing.assert_array_equal(
                        np.array(initial_kv_caches[i]),
                        np.array(final_device_kv_caches[i]))


if __name__ == "__main__":
    unittest.main()
