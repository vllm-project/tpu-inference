import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.distributed.cache_util import jitted_insert_kv_cache_slices


def original_jitted_insert_kv_cache_slices(
    block_size,
    kv_caches: list[jax.Array],
    kv_cache_slices: list[jax.Array],
    block_numbers: jax.Array,
) -> list[jax.Array]:
    """
    This is the original implementation that expects concatenated slices.
    It reshapes the single slice per layer into multiple blocks.
    """

    def _update_layer(cache, slices):
        """The function to apply to each layer's cache and slices."""
        # Original method reshapes a large slice into blocks
        num_blocks = len(block_numbers)
        reshaped_slices = slices.reshape(
            (num_blocks, 1, block_size, *slices.shape[1:]))
        for i, block_idx in enumerate(block_numbers):
            cache = jax.lax.dynamic_update_slice_in_dim(cache,
                                                        reshaped_slices[i],
                                                        block_idx,
                                                        axis=0)
        return cache

    return jax.tree.map(_update_layer, kv_caches, kv_cache_slices)


class TestCacheInsertion(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the tests."""
        self.num_layers = 2
        self.num_blocks_total = 32
        self.block_size = 16
        self.num_kv_heads = 4
        self.head_dim = 128

        # We will load 3 new blocks
        self.num_blocks_to_load = 3

        # Shape for a single block in the main KV cache
        self.kv_cache_shape = (
            self.num_blocks_total,
            self.block_size,
            self.num_kv_heads,
            2,
            self.head_dim,
        )

        # Shape for one chunk/slice of tokens to be inserted
        self.slice_shape = (
            self.block_size,
            self.num_kv_heads,
            2,
            self.head_dim,
        )

        # Destination block indices in the main KV cache
        self.dst_blocks = jnp.array([5, 12, 21])

        # --- Test Data ---

        # 1. Initial (empty) KV caches for all layers
        key = jax.random.PRNGKey(0)
        self.initial_kv_caches1 = [
            jnp.zeros(self.kv_cache_shape, dtype=jnp.float32)
            for _ in range(self.num_layers)
        ]

        self.initial_kv_caches2 = [
            jnp.zeros(self.kv_cache_shape, dtype=jnp.float32)
            for _ in range(self.num_layers)
        ]

        # 2. The raw, chunked KV data (input for the new method)
        # This is a list of lists: List[layer -> List[chunk]]
        self.raw_chunked_kv = []
        for _ in range(self.num_layers):
            key, subkey = jax.random.split(key)
            layer_chunks = [
                jax.random.normal(subkey, self.slice_shape)
                for _ in range(self.num_blocks_to_load)
            ]
            self.raw_chunked_kv.append(layer_chunks)

        # 3. The concatenated KV data (input for the original method)
        # This is a list of arrays: List[layer -> concatenated_array]
        self.concatenated_kv = [
            jax.lax.concatenate(layer_chunks, dimension=0)
            for layer_chunks in self.raw_chunked_kv
        ]

    def test_jitted_insert_kv_cache_slices_equivalence(self):
        """
        Verify that the new and original methods for inserting KV cache slices
        produce identical results.
        """
        # --- Approach 1: Original Method ---
        # This method takes concatenated slices.
        original_output = original_jitted_insert_kv_cache_slices(
            self.block_size, self.initial_kv_caches1, self.concatenated_kv,
            self.dst_blocks)

        # --- Approach 2: New Method ---
        # This method takes a list of chunked slices.
        new_output = jitted_insert_kv_cache_slices(self.block_size,
                                                   self.initial_kv_caches2,
                                                   self.raw_chunked_kv,
                                                   self.dst_blocks)

        # --- Verification ---
        # Check that the outputs for each layer are identical.
        for i in range(self.num_layers):
            np.testing.assert_array_equal(np.array(original_output[i]),
                                          np.array(new_output[i]))
        print("\nTest passed: Both methods produce identical KV caches.")


if __name__ == "__main__":
    unittest.main()
