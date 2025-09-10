import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.attention import get_kv_cache_shape
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.attention import Attention

KVCache = Tuple[jax.Array, jax.Array]


class TestAttention(unittest.TestCase):
    """Unit test suite for the JAX Attention module."""

    def setUp(self):
        """Sets up the testing environment before each test."""
        self.mesh = Mesh(
            np.array(jax.devices()).reshape(1, -1),
            axis_names=(
                "expert",
                "model",
            ),
        )

    def test_attention_forward_pass(self):
        """Tests the forward pass of the Attention module in prefill mode."""
        hidden_size = 1024
        num_attention_heads = 8
        head_dim = hidden_size // num_attention_heads

        with jax.set_mesh(self.mesh):
            attention = Attention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_attention_heads,
                head_dim=head_dim,
                rope_theta=10000.0,
                rope_scaling={},
                dtype=jnp.bfloat16,
                mesh=self.mesh,
                random_init=True,
                rngs=nnx.Rngs(42),
            )

            seq_len = 64
            x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)

            block_size = 16
            num_blocks = 8
            kv_dtype = jnp.bfloat16
            cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                             num_attention_heads, head_dim,
                                             kv_dtype)

            kv_cache = jnp.zeros(cache_shape, dtype=kv_dtype)

            num_required_blocks = seq_len // block_size

            attention_metadata = AttentionMetadata(
                input_positions=jnp.arange(seq_len, dtype=jnp.int32),
                block_tables=jnp.array(list(range(num_required_blocks)),
                                       dtype=jnp.int32),
                seq_lens=jnp.array([seq_len], dtype=jnp.int32),
                query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
                request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
            )

            new_kv_cache, output = attention(
                x,
                is_prefill=True,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata,
            )

            self.assertEqual(output.shape, (seq_len, hidden_size))

            self.assertEqual(new_kv_cache.shape, kv_cache.shape)


if __name__ == "__main__":
    unittest.main()
