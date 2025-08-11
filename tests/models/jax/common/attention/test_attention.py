import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.attention import Attention
from tpu_commons.models.jax.common.base import ParamFactory

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
        self.param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=True,
        )

    def test_attention_forward_pass(self):
        """Tests the forward pass of the Attention module in prefill mode."""
        hidden_size = 1024
        num_attention_heads = 8
        head_dim = hidden_size // num_attention_heads

        dummy_sharding = NamedSharding(self.mesh, P())

        attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=head_dim,
            rope_theta=10000.0,
            rope_scaling={},
            dtype=jnp.bfloat16,
            mesh=self.mesh,
            param_factory=self.param_factory,
            quant=None,
            dnh_sharding=dummy_sharding,
            dkh_sharding=dummy_sharding,
            nhd_sharding=dummy_sharding,
            activation_q_td=dummy_sharding,
            query_tnh=dummy_sharding,
            keyvalue_skh=dummy_sharding,
            keyvalue_cache_lskh=dummy_sharding,
            attn_o_tnh=dummy_sharding,
        )
        attention.generate_kernel(nnx.Rngs(42))

        seq_len = 64
        x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)

        block_size = 16
        num_blocks = 8
        cache_shape = (
            num_blocks,
            block_size,
            num_attention_heads * 2,
            head_dim,
        )

        kv_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

        num_required_blocks = seq_len // block_size
        num_slices = 8

        slot_mapping = jnp.zeros((3, num_slices), dtype=jnp.int32)
        slot_mapping = slot_mapping.at[:, 0].set(jnp.array([0, 0, seq_len]))

        attention_metadata = AttentionMetadata(
            input_positions=jnp.arange(seq_len, dtype=jnp.int32),
            slot_mapping=slot_mapping,
            block_tables=jnp.array([list(range(num_required_blocks))],
                                   dtype=jnp.int32),
            seq_lens=jnp.array([seq_len], dtype=jnp.int32),
            num_seqs=jnp.array([1], dtype=jnp.int32),
            query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
            num_slices=jnp.array([1], dtype=jnp.int32),
            request_distribution=None,
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
