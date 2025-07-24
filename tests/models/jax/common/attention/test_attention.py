import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.attention import (Attention,
                                                               AttentionConfig)
from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.sharding import (ShardingConfig,
                                                    ShardingRulesConfig)

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
        self.sharding_cfg = ShardingConfig(
            default_rules_cls=ShardingRulesConfig,
            prefill_rules=ShardingRulesConfig,
            generate_rules=ShardingRulesConfig,
        )

    def test_attention_forward_pass(self):
        """Tests the forward pass of the Attention module in prefill mode."""
        hidden_size = 1024
        num_attention_heads = 8
        head_dim = hidden_size // num_attention_heads

        attention_config = AttentionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=head_dim,
            rope_theta=10000.0,
            rope_scaling={},
            dtype=jnp.bfloat16,
            vllm_config=None,
        )

        attention = Attention(
            cfg=attention_config,
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.sharding_cfg,
            quant=None,
        )
        attention.generate_kernel(nnx.Rngs(42))

        seq_len = 64
        x = jnp.ones((seq_len, attention_config.hidden_size),
                     dtype=jnp.bfloat16)

        block_size = 16
        num_blocks = 8
        cache_shape = (
            num_blocks,
            block_size,
            attention_config.num_key_value_heads * 2,
            attention_config.head_dim,
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
        )

        new_kv_cache, output = attention(
            x,
            is_prefill=True,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata,
        )

        self.assertEqual(output.shape, (seq_len, attention_config.hidden_size))

        self.assertEqual(new_kv_cache.shape, kv_cache.shape)


if __name__ == "__main__":
    unittest.main()
