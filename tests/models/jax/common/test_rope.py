from jax import numpy as jnp
from jax._src import test_util as jtu

from tpu_commons.models.jax.common.rope import (DeepseekScalingRotaryEmbedding,
                                                RotaryEmbedding)


class RotaryEmbeddingTest(jtu.JaxTestCase):

    def test_apply_rope(self):
        head_dim = 128
        rope_theta = 10000
        original_max_position_embeddings = 4096
        rope = RotaryEmbedding(head_dim, rope_theta,
                               original_max_position_embeddings)
        self.assertTrue(
            rope.sin_cos_cache.shape == (original_max_position_embeddings,
                                         head_dim))
        num_tokens = 100
        num_heads = 8
        positions = jnp.arange(num_tokens)
        x = jnp.ones((num_tokens, num_heads, head_dim))
        x_rope = rope.apply_rope(positions, x)
        self.assertTrue(x_rope.shape == x.shape)
        self.assertFalse(jnp.array_equal(x_rope, x))


class DeepseekScalingRotaryEmbeddingTest(jtu.JaxTestCase):

    def test_apply_rope(self):
        head_dim = 128
        rope_theta = 10000
        original_max_position_embeddings = 4096
        scaling_factor = 4
        rope = DeepseekScalingRotaryEmbedding(
            head_dim, rope_theta, original_max_position_embeddings,
            scaling_factor)
        self.assertTrue(
            rope.sin_cos_cache.shape == (scaling_factor *
                                         original_max_position_embeddings,
                                         head_dim))
        num_tokens = 100
        num_heads = 8
        positions = jnp.arange(num_tokens)
        x = jnp.ones((num_tokens, num_heads, head_dim))
        x_rope = rope.apply_rope(positions, x)
        self.assertTrue(x_rope.shape == x.shape)
        self.assertFalse(jnp.array_equal(x_rope, x))
