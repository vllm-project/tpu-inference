# Copyright 2025 Google LLC
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

import jax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_inference.layers.jax.rope import (DeepseekScalingRotaryEmbedding,
                                           RotaryEmbedding)


class RotaryEmbeddingTest(jtu.JaxTestCase):

    def test_apply_rope(self):
        head_dim = 2
        rope_theta = 10000
        original_max_position_embeddings = 2
        rope = RotaryEmbedding(
            rotary_dim=head_dim,
            rope_theta=rope_theta,
            original_max_position_embeddings=original_max_position_embeddings,
            dtype=jnp.float32)
        rope.initialize_cache()
        self.assertTrue(
            rope.sin_cos_cache.shape == (original_max_position_embeddings,
                                         head_dim))
        expected_sin_cos = jnp.array([[1, 0], [0.5403023, 0.841471]],
                                     dtype=jnp.float32)
        self.assertArraysAllClose(rope.sin_cos_cache, expected_sin_cos)

        num_tokens = 2
        num_heads = 1
        positions = jnp.arange(num_tokens)
        x = jnp.ones((num_tokens, num_heads, head_dim))
        x_rope = rope.apply_rope(positions, x)
        expected_x_rope = jnp.array([[[1, 1]], [[-0.30116874, 1.3817732]]],
                                    dtype=jnp.float32)
        self.assertTrue(x_rope.shape == x.shape)
        self.assertArraysAllClose(x_rope, expected_x_rope)


class DeepseekScalingRotaryEmbeddingTest(jtu.JaxTestCase):

    def test_apply_rope(self):
        head_dim = 2
        rope_theta = 10000
        original_max_position_embeddings = 1
        scaling_factor = 2
        devices = jax.devices()
        mesh = Mesh(devices, ('data', ))

        rope = DeepseekScalingRotaryEmbedding(
            rotary_dim=head_dim,
            rope_theta=rope_theta,
            original_max_position_embeddings=original_max_position_embeddings,
            scaling_factor=scaling_factor,
            dtype=jnp.float32)
        rope.initialize_cache(mesh)
        expected_padded_dim = 128
        self.assertTrue(
            rope.sin_cos_cache.shape == (scaling_factor *
                                         original_max_position_embeddings,
                                         expected_padded_dim))

        valid_cache_slice = rope.sin_cos_cache[:, :head_dim]

        expected_sin_cos = jnp.array([[1.0693147, 0], [0.5777532, 0.8997973]],
                                     dtype=jnp.float32)

        self.assertArraysAllClose(valid_cache_slice, expected_sin_cos)

        num_tokens = 2
        num_heads = 1
        positions = jnp.arange(num_tokens)
        x = jnp.ones((num_tokens, num_heads, head_dim))
        x_rope = rope.apply_rope(positions, x)
        expected_x_rope = jnp.array(
            [[[1.0693147, 1.0693147]], [[-0.32204413, 1.4775505]]],
            dtype=jnp.float32)
        self.assertTrue(x_rope.shape == x.shape)
        self.assertArraysAllClose(x_rope, expected_x_rope)
