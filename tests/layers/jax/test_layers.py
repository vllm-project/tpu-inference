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

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.jax.layers import DenseFFW, Embedder, RMSNorm


class TestLayers(unittest.TestCase):
    """Unit test suite for common JAX layer blocks."""

    def setUp(self):
        """Sets up the testing environment before each test."""
        self.mesh = Mesh(
            np.array(jax.devices()).reshape(1, -1),
            axis_names=(
                "expert",
                "model",
            ),
        )

    def test_rmsnorm_forward_pass(self):
        """Tests the forward pass of the RMSNorm module."""
        with jax.set_mesh(self.mesh):
            dims = 512
            epsilon = 1e-5

            norm = RMSNorm(
                dims=dims,
                random_init=True,
                epsilon=epsilon,
                rngs=nnx.Rngs(0),
                dtype=jnp.float32,
            )

            seq_len = 128
            x = jax.random.normal(jax.random.PRNGKey(42), (seq_len, dims))

            output = norm(x)

            self.assertEqual(output.shape, x.shape)
            self.assertEqual(output.dtype, jnp.float32)

            mean_of_squares = jnp.mean(jnp.square(output), axis=-1)
            self.assertTrue(
                jnp.allclose(mean_of_squares, 1.0, atol=1e-5).all())

    def test_denseffw_forward_pass(self):
        """Tests the forward pass of the DenseFFW module."""
        with jax.set_mesh(self.mesh):
            hidden_size = 512
            intermediate_size = 2048

            ffw_layer = DenseFFW(
                random_init=True,
                dtype=jnp.bfloat16,
                hidden_act="silu",
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rngs=nnx.Rngs(0),
            )

            seq_len = 128
            x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)

            output = ffw_layer(x)

            self.assertEqual(output.shape, x.shape)
            self.assertEqual(output.dtype, x.dtype)

    def test_embedder_forward_pass(self):
        """Tests both the encode and decode passes of the Embedder module."""
        with jax.set_mesh(self.mesh):
            hidden_size = 512
            vocab_size = 32000
            dtype = jnp.bfloat16

            embedder = Embedder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                dtype=dtype,
                random_init=True,
                rngs=nnx.Rngs(0),
            )

            seq_len = 128
            token_ids = jnp.arange(seq_len, dtype=jnp.int32) % vocab_size
            embeddings = embedder(token_ids, decode=False)
            self.assertEqual(embeddings.shape, (seq_len, hidden_size))
            self.assertEqual(embeddings.dtype, dtype)

            hidden_states = jnp.ones((seq_len, hidden_size),
                                     dtype=jnp.bfloat16)
            logits = embedder(hidden_states, decode=True)
            self.assertEqual(logits.shape, (seq_len, vocab_size))
            self.assertEqual(logits.dtype, dtype)

    def test_embedder_normalization(self):
        """Tests the embedding normalization feature."""
        with jax.set_mesh(self.mesh):
            hidden_size = 512
            vocab_size = 32000

            rngs_1 = nnx.Rngs(42)
            rngs_2 = nnx.Rngs(42)

            embedder_norm = Embedder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                dtype=jnp.float32,
                normalize_embeddings=True,
                random_init=True,
                rngs=rngs_1,
            )

            embedder_no_norm = Embedder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                dtype=jnp.float32,
                random_init=True,
                rngs=rngs_2,
            )

            token_ids = jnp.arange(10, dtype=jnp.int32)

            embeddings_norm = embedder_norm(token_ids, decode=False)
            embeddings_no_norm = embedder_no_norm(token_ids, decode=False)

            scaling_factor = jnp.sqrt(hidden_size)
            expected_embeddings = embeddings_no_norm * scaling_factor

            self.assertTrue(
                jnp.allclose(embeddings_norm, expected_embeddings,
                             atol=1e-6).all())


if __name__ == "__main__":
    unittest.main()
