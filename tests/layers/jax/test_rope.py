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
import numpy as np
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_inference.layers.jax.rope import (DeepseekScalingRotaryEmbedding,
                                           Llama4VisionRotaryEmbedding,
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
        with jax.set_mesh(mesh):
            rope.initialize_cache()
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


def _hf_llama4_vision_freqs(image_size, patch_size, hidden_size,
                            num_attention_heads, rope_theta):
    """NumPy transcription of HF Llama4VisionRotaryEmbedding._compute_freqs_ci
    (cos and sin of the rotation angles, shape (num_patches + 1, dim / 2))."""
    idx = image_size // patch_size
    img_idx = np.arange(idx**2, dtype=np.int32).reshape(idx**2, 1)
    img_idx = np.concatenate([img_idx, img_idx[:1]], axis=0)
    img_idx[-1, -1] = -2  # ID_CLS_TOKEN
    frequencies_x = img_idx % idx
    frequencies_y = img_idx // idx
    freq_dim = hidden_size // num_attention_heads // 2
    rope_freq = 1.0 / (rope_theta**(np.arange(
        0, freq_dim, 2)[:(freq_dim // 2)].astype(np.float32) / freq_dim))
    freqs_x = ((frequencies_x + 1)[..., None] *
               rope_freq[None, None, :]).repeat(2, axis=-1)
    freqs_y = ((frequencies_y + 1)[..., None] *
               rope_freq[None, None, :]).repeat(2, axis=-1)
    freqs = np.concatenate([freqs_x, freqs_y], axis=-1)[..., ::2]
    freqs = np.where(img_idx.reshape(-1, 1, 1) < 0, 0.0, freqs)
    freqs = freqs.reshape(freqs.shape[0], -1).astype(np.float32)
    return np.cos(freqs), np.sin(freqs)


class Llama4VisionRotaryEmbeddingTest(jtu.JaxTestCase):
    CONFIG = dict(image_size=4,
                  patch_size=1,
                  hidden_size=32,
                  num_attention_heads=2,
                  rope_theta=10000.0)

    def _table(self):
        # (num_patches + 1, dim / 2, 2) holding (cos, sin).
        return np.asarray(Llama4VisionRotaryEmbedding(**self.CONFIG)())

    def test_cls_row_is_identity_rotation(self):
        cls_row = self._table()[-1]
        np.testing.assert_array_equal(cls_row[..., 0], 1.0)
        np.testing.assert_array_equal(cls_row[..., 1], 0.0)

    def test_patch_rows_match_hf_reference(self):
        table = self._table()
        cos, sin = _hf_llama4_vision_freqs(**self.CONFIG)
        np.testing.assert_allclose(table[:-1, :, 0], cos[:-1], atol=1e-6)
        np.testing.assert_allclose(table[:-1, :, 1], sin[:-1], atol=1e-6)

    def test_full_table_matches_hf_reference(self):
        table = self._table()
        cos, sin = _hf_llama4_vision_freqs(**self.CONFIG)
        self.assertEqual(table.shape, (17, 8, 2))
        np.testing.assert_allclose(table[..., 0], cos, atol=1e-6)
        np.testing.assert_allclose(table[..., 1], sin, atol=1e-6)
