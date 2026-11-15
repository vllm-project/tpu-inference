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
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_interface import get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention.gpt_oss_attention import \
    GptOssAttention


class TestGptOssAttention(unittest.TestCase):
    """Unit tests for GptOssAttention JAX layer."""

    def setUp(self):
        self.mesh = Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1, 1, -1),
            axis_names=(
                "data",
                "attn_dp",
                "expert",
                "model",
            ),
        )

    @patch(
        "tpu_inference.layers.jax.attention.gpt_oss_attention.ragged_paged_attention_hd64"
    )
    def test_sinks_cast_to_float32(self, mock_kernel):
        """Verify that sinks array is cast to float32 when calling attention."""
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 2
        head_dim = 64

        # Mock kernel return: output_TNH, kv_cache
        # We need to construct output shape matching sharding spec (data, model, None)
        # and kv_cache.
        def mock_impl(*args, **kwargs):
            q = args[0]
            kv_cache = args[3]
            sinks = args[8]

            # Verify that sinks is indeed float32
            self.assertEqual(sinks.dtype, jnp.float32)

            output = jnp.zeros_like(q)
            return output, kv_cache

        mock_kernel.side_effect = mock_impl

        with jax.set_mesh(self.mesh):
            attention = GptOssAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=jnp.bfloat16,
                kv_cache_dtype="auto",
                rope_theta=10000.0,
                initial_context_length=2048,
                rope_scaling_factor=1.0,
                rope_ntk_alpha=1.0,
                rope_ntk_beta=1.0,
                rngs=nnx.Rngs(42),
                random_init=True,
                mesh=self.mesh,
            )

            seq_len = 16
            q_TNH = jnp.ones((seq_len, num_attention_heads, head_dim),
                             dtype=jnp.bfloat16)
            k_SKH = jnp.ones((seq_len, num_key_value_heads, head_dim),
                             dtype=jnp.bfloat16)
            v_SKH = jnp.ones((seq_len, num_key_value_heads, head_dim),
                             dtype=jnp.bfloat16)

            # Sinks are initialized as bfloat16 to verify it gets cast to float32
            sinks = jnp.ones((4, ), dtype=jnp.bfloat16)

            block_size = 16
            num_blocks = 2
            cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                             num_key_value_heads, head_dim,
                                             jnp.bfloat16)
            kv_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

            attention_metadata = AttentionMetadata(
                input_positions=jnp.arange(seq_len, dtype=jnp.int32),
                block_tables=jnp.zeros((1, 2), dtype=jnp.int32).reshape(-1),
                seq_lens=jnp.array([seq_len], dtype=jnp.int32),
                query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
                request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
            )
            attention_metadata.sliding_window = None

            # Call the attention method
            new_kv_cache, output = attention.attention(
                kv_cache=kv_cache,
                q_TNH=q_TNH,
                k_SKH=k_SKH,
                v_SKH=v_SKH,
                sinks=sinks,
                attention_metadata=attention_metadata,
                mesh=self.mesh,
            )

            self.assertEqual(output.shape, q_TNH.shape)
            self.assertTrue(mock_kernel.called)


if __name__ == "__main__":
    unittest.main()
