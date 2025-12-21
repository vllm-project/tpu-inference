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
from unittest.mock import MagicMock

import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.jax.layers import DenseFFW
from tpu_inference.layers.jax.moe.moe import MoE
from tpu_inference.layers.jax.transformer_block import (
    SharedExpertsTransformerBlock, TransformerBlock)


class TestTransformerBlock(unittest.TestCase):
    """Unit test suite for the JAX TransformerBlock module."""

    def test_transformer_block_dense_logic(self):
        """
        Tests the forward pass logic of a dense TransformerBlock by mocking its sub-modules.
        This test verifies the sequence of operations and residual connections.
        """
        hidden_size = 1024

        mock_pre_attn_norm = MagicMock(spec=nnx.Module)
        mock_pre_mlp_norm = MagicMock(spec=nnx.Module)

        mock_attn = MagicMock(spec=nnx.Module)
        dummy_attn_output = jnp.full((64, hidden_size),
                                     2.0,
                                     dtype=jnp.bfloat16)
        dummy_kv_cache = jnp.zeros((8, 16, 16, 128), dtype=jnp.bfloat16)
        mock_attn.return_value = (dummy_kv_cache, dummy_attn_output)

        mock_mlp = MagicMock(spec=DenseFFW)
        dummy_mlp_output = jnp.full((64, hidden_size), 3.0, dtype=jnp.bfloat16)
        mock_mlp.return_value = dummy_mlp_output

        transformer_block = TransformerBlock(
            pre_attention_norm=mock_pre_attn_norm,
            pre_mlp_norm=mock_pre_mlp_norm,
            custom_module=mock_mlp,
            attn=mock_attn,
        )

        seq_len = 64
        x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)
        initial_kv_cache = MagicMock()
        attention_metadata = MagicMock()

        mock_pre_attn_norm.side_effect = lambda val: val
        mock_pre_mlp_norm.side_effect = lambda val: val

        new_kv_cache, final_output = transformer_block(
            x,
            is_prefill=True,
            kv_cache=initial_kv_cache,
            attention_metadata=attention_metadata,
        )

        mock_pre_attn_norm.assert_called_once()
        self.assertTrue(
            jnp.array_equal(mock_pre_attn_norm.call_args.args[0], x))

        mock_attn.assert_called_once_with(x, True, initial_kv_cache,
                                          attention_metadata, True)

        expected_mlp_norm_input = dummy_attn_output + x

        mock_pre_mlp_norm.assert_called_once()
        self.assertTrue(
            jnp.array_equal(mock_pre_mlp_norm.call_args.args[0],
                            expected_mlp_norm_input))

        mock_mlp.assert_called_once()
        self.assertTrue(
            jnp.array_equal(mock_mlp.call_args.args[0],
                            expected_mlp_norm_input))

        expected_final_output = dummy_mlp_output + expected_mlp_norm_input
        self.assertTrue(jnp.allclose(final_output, expected_final_output))

        self.assertTrue(jnp.array_equal(new_kv_cache, dummy_kv_cache))

    def test_shared_experts_transformer_block_logic(self):
        """Tests the forward pass logic of a SharedExpertsTransformerBlock."""
        hidden_size = 1024

        mock_pre_attn_norm = MagicMock(spec=nnx.Module)
        mock_pre_mlp_norm = MagicMock(spec=nnx.Module)

        mock_attn = MagicMock(spec=nnx.Module)
        dummy_attn_output = jnp.full((64, hidden_size),
                                     2.0,
                                     dtype=jnp.bfloat16)
        dummy_kv_cache = jnp.zeros((8, 16, 16, 128), dtype=jnp.bfloat16)
        mock_attn.return_value = (dummy_kv_cache, dummy_attn_output)

        mock_moe = MagicMock(spec=MoE)
        dummy_moe_output = jnp.full((64, hidden_size), 3.0, dtype=jnp.bfloat16)
        mock_moe.return_value = dummy_moe_output

        mock_shared_experts = MagicMock(spec=DenseFFW)
        dummy_shared_experts_output = jnp.full((64, hidden_size),
                                               4.0,
                                               dtype=jnp.bfloat16)
        mock_shared_experts.return_value = dummy_shared_experts_output

        transformer_block = SharedExpertsTransformerBlock(
            pre_attention_norm=mock_pre_attn_norm,
            pre_mlp_norm=mock_pre_mlp_norm,
            custom_module=mock_moe,
            attn=mock_attn,
            shared_experts=mock_shared_experts,
        )

        seq_len = 64
        x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)
        initial_kv_cache = MagicMock()
        attention_metadata = MagicMock()

        mock_pre_attn_norm.side_effect = lambda val: val
        mock_pre_mlp_norm.side_effect = lambda val: val

        new_kv_cache, final_output = transformer_block(
            x,
            is_prefill=True,
            kv_cache=initial_kv_cache,
            attention_metadata=attention_metadata,
        )
        self.assertTrue(jnp.array_equal(new_kv_cache, dummy_kv_cache))
        self.assertEqual(final_output.shape, (seq_len, hidden_size))

        self.assertEqual(mock_moe.call_count, 1)
        self.assertEqual(mock_attn.call_count, 1)
        self.assertEqual(mock_shared_experts.call_count, 1)


if __name__ == "__main__":
    unittest.main()
