import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from tpu_commons.models.jax.common.attention.attention import AttentionConfig
from tpu_commons.models.jax.common.layers import DenseFFWConfig
from tpu_commons.models.jax.common.moe.moe import (MoEConfig, RouterConfig,
                                                   RouterType)
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.common.transformer_block import (
    SharedExpertsTransformerBlock, SharedExpertsTransformerBlockConfig,
    TransformerBlock, TransformerBlockConfig)


class TestTransformerBlock(unittest.TestCase):
    """Unit test suite for the JAX TransformerBlock module."""

    @patch.object(TransformerBlock, '_create_module')
    @patch('tpu_commons.models.jax.common.transformer_block.RMSNorm')
    def test_transformer_block_dense_logic(self, MockRMSNorm,
                                           mock_create_module):
        """
        Tests the forward pass logic of a dense TransformerBlock by mocking its sub-modules.
        This test verifies the sequence of operations and residual connections.
        """
        hidden_size = 1024
        intermediate_size = 4096
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
        dense_ffw_config = DenseFFWConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            dtype=jnp.bfloat16,
            vllm_config=None,
        )
        transformer_config = TransformerBlockConfig(
            attention=attention_config,
            dense_ffw=dense_ffw_config,
            rms_norm_eps=1e-6,
            vllm_config=None,
        )

        mock_pre_attn_norm = MagicMock()
        mock_pre_mlp_norm = MagicMock()

        MockRMSNorm.side_effect = [mock_pre_attn_norm, mock_pre_mlp_norm]

        mock_attn = MagicMock()
        dummy_attn_output = jnp.full((64, hidden_size),
                                     2.0,
                                     dtype=jnp.bfloat16)
        dummy_kv_cache = jnp.zeros((8, 16, 16, 128), dtype=jnp.bfloat16)
        mock_attn.return_value = (dummy_kv_cache, dummy_attn_output)

        mock_mlp = MagicMock()
        dummy_mlp_output = jnp.full((64, hidden_size), 3.0, dtype=jnp.bfloat16)
        mock_mlp.return_value = dummy_mlp_output
        mock_create_module.side_effect = [mock_attn, mock_mlp]

        transformer_block = TransformerBlock(
            cfg=transformer_config,
            block_type="dense",
            param_factory=None,
            mesh=None,
            sharding_cfg=ShardingConfig(),
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
        self.assertEqual(mock_mlp.call_args.args[1], 'prefill')
        self.assertEqual(mock_mlp.call_args.kwargs, {})

        expected_final_output = dummy_mlp_output + expected_mlp_norm_input
        self.assertTrue(jnp.allclose(final_output, expected_final_output))

        self.assertTrue(jnp.array_equal(new_kv_cache, dummy_kv_cache))

    @patch.object(TransformerBlock, '_create_module')
    @patch('tpu_commons.models.jax.common.transformer_block.RMSNorm')
    def test_shared_experts_transformer_block_logic(self, MockRMSNorm,
                                                    mock_create_module):
        """Tests the forward pass logic of a SharedExpertsTransformerBlock."""
        hidden_size = 1024
        intermediate_size_moe = 4096
        intermediate_size = 8192
        num_attention_heads = 8
        head_dim = hidden_size // num_attention_heads
        num_local_experts = 4
        shared_experts = 2
        num_experts_per_tok = 1

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
        moe_config = MoEConfig(hidden_size=hidden_size,
                               intermediate_size_moe=intermediate_size_moe,
                               dtype=jnp.bfloat16,
                               num_local_experts=num_local_experts,
                               hidden_act="silu",
                               apply_expert_weight_before_computation=True,
                               router=RouterConfig(
                                   hidden_size=hidden_size,
                                   num_local_experts=num_local_experts,
                                   num_experts_per_token=num_experts_per_tok,
                                   router_type=RouterType.TOP_K,
                                   router_act="sigmoid",
                                   expert_capacity=-1,
                                   dtype=jnp.bfloat16,
                                   vllm_config=None))

        dense_ffw_config = DenseFFWConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            dtype=jnp.bfloat16,
            vllm_config=None,
        )

        shared_config = SharedExpertsTransformerBlockConfig(
            attention=attention_config,
            moe=moe_config,
            dense_ffw=dense_ffw_config,
            shared_experts=shared_experts,
            rms_norm_eps=1e-6,
            vllm_config=None,
        )

        mock_pre_attn_norm = MagicMock()
        mock_pre_mlp_norm = MagicMock()

        MockRMSNorm.side_effect = [mock_pre_attn_norm, mock_pre_mlp_norm]

        mock_attn = MagicMock()
        dummy_attn_output = jnp.full((64, hidden_size),
                                     2.0,
                                     dtype=jnp.bfloat16)
        dummy_kv_cache = jnp.zeros((8, 16, 16, 128), dtype=jnp.bfloat16)
        mock_attn.return_value = (dummy_kv_cache, dummy_attn_output)

        mock_moe = MagicMock()
        dummy_moe_output = jnp.full((64, hidden_size), 3.0, dtype=jnp.bfloat16)
        mock_moe.return_value = dummy_moe_output

        mock_shared_experts = MagicMock()
        dummy_shared_experts_output = jnp.full((64, hidden_size),
                                               4.0,
                                               dtype=jnp.bfloat16)
        mock_shared_experts.return_value = dummy_shared_experts_output
        mock_create_module.side_effect = [
            mock_attn, mock_moe, mock_shared_experts
        ]

        transformer_block = SharedExpertsTransformerBlock(
            cfg=shared_config,
            block_type="moe",  # Assuming MoE block type for this test
            param_factory=MagicMock(),
            mesh=MagicMock(),
            sharding_cfg=MagicMock(),
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

        # Basic check that the moe and shared experts are used.
        self.assertTrue(mock_moe.call_count == 1)
        self.assertTrue(mock_attn.call_count == 1)
        self.assertTrue(mock_shared_experts.call_count == 1)


if __name__ == "__main__":
    unittest.main()
