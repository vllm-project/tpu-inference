import os
import unittest
from dataclasses import dataclass, field

import chex

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.attention.llama4_attention import (
    L2Norm, Llama4Attention)
from tpu_commons.models.jax.common.sharding import build_mesh


@dataclass
class SimpleVLLMConfig:
    additional_config: dict = field(default_factory=dict)


class Llama4AttentionTest(unittest.TestCase):
    """Unit test suite for Llama4-specific attention components."""

    def setUp(self):
        devices = jax.devices()
        sharding_strategy = {"tensor_parallelism": len(devices)}
        self.mesh = build_mesh(devices, sharding_strategy)

    def test_l2norm_forward_pass(self):
        """Tests the forward pass of the L2Norm module with hardcoded values."""
        eps = 1e-5
        l2_norm = L2Norm(eps=eps)
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)

        output = l2_norm(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

        # Expected values calculated manually:
        # mean_sq = (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1+4+9+16)/4 = 30/4 = 7.5
        # norm_val = sqrt(7.5 + 1e-5)
        # expected = x / norm_val
        expected_output = jnp.array([[0.365148, 0.730297, 1.095445, 1.460594]],
                                    dtype=jnp.float32)
        self.assertTrue(jnp.allclose(output, expected_output, atol=1e-6))

    def test_l2norm_with_zeros(self):
        """Tests L2Norm with an all-zero input."""
        l2_norm = L2Norm(eps=1e-5)
        x = jnp.zeros((4, 8, 16))
        output = l2_norm(x)
        self.assertEqual(output.shape, x.shape)
        # Output should be all zeros.
        self.assertTrue(jnp.all(output == 0))

    def test_l2norm_eps_effect(self):
        """Tests the effect of the epsilon value in L2Norm."""
        eps = 1e-3
        l2_norm = L2Norm(eps=eps)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 128))
        output = l2_norm(x)

        mean_sq = jnp.mean(x**2, axis=-1, keepdims=True)
        expected_output = x * jax.lax.rsqrt(mean_sq + eps)

        self.assertTrue(jnp.allclose(output, expected_output))

    def test_apply_temperature_tuning(self):
        hidden_size = 64
        num_attention_heads = 4
        head_dim = hidden_size // num_attention_heads

        # Create dummy sharding objects
        dummy_sharding = NamedSharding(self.mesh, P())

        llama4_attention = Llama4Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=head_dim,
            rope_theta=10000.0,
            rope_scaling={},
            dtype=jnp.bfloat16,
            use_qk_norm=False,
            temperature_tuning=True,
            temperature_tuning_scale=2.0,
            temperature_tuning_floor_scale=2.0,
            mesh=self.mesh,
            random_init=True,
            quant=None,
            activation_attention_td=dummy_sharding,
            activation_attention_out_td=dummy_sharding,
            dnh_sharding=dummy_sharding,
            dkh_sharding=dummy_sharding,
            nhd_sharding=dummy_sharding,
            activation_q_td=dummy_sharding,
            query_tnh=dummy_sharding,
            keyvalue_skh=dummy_sharding,
            keyvalue_cache_lskh=dummy_sharding,
            attn_o_tnh=dummy_sharding,
        )

        seq_len = 8
        input_arr_TNH = jnp.ones((seq_len, num_attention_heads, head_dim),
                                 dtype=jnp.bfloat16)
        attention_metadata = AttentionMetadata(input_positions=jnp.arange(
            seq_len, dtype=jnp.int32),
                                               slot_mapping=None,
                                               block_tables=None,
                                               seq_lens=None,
                                               num_slices=None,
                                               num_seqs=None,
                                               query_start_loc=None)
        expected_scales = jnp.array(
            [1, 2.375, 2.375, 3.20312, 3.20312, 3.76562, 3.76562, 4.21875],
            dtype=jnp.bfloat16)
        output = llama4_attention.apply_temperature_tuning(
            attention_metadata, input_arr_TNH)
        chex.assert_shape(output, (seq_len, num_attention_heads, head_dim))

        expected_output = jnp.ones_like(input_arr_TNH) * expected_scales[:,
                                                                         None,
                                                                         None]
        chex.assert_trees_all_close(output, expected_output, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
