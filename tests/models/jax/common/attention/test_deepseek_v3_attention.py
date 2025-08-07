import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.deepseek_v3_attention import MLA
from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.recipes.deepseek_v3 import (
    DeepSeekV3GenerateShardingRulesConfig,
    DeepSeekV3PrefillShardingRulesConfig, DeepSeekV3ShardingRulesConfig)


class TestMLA(unittest.TestCase):

    def setUp(self):
        self.mesh = Mesh(
            np.array(jax.devices("tpu")).reshape(1, -1),
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
            default_rules_cls=DeepSeekV3ShardingRulesConfig,
            prefill_rules=DeepSeekV3PrefillShardingRulesConfig,
            generate_rules=DeepSeekV3GenerateShardingRulesConfig,
        )

    def test_mla_forward_pass(self):
        hidden_size = 256

        num_key_value_heads = 32
        qk_nope_head_dim = 64
        qk_rope_head_dim = 32

        mla = MLA(
            hidden_size=hidden_size,
            num_attention_heads=32,
            num_key_value_heads=num_key_value_heads,
            head_dim=64,  # MLA uses v_head_dim as head_dim
            rope_theta=10000,
            dtype=jnp.bfloat16,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_nope_head_dim=
            qk_nope_head_dim,  # Half of DeepSeek v3's real values
            qk_rope_head_dim=
            qk_rope_head_dim,  # Half of DeepSeek v3's real values
            v_head_dim=64,  # Half of DeepSeek v3's real values
            rms_norm_eps=1e-5,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.sharding_cfg,
            quant=None,
        )
        mla.generate_kernel(nnx.Rngs(42))

        # Create input tensor
        seq_len = 32
        x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)

        # Create KV cache
        # TODO(wenxindongwork): test with unpadded head dimension once
        # MLA kv cache implementation is added.
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        multiple_of_128 = ((qk_head_dim - 1) // 128 + 1) * 128
        block_size = 16
        num_blocks = 8
        cache_shape = (
            num_blocks,
            block_size,
            num_key_value_heads * 2,
            multiple_of_128,
        )
        kv_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

        # Create attention metadata
        num_slices = 8
        attention_metadata = AttentionMetadata(
            input_positions=jnp.arange(seq_len, dtype=jnp.int32),
            slot_mapping=jnp.array(
                [
                    [0 for _ in range(num_slices)],
                    [0 for _ in range(num_slices)],
                    [block_size for _ in range(num_slices)],
                ],
                dtype=jnp.int32,
            ),
            block_tables=jnp.zeros((1, 8), dtype=jnp.int32),
            seq_lens=jnp.ones((1, ), dtype=jnp.int32) * seq_len,
            num_slices=jnp.array([num_slices], dtype=jnp.int32),
            num_seqs=jnp.ones((1, ), dtype=jnp.int32),
            query_start_loc=jnp.array([0, seq_len],
                                      dtype=jnp.int32),  # This is cu_q_lens
            request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        )

        # Run forward pass
        new_kv_cache, output = mla(x,
                                   is_prefill=True,
                                   kv_cache=kv_cache,
                                   attention_metadata=attention_metadata)

        # Verify output shapes
        self.assertEqual(output.shape, (seq_len, hidden_size))
        self.assertEqual(new_kv_cache.shape, kv_cache.shape)


if __name__ == "__main__":
    unittest.main()
