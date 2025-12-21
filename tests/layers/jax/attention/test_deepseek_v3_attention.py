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

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, PartitionSpec
from parameterized import parameterized

import tpu_inference.kernels.mla.v1.kernel as mla
from tpu_inference.layers.common.attention_interface import get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.attention.deepseek_v3_attention import MLA


class TestMLA(unittest.TestCase):

    def setUp(self):
        os.environ["NEW_MODEL_DESIGN"] = "1"
        self.mesh = Mesh(
            np.array(jax.devices("tpu")[:1]).reshape(1, 1, 1, 1),
            axis_names=("data", "attn_dp", "expert", "model"),
        )

    @parameterized.expand([["auto"], ["fp8"]])
    def test_mla_forward_pass(self, kv_cache_str):
        hidden_size = 256

        num_key_value_heads = 32
        qk_nope_head_dim = 64
        qk_rope_head_dim = 32

        with jax.set_mesh(self.mesh):
            query_tnh_spec = PartitionSpec(None, ShardingAxisName.MLP_TENSOR,
                                           None)
            keyvalue_skh_spec = PartitionSpec(None,
                                              ShardingAxisName.MLP_TENSOR,
                                              None)
            attn_o_tnh_spec = PartitionSpec(None, ShardingAxisName.MLP_TENSOR,
                                            None)

            mla_layer = MLA(
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
                rngs=nnx.Rngs(42),
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
                random_init=True,
                kv_cache_dtype=kv_cache_str,
                query_tnh=query_tnh_spec,
                keyvalue_skh=keyvalue_skh_spec,
                attn_o_tnh=attn_o_tnh_spec,
                q_da_sharding=(None, ShardingAxisName.VOCAB),
                anh_sharding=(None, ShardingAxisName.MLP_TENSOR, None),
                ap_sharding=(None, ShardingAxisName.MLP_TENSOR),
                kv_da_sharding=(None, ShardingAxisName.VOCAB),
                rd_sharding=(ShardingAxisName.MLP_TENSOR, None),
            )

            # Create input tensor
            seq_len = 32
            x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)

            # Create KV cache
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            block_size = 16
            num_blocks = 8
            kv_dtype = jnp.float8_e4m3fn if kv_cache_str == "fp8" else jnp.bfloat16
            cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                             num_key_value_heads, qk_head_dim,
                                             kv_dtype)
            kv_cache = jnp.zeros(cache_shape, dtype=kv_dtype)

            # Create attention metadata
            attention_metadata = AttentionMetadata(
                input_positions=jnp.arange(seq_len, dtype=jnp.int32),
                block_tables=jnp.zeros((8, ), dtype=jnp.int32),
                seq_lens=jnp.ones((1, ), dtype=jnp.int32) * seq_len,
                query_start_loc=jnp.array(
                    [0, seq_len], dtype=jnp.int32),  # This is cu_q_lens
                request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
            )

            mla_layer.rope.initialize_cache(self.mesh)

            # Run forward pass
            new_kv_cache, output = mla_layer(
                x,
                is_prefill=True,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata)

            # Verify output shapes
            self.assertEqual(output.shape, (seq_len, hidden_size))
            self.assertEqual(new_kv_cache.shape, kv_cache.shape)

    @parameterized.expand([["auto"]])  # MLA kernel does not support fp8 yet
    def test_mla_kernel_forward_pass(self, kv_cache_str):
        hidden_size = 256

        num_key_value_heads = 1
        qk_nope_head_dim = 64
        qk_rope_head_dim = 32
        v_head_dim = 64
        kv_lora_rank = 512

        with jax.set_mesh(self.mesh):
            query_tnh_spec = PartitionSpec(ShardingAxisName.MLP_TENSOR, None,
                                           None)
            keyvalue_skh_spec = PartitionSpec(ShardingAxisName.MLP_TENSOR,
                                              None)
            attn_o_tnh_spec = PartitionSpec(ShardingAxisName.MLP_TENSOR, None,
                                            None)

            mla_layer = MLA(
                hidden_size=hidden_size,
                num_attention_heads=32,
                num_key_value_heads=num_key_value_heads,
                head_dim=v_head_dim,  # MLA uses v_head_dim as head_dim
                rope_theta=10000,
                dtype=jnp.bfloat16,
                q_lora_rank=512,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                rms_norm_eps=1e-5,
                rngs=nnx.Rngs(42),
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
                random_init=True,
                kv_cache_dtype=kv_cache_str,
                use_mla_kernel=
                True,  # Set to true, in order to trigger MLA kernel.
                query_tnh=query_tnh_spec,
                keyvalue_skh=keyvalue_skh_spec,
                attn_o_tnh=attn_o_tnh_spec,
                q_da_sharding=(None, ShardingAxisName.VOCAB),
                anh_sharding=(None, ShardingAxisName.MLP_TENSOR, None),
                ap_sharding=(None, ShardingAxisName.MLP_TENSOR),
                kv_da_sharding=(None, ShardingAxisName.VOCAB),
                rd_sharding=(ShardingAxisName.MLP_TENSOR, None),
            )

            # Create input tensor
            seq_len = 32
            x = jnp.ones((seq_len, hidden_size), dtype=jnp.bfloat16)

            # Create KV cache for MLA kernel
            block_size = 16
            num_blocks = 8
            kv_dtype = jnp.float8_e4m3fn if kv_cache_str == "fp8" else jnp.bfloat16

            # For the MLA kernel, the head dimension is the sum of qk_nope_head_dim and v_head_dim
            # and lora rank
            cache_shape = mla.get_kv_cache_shape(
                num_blocks, block_size, kv_lora_rank + qk_rope_head_dim,
                kv_dtype)
            kv_cache = jnp.zeros(cache_shape, dtype=kv_dtype)

            # Create attention metadata
            attention_metadata = AttentionMetadata(
                input_positions=jnp.arange(seq_len, dtype=jnp.int32),
                block_tables=jnp.zeros((8, ), dtype=jnp.int32),
                seq_lens=jnp.ones((1, ), dtype=jnp.int32) * seq_len,
                query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
                request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
            )

            mla_layer.rope.initialize_cache(self.mesh)

            # Run forward pass
            new_kv_cache, output = mla_layer(
                x,
                is_prefill=True,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata)

            # Verify output shapes
            self.assertEqual(output.shape, (seq_len, hidden_size))
            self.assertEqual(new_kv_cache.shape, kv_cache.shape)


if __name__ == "__main__":
    unittest.main()


def tearDownModule():
    del os.environ["NEW_MODEL_DESIGN"]
