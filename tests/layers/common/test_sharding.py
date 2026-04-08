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
from unittest.mock import MagicMock, patch

from tpu_inference.layers.common.sharding import (LazyShardingAxisName,
                                                  ShardingAxisName2D,
                                                  ShardingAxisNameBase,
                                                  ShardingConfigManager)


class TestShardingConfigManager(unittest.TestCase):

    @patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN", True)
    def test_sharding_config_manager_from_vllm_config(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.parallel_config.data_parallel_size = 2
        vllm_config.model_config.use_mla = True
        vllm_config.model_config.get_total_num_kv_heads.return_value = 1
        vllm_config.speculative_config = None
        vllm_config.lora_config = None

        # Test default sharding strategy
        vllm_config.additional_config = {"sharding": {}}

        manager = ShardingConfigManager.from_vllm_config(vllm_config)
        self.assertEqual(manager.tp_size, 8)
        self.assertEqual(manager.model_dp_size, 2)
        self.assertEqual(manager.attn_dp_size, 1)
        self.assertEqual(manager.attn_dp_expert_size, 1)
        self.assertEqual(manager.total_dp_size, 2)

    @patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN", True)
    def test_sharding_config_manager_with_dp_attention(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.model_config.use_mla = True
        vllm_config.model_config.get_total_num_kv_heads.return_value = 1
        vllm_config.speculative_config = None
        vllm_config.lora_config = None
        vllm_config.cache_config.cache_dtype = "bfloat16"

        # Test with enable_dp_attention=True and expert_parallelism
        vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True,
                    "expert_parallelism": 4
                }
            }
        }

        manager = ShardingConfigManager.from_vllm_config(vllm_config)
        # num_kv_heads = 1 (MLA), packing = 2 (BF16)
        # num_kv_heads_per_device_in_kv_cache = max(1, (1 * 2) / 2) = 1
        # attn_dp = max(int(8 // 1), 1) = 8
        # tensor_parallelism = 8 // 8 = 1
        # attn_dp_expert = expert_parallelism = 4
        # expert_parallelism = 1

        self.assertEqual(manager.tp_size, 1)
        self.assertEqual(manager.attn_dp_size, 8)
        self.assertEqual(manager.attn_dp_expert_size, 4)
        self.assertEqual(manager.expert_size, 1)
        self.assertEqual(manager.total_dp_size, 32)  # 1 * 8 * 4

    def test_sharding_config_manager_explicit_tp(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.model_config.use_mla = True
        vllm_config.speculative_config = None

        vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": 8
                }
            }
        }

        manager = ShardingConfigManager.from_vllm_config(vllm_config)
        self.assertEqual(manager.tp_size, 8)

    def test_sharding_config_manager_tp_none(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.model_config.use_mla = True
        vllm_config.speculative_config = None

        # Test when sharding_strategy tensor_parallelism is None
        vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": None
                }
            }
        }

        manager = ShardingConfigManager.from_vllm_config(vllm_config)
        # Should fallback to pc_tensor_parallelism (8)
        self.assertEqual(manager.tp_size, 8)

    def test_sharding_config_manager_tp_equal_1(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.model_config.use_mla = True
        vllm_config.speculative_config = None

        # Test when sharding_strategy tensor_parallelism is 1
        vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": 1
                }
            }
        }

        manager = ShardingConfigManager.from_vllm_config(vllm_config)
        # Should use ss_tensor_parallelism (1)
        self.assertEqual(manager.tp_size, 1)

    def test_sharding_config_manager_tp_different(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.model_config.use_mla = True
        vllm_config.speculative_config = None

        # Test when sharding_strategy tensor_parallelism is different from pc_tensor_parallelism
        vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "tensor_parallelism": 4
                }
            }
        }

        manager = ShardingConfigManager.from_vllm_config(vllm_config)
        # Should use ss_tensor_parallelism (4)
        self.assertEqual(manager.tp_size, 4)


class TestLazyShardingAxisName(unittest.TestCase):

    def test_initial_state_is_uninitialized(self):
        lazy = LazyShardingAxisName()
        self.assertIsNone(lazy._cls)

    @patch("tpu_inference.layers.common.sharding.envs.USE_2D_TP", True)
    @patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN", False)
    def test_use_2d_tp_selects_base(self):
        lazy = LazyShardingAxisName()
        _ = lazy.SEQUENCE
        self.assertIs(lazy._cls, ShardingAxisNameBase)

    @patch("tpu_inference.layers.common.sharding.envs.USE_2D_TP", False)
    @patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN", False)
    def test_both_false_selects_2d(self):
        lazy = LazyShardingAxisName()
        _ = lazy.SEQUENCE
        self.assertIs(lazy._cls, ShardingAxisName2D)

    def test_cls_cached_after_first_access(self):
        lazy = LazyShardingAxisName()
        with patch("tpu_inference.layers.common.sharding.envs.USE_2D_TP",
                   False), \
             patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN",
                   False):
            _ = lazy.SEQUENCE
            first_cls = lazy._cls

        # Even with different env vars, _cls should not change
        with patch("tpu_inference.layers.common.sharding.envs.USE_2D_TP",
                   True), \
             patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN",
                   True):
            _ = lazy.SEQUENCE
            self.assertIs(lazy._cls, first_cls)

    def test_initialized_after_env_change_uses_new_value(self):
        # LazyShardingAxisName is created while USE_2D_TP=False,
        # then USE_2D_TP changes to True before the first attribute access.
        # It should pick up the value at initialization time (first access),
        # not at construction time.
        with patch("tpu_inference.layers.common.sharding.envs.USE_2D_TP",
                   False), \
             patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN",
                   False):
            lazy = LazyShardingAxisName()
            self.assertIsNone(lazy._cls)  # not yet initialized

        with patch("tpu_inference.layers.common.sharding.envs.USE_2D_TP",
                   True), \
             patch("tpu_inference.layers.common.sharding.envs.NEW_MODEL_DESIGN",
                   False):
            _ = lazy.SEQUENCE  # initialized here, after env changed
            self.assertIs(lazy._cls, ShardingAxisNameBase)


if __name__ == "__main__":
    unittest.main()
