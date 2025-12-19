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

import jax

from tpu_inference.layers.common.sharding import (Sharding, ShardingConfig,
                                                  ShardingRulesConfig,
                                                  ShardingStrategy)


class TestSharding(unittest.TestCase):
    """Unit test suite for the sharding configuration logic."""

    def setUp(self):
        """Sets up the testing environment before each test."""

        self.mock_devices = [MagicMock(coords=i) for i in range(8)]
        self.original_jax_devices = jax.devices
        jax.devices = lambda: self.mock_devices

    def tearDown(self):
        """Restores the original jax.devices function after tests."""
        jax.devices = self.original_jax_devices

    def test_sharding_strategy_init(self):
        """Tests the initialization of the ShardingStrategy."""
        strategy = ShardingStrategy(
            tensor_parallelism=2,
            expert_parallelism=4,
            data_parallelism=1,
            sequence_parallelism=1,
        )
        self.assertEqual(strategy.tensor_parallelism, 2)
        self.assertEqual(strategy.expert_parallelism, 4)

    def test_sharding_config_init(self):
        """Tests the initialization of ShardingConfig."""
        config = ShardingConfig()
        self.assertIsInstance(config.prefill_rules, ShardingRulesConfig)
        self.assertIsInstance(config.generate_rules, ShardingRulesConfig)

        custom_rules = ShardingRulesConfig(activation_ffw_td=("model", None))
        config_with_rules = ShardingConfig(prefill_rules=custom_rules)
        self.assertEqual(config_with_rules.prefill_rules.activation_ffw_td,
                         ("model", None))

    def test_apply_overrides(self):
        """Tests the _apply_overrides method for valid and invalid keys."""
        sharding = Sharding(
            prefill_rules={},
            generate_rules={},
        )
        config_obj = ShardingRulesConfig()

        valid_overrides = {"activation_ffw_td": ("model", None)}
        sharding._apply_overrides(config_obj, valid_overrides)
        self.assertEqual(config_obj.activation_ffw_td, ("model", None))

        invalid_overrides = {"non_existent_attribute": (None, "model")}
        with self.assertRaises(AttributeError):
            sharding._apply_overrides(config_obj, invalid_overrides)

    def test_default_sharding_config(self):
        """Tests that default sharding rules are created correctly."""
        sharding = Sharding(
            prefill_rules={},
            generate_rules={},
        )

        sharding_cfg = sharding.get_sharding_cfg()
        generate_rules = sharding_cfg.generate_rules

        self.assertEqual(generate_rules.ffw_weight_df, (None, "model"))
        self.assertEqual(generate_rules.moe_router_de, (None, "model"))
        self.assertEqual(generate_rules.attn_q_weight_dnh,
                         (None, "model", None))

    def test_sharding_init_with_overrides(self):
        """Tests Sharding initialization with programmatic overrides."""
        generate_overrides = {"logits_tv": ("data", "model")}

        sharding = Sharding(
            generate_rules=generate_overrides,
            prefill_rules={},
        )

        sharding_cfg = sharding.get_sharding_cfg()
        self.assertNotEqual(sharding_cfg.generate_rules.logits_tv,
                            (None, "model"))
        self.assertEqual(sharding_cfg.generate_rules.logits_tv,
                         ("data", "model"))

    def test_get_overrides_from_vllm_config(self):
        """Tests fetching sharding overrides from a mock VllmConfig."""

        mock_vllm_config_prefill = MagicMock()
        mock_vllm_config_prefill.additional_config = {
            "sharding": {
                "logical_rules": {
                    "all": {
                        "norm_scale": ("model", )
                    },
                    "prefill": {
                        "activation_ffw_td": ("data", "model")
                    },
                }
            }
        }
        sharding_prefill = Sharding(
            vllm_config=mock_vllm_config_prefill,
            prefill_rules={},
            generate_rules={},
        )
        prefill_overrides = sharding_prefill._get_overrides("prefill")

        self.assertEqual(prefill_overrides["norm_scale"], ("model", ))
        self.assertEqual(prefill_overrides["activation_ffw_td"],
                         ("data", "model"))

        mock_vllm_config_generate = MagicMock()
        mock_vllm_config_generate.additional_config = {
            "sharding": {
                "logical_rules": {
                    "all": {
                        "norm_scale": ("model", )
                    },
                    "prefill": {
                        "activation_ffw_td": ("data", "model")
                    },
                }
            }
        }
        sharding_generate = Sharding(
            vllm_config=mock_vllm_config_generate,
            prefill_rules={},
            generate_rules={},
        )
        generate_overrides = sharding_generate._get_overrides("generate")

        self.assertEqual(generate_overrides["norm_scale"], ("model", ))
        self.assertNotIn("activation_ffw_td", generate_overrides)


if __name__ == "__main__":
    unittest.main()
