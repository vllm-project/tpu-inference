# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

from vllm.config import VllmConfig, ModelConfig, ParallelConfig

from tpu_inference.core.sched.utils import get_dp_size


class TestSchedUtils(unittest.TestCase):
    """Test cases for scheduler utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock VllmConfig for testing
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        
        # Create mock model config with HF config
        self.mock_model_config = MagicMock(spec=ModelConfig)
        self.mock_hf_config = MagicMock()
        self.mock_model_config.hf_config = self.mock_hf_config
        
        # Create mock parallel config
        self.mock_parallel_config = MagicMock(spec=ParallelConfig)
        
        # Set up the mock config structure
        self.mock_vllm_config.model_config = self.mock_model_config
        self.mock_vllm_config.parallel_config = self.mock_parallel_config

    def test_get_dp_size_with_explicit_config(self):
        """Test get_dp_size when data parallelism is explicitly configured."""
        # Set up config with explicit DP size
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 8
                }
            }
        }
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 4
        self.mock_parallel_config.tensor_parallel_size = 8
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 8)
        self.assertEqual(attn_dp, 2)  # max(8 // 4, 1) = 2
        self.assertEqual(total_dp, 16)  # 8 * 2 = 16

    def test_get_dp_size_without_explicit_config(self):
        """Test get_dp_size when no explicit DP config is provided."""
        # Set up config without DP configuration
        self.mock_vllm_config.additional_config = {}
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 8
        self.mock_parallel_config.tensor_parallel_size = 16
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 1)  # Default when no config
        self.assertEqual(attn_dp, 2)  # max(16 // 8, 1) = 2
        self.assertEqual(total_dp, 2)  # 1 * 2 = 2

    def test_get_dp_size_missing_additional_config(self):
        """Test get_dp_size when additional_config is missing entirely."""
        # Remove additional_config attribute
        delattr(self.mock_vllm_config, 'additional_config')
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 2
        self.mock_parallel_config.tensor_parallel_size = 4
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 1)  # Default when no config
        self.assertEqual(attn_dp, 2)  # max(4 // 2, 1) = 2
        self.assertEqual(total_dp, 2)  # 1 * 2 = 2

    def test_get_dp_size_none_additional_config(self):
        """Test get_dp_size when additional_config is None."""
        self.mock_vllm_config.additional_config = None
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 1
        self.mock_parallel_config.tensor_parallel_size = 8
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 1)  # Default when config is None
        self.assertEqual(attn_dp, 8)  # max(8 // 1, 1) = 8
        self.assertEqual(total_dp, 8)  # 1 * 8 = 8

    def test_get_dp_size_incomplete_sharding_config(self):
        """Test get_dp_size with incomplete sharding configuration."""
        # Set up incomplete config (missing data_parallelism key)
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {}
            }
        }
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 4
        self.mock_parallel_config.tensor_parallel_size = 4
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 1)  # Default when key is missing
        self.assertEqual(attn_dp, 1)  # max(4 // 4, 1) = 1
        self.assertEqual(total_dp, 1)  # 1 * 1 = 1

    def test_get_dp_size_missing_model_config(self):
        """Test get_dp_size when model_config is missing."""
        # Set up DP config
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 4
                }
            }
        }
        
        # Remove model_config
        delattr(self.mock_vllm_config, 'model_config')
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 4)
        self.assertEqual(attn_dp, 1)  # Default when model config missing
        self.assertEqual(total_dp, 4)  # 4 * 1 = 4

    def test_get_dp_size_missing_hf_config(self):
        """Test get_dp_size when hf_config is missing."""
        # Set up DP config
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 2
                }
            }
        }
        
        # Remove hf_config
        delattr(self.mock_model_config, 'hf_config')
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 2)
        self.assertEqual(attn_dp, 1)  # Default when hf_config missing
        self.assertEqual(total_dp, 2)  # 2 * 1 = 2

    def test_get_dp_size_missing_parallel_config(self):
        """Test get_dp_size when parallel_config is missing."""
        # Set up DP config
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 3
                }
            }
        }
        
        # Set up model config with HF config
        self.mock_hf_config.num_key_value_heads = 8
        
        # Remove parallel_config
        delattr(self.mock_vllm_config, 'parallel_config')
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 3)
        self.assertEqual(attn_dp, 1)  # Default when parallel_config missing
        self.assertEqual(total_dp, 3)  # 3 * 1 = 3

    def test_get_dp_size_attn_dp_minimum_one(self):
        """Test that attn_dp is at least 1 even when TP < num_kv_heads."""
        # Set up config
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 2
                }
            }
        }
        
        # Set up attention parameters where TP < num_kv_heads
        self.mock_hf_config.num_key_value_heads = 16
        self.mock_parallel_config.tensor_parallel_size = 4
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 2)
        self.assertEqual(attn_dp, 1)  # max(4 // 16, 1) = max(0, 1) = 1
        self.assertEqual(total_dp, 2)  # 2 * 1 = 2

    def test_get_dp_size_large_numbers(self):
        """Test get_dp_size with large numbers."""
        # Set up config with large DP size
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 128
                }
            }
        }
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 8
        self.mock_parallel_config.tensor_parallel_size = 64
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 128)
        self.assertEqual(attn_dp, 8)  # max(64 // 8, 1) = 8
        self.assertEqual(total_dp, 1024)  # 128 * 8 = 1024

    def test_get_dp_size_zero_dp_config(self):
        """Test get_dp_size when DP config is zero (edge case)."""
        # Set up config with zero DP size
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 0
                }
            }
        }
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 4
        self.mock_parallel_config.tensor_parallel_size = 8
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        self.assertEqual(dp_size, 0)  # Returns configured value even if zero
        self.assertEqual(attn_dp, 2)  # max(8 // 4, 1) = 2
        self.assertEqual(total_dp, 0)  # 0 * 2 = 0

    def test_get_dp_size_nested_dict_access_errors(self):
        """Test various nested dictionary access error scenarios."""
        test_cases = [
            # Missing sharding key
            {"other_key": "value"},
            # Missing sharding_strategy key
            {"sharding": {"other_key": "value"}},
            # sharding is not a dict
            {"sharding": "not_a_dict"},
            # sharding_strategy is not a dict
            {"sharding": {"sharding_strategy": "not_a_dict"}},
        ]
        
        for config in test_cases:
            with self.subTest(config=config):
                self.mock_vllm_config.additional_config = config
                
                # Set up attention parameters
                self.mock_hf_config.num_key_value_heads = 4
                self.mock_parallel_config.tensor_parallel_size = 8
                
                dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
                
                self.assertEqual(dp_size, 1)  # Should default to 1
                self.assertEqual(attn_dp, 2)  # max(8 // 4, 1) = 2
                self.assertEqual(total_dp, 2)  # 1 * 2 = 2

    def test_get_dp_size_return_types(self):
        """Test that get_dp_size returns integers."""
        # Set up basic config
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 4
                }
            }
        }
        
        # Set up attention parameters
        self.mock_hf_config.num_key_value_heads = 2
        self.mock_parallel_config.tensor_parallel_size = 4
        
        dp_size, attn_dp, total_dp = get_dp_size(self.mock_vllm_config)
        
        # Verify all return values are integers
        self.assertIsInstance(dp_size, int)
        self.assertIsInstance(attn_dp, int)
        self.assertIsInstance(total_dp, int)


if __name__ == '__main__':
    unittest.main()