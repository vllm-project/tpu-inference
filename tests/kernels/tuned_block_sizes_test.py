# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import unittest
from unittest import mock

import jax.numpy as jnp

from tpu_inference.kernels.quantized_matmul import \
    tuned_block_sizes as matmul_tuning
from tpu_inference.kernels.ragged_paged_attention.v3 import \
    tuned_block_sizes as rpa_tuning


class QuantizedMatmulTuningTest(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test to ensure isolation
        matmul_tuning._TUNING_DATA_CACHE.clear()

    @mock.patch.object(matmul_tuning, '_load_tuning_data')
    @mock.patch(
        'tpu_inference.kernels.quantized_matmul.tuned_block_sizes.get_tpu_name_slug'
    )
    def test_logic_with_mock_data(self, mock_get_slug, mock_load_data):
        """Verify that get_tuned_block_sizes returns the correct values from a mocked registry."""
        mock_get_slug.return_value = 'tpu_v7'

        # Mock data structure: { "n_batch,n_out,n_in,x_dtype,w_dtype": [B, O, I] }
        mock_data = {
            "1,1024,1024,int8,int8": [128, 256, 512],
            "config_style_key": {
                "config": {
                    "batch_block_size": 64,
                    "out_block_size": 64,
                    "in_block_size": 64
                }
            }
        }
        mock_load_data.return_value = mock_data

        # Test standard list format
        val = matmul_tuning.get_tuned_block_sizes(n_batch=1,
                                                  n_out=1024,
                                                  n_in=1024,
                                                  x_q_dtype='int8',
                                                  w_q_dtype='int8')
        self.assertEqual(val.batch_block_size, 128)
        self.assertEqual(val.out_block_size, 256)
        self.assertEqual(val.in_block_size, 512)

    @mock.patch.object(matmul_tuning, '_load_tuning_data')
    @mock.patch(
        'tpu_inference.kernels.quantized_matmul.tuned_block_sizes.get_tpu_name_slug'
    )
    @mock.patch(
        'tpu_inference.kernels.quantized_matmul.tuned_block_sizes.get_tpu_generation'
    )
    def test_default_fallback(self, mock_get_gen, mock_get_slug,
                              mock_load_data):
        """Verify fallback to defaults when key is missing."""
        mock_get_slug.return_value = 'tpu_v7'
        mock_load_data.return_value = {}  # Empty registry
        mock_get_gen.return_value = 7  # Just needed for logging if checking that, or if logic changes

        val = matmul_tuning.get_tuned_block_sizes(
            n_batch=999,  # Unknown
            n_out=999,
            n_in=999,
            x_q_dtype='int8',
            w_q_dtype='int8')
        # Expect defaults from code
        self.assertEqual(val.batch_block_size, 128)
        self.assertEqual(val.out_block_size, 128)
        self.assertEqual(val.in_block_size, 128)

    def test_registry_integrity(self):
        """Walk all Matmul JSON files and ensure they are valid."""
        # Assume repo structure relative to this test file
        # tpu_inference/kernels/tuned_data/quantized_matmul/*.json
        base_path = pathlib.Path(__file__).parent.parent.parent
        data_dir = base_path / "tpu_inference" / "kernels" / "tuned_data" / "quantized_matmul"

        self.assertTrue(data_dir.exists(), f"Directory not found: {data_dir}")

        json_files = list(data_dir.glob("*.json"))
        self.assertTrue(len(json_files) > 0, "No JSON files found for Matmul")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    self.assertIsInstance(data, dict,
                                          f"Root must be dict in {json_file}")
                    # Basic schema check for a random key if not empty
                    if data:
                        key = next(iter(data))
                        val = data[key]
                        # Must be list or dict with config
                        valid_val = (isinstance(val, list) and len(val) == 3) or \
                                    (isinstance(val, dict) and "config" in val)
                        self.assertTrue(
                            valid_val,
                            f"Invalid value format in {json_file} for key {key}"
                        )

                except json.JSONDecodeError as e:
                    self.fail(f"Failed to parse JSON {json_file}: {e}")


class RaggedPagedAttentionTuningTest(unittest.TestCase):

    def setUp(self):
        rpa_tuning._TUNING_DATA_CACHE.clear()

    @mock.patch.object(rpa_tuning, '_load_tuning_data')
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_device_name'
    )
    def test_logic_with_mock_data(self, mock_get_dev, mock_load_data):
        """Verify successful lookup logic with mocked data."""
        mock_get_dev.return_value = "TPU v5 Lite"  # Maps to tpu_v5e

        # Construct a key that get_lookup_keys would generate
        # Key format: (device, page_size, dtypes, head_dims, extra)
        # but the JSON is nested: data[page_size][dtypes][head_dims][extra]

        # Let's mock the JSON structure directly
        mock_data = {
            "16": {  # page_size
                "q_bfloat16_kv_float8_e4m3fn": {
                    "q_head-128_kv_head-16_head-128": {
                        "max_model_len-16384-sw-None": [8, 32]  # bkv_p, bq
                    }
                }
            }
        }
        mock_load_data.return_value = mock_data

        bkv_p, bq = rpa_tuning.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.float8_e4m3fn,
            actual_num_q_heads=128,
            actual_num_kv_heads=16,
            head_dim=128,
            page_size=16,
            max_num_tokens=1024,
            pages_per_seq=1024,
            sliding_window=None)
        self.assertEqual(bkv_p, 8)
        self.assertEqual(bq, 32)

    @mock.patch.object(rpa_tuning, '_load_tuning_data')
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_device_name'
    )
    def test_capping_logic(self, mock_get_dev, mock_load_data):
        """Verify that block sizes are capped by sequence length/token count."""
        mock_data = {
            "16": {
                "q_bfloat16_kv_bfloat16": {
                    "q_head-128_kv_head-128_head-128": {
                        "max_model_len-8192-sw-None": [64, 64]  # Big blocks
                    }
                }
            }
        }
        mock_load_data.return_value = mock_data

        # Request with small seq limits
        bkv_p, bq = rpa_tuning.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=128,
            actual_num_kv_heads=128,
            head_dim=128,
            page_size=16,
            max_num_tokens=32,  # Cap bq to 32
            pages_per_seq=10,  # Cap bkv_p to 10
            sliding_window=None)

        self.assertEqual(bkv_p, 10)  # 10 < 64
        self.assertEqual(bq, 32)  # 32 < 64

    def test_hd64_key_logic(self):
        """Verify that head_dim=64 triggers the specific legacy key generation."""
        # we can test get_simplified_raw_key directly

        # Case 1: Standard (head_dim=128)
        # Should align head to 128
        keys = rpa_tuning.get_simplified_raw_key(page_size=16,
                                                 q_dtype=jnp.bfloat16,
                                                 kv_dtype=jnp.bfloat16,
                                                 actual_num_q_heads=32,
                                                 actual_num_kv_heads=32,
                                                 head_dim=128,
                                                 max_model_len=1000,
                                                 sliding_window=None)
        # keys[5] is head_dim
        self.assertEqual(keys[5], 128)

        # Case 2: HD64
        # Should KEEP head_dim as 64 (not align to 128)
        keys_64 = rpa_tuning.get_simplified_raw_key(page_size=16,
                                                    q_dtype=jnp.bfloat16,
                                                    kv_dtype=jnp.bfloat16,
                                                    actual_num_q_heads=32,
                                                    actual_num_kv_heads=32,
                                                    head_dim=64,
                                                    max_model_len=1000,
                                                    sliding_window=None)
        self.assertEqual(keys_64[5], 64)

    def test_registry_integrity(self):
        """Walk all RPA JSON files and ensure they are valid."""
        base_path = pathlib.Path(__file__).parent.parent.parent
        data_dir = base_path / "tpu_inference" / "kernels" / "tuned_data" / "ragged_paged_attention" / "v3"

        self.assertTrue(data_dir.exists(), f"Directory not found: {data_dir}")
        json_files = list(data_dir.glob("*.json"))
        self.assertTrue(len(json_files) > 0, "No JSON files found for RPA")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    self.assertIsInstance(data, dict)

                    # RPA structure is deeply nested.
                    # If it has "512" (partition size) or "16" (page size), check deeper

                    # Just verify it's not empty and has expected string keys
                    if data:
                        key = next(iter(data))
                        # Should be a number string like "128", "16", "32" etc.
                        self.assertTrue(
                            key.isdigit(),
                            f"Top level keys in {json_file.name} should be numeric strings, got {key}"
                        )

                except json.JSONDecodeError as e:
                    self.fail(f"Failed to parse JSON {json_file}: {e}")


if __name__ == '__main__':
    unittest.main()
