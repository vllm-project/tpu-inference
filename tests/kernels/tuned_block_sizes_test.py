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

import json
import pathlib
import unittest
from unittest import mock

import jax.numpy as jnp

from tpu_inference.kernels.ragged_paged_attention.v3 import tuned_block_sizes


class TunedBlockSizesTest(unittest.TestCase):

    def test_rpa_registry_structure(self):
        """Walk all RPA JSON files and ensure they have valid structure."""
        # Locate the tuned_data directory relative to this test file
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
                    if data:
                        # Top level keys should be numeric strings (e.g. "128", "16")
                        key = next(iter(data))
                        self.assertTrue(
                            key.isdigit(),
                            f"{json_file.name}: Top key '{key}' should be numeric"
                        )
                except json.JSONDecodeError as e:
                    self.fail(f"Failed to parse JSON {json_file}: {e}")

    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes._load_tuning_data'
    )
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_device_name'
    )
    def test_get_tuned_block_sizes_hit(self, mock_get_device, mock_load_data):
        """Test happy path for cache hit with real key generation."""
        mock_get_device.return_value = "TPU v6e"

        # Setup mock data to match expected key structure for:
        # page_size=16, head_dim=128, q_heads=2, kv_heads=1

        # Structure: [page_size_key][dtypes][head_dims][extra]
        mock_data = {
            "16": {
                "q_bfloat16_kv_bfloat16": {
                    "q_head-2_kv_head-1_head-128": {
                        # "max_model_len-1024-sw-None"
                        "max_model_len-1024-sw-None": [64, 32]
                    }
                }
            }
        }
        mock_load_data.return_value = mock_data

        # Inputs chosen to match "q_head-2_kv_head-1_head-128" key structure:
        # q_heads=2, kv_heads=1, head_dim=128, bfloat16

        bkv_p, bq = tuned_block_sizes.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=2,
            actual_num_kv_heads=1,
            head_dim=128,
            page_size=16,
            max_num_tokens=1024,
            pages_per_seq=64,
            sliding_window=None)

        self.assertEqual(bkv_p, 64)
        self.assertEqual(bq, 32)

    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes._load_tuning_data'
    )
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_device_name'
    )
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_tpu_generation'
    )
    def test_get_tuned_block_sizes_fallback(self, mock_gen, mock_get_device,
                                            mock_load_data):
        """Test fallback to static values when data is missing."""
        mock_load_data.return_value = None
        mock_gen.return_value = 4  # TPU v4

        bkv_p, bq = tuned_block_sizes.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=1,
            actual_num_kv_heads=1,
            head_dim=128,
            page_size=16,
            max_num_tokens=1024,
            pages_per_seq=64)

        # v4 fallback: 512 // 16 = 32
        self.assertEqual(bkv_p, 32)
        self.assertEqual(bq, 32)


if __name__ == '__main__':
    unittest.main()
