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
from unittest import mock

import jax.numpy as jnp

from tpu_inference.kernels.ragged_paged_attention.v3 import tuned_block_sizes


class TunedBlockSizesTest(unittest.TestCase):

    def test_get_tuning_file_path(self):
        """Test that device names map to correct JSON paths."""
        path_v6e = tuned_block_sizes._get_tuning_file_path("TPU v6e")
        self.assertTrue(
            path_v6e.endswith(
                "tpu_inference/kernels/tuned_data/ragged_paged_attention/v3/tpu_v6e.json"
            ))

        path_v5lite = tuned_block_sizes._get_tuning_file_path("TPU v5 Lite")
        self.assertTrue(path_v5lite.endswith("tpu_v5e.json"))

    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes._load_tuning_data'
    )
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_device_name'
    )
    @mock.patch(
        'tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes.get_lookup_keys'
    )
    def test_get_tuned_block_sizes_hit(self, mock_get_keys, mock_get_device,
                                       mock_load_data):
        """Test cache hit logic."""
        # Mock keys
        mock_get_keys.return_value = (
            "tpu_v6e",
            16,  # page_size
            "q_bfloat16_kv_bfloat16",
            "dummy_dims",
            "dummy_extra")
        mock_get_device.return_value = "TPU v6e"

        # Mock data structure matching the lookup chain
        mock_data = {
            "16": {
                "q_bfloat16_kv_bfloat16": {
                    "dummy_dims": {
                        "dummy_extra": [64, 32]  # bkv_p, bq
                    }
                }
            }
        }
        mock_load_data.return_value = mock_data

        bkv_p, bq = tuned_block_sizes.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=1,
            actual_num_kv_heads=1,
            head_dim=128,
            page_size=16,
            max_num_tokens=1024,
            pages_per_seq=64)

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
    def test_get_tuned_block_sizes_fallback_v4(self, mock_gen, mock_get_device,
                                               mock_load_data):
        """Test fallback logic for TPU v4."""
        mock_load_data.return_value = None  # No data found
        mock_gen.return_value = 4

        bkv_p, bq = tuned_block_sizes.get_tuned_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=1,
            actual_num_kv_heads=1,
            head_dim=128,
            page_size=16,
            max_num_tokens=1024,
            pages_per_seq=64)

        # Fallback for v4 is (512 // page_size, 32) -> (32, 32)
        self.assertEqual(bkv_p, 32)
        self.assertEqual(bq, 32)


if __name__ == '__main__':
    unittest.main()
