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

from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from tpu_inference.kernels.flash_attention.kernel import (
    BlockSizes, calculate_vmem_usage_bytes)

jax.config.parse_flags_with_absl()


class MockTpuInfo:
    num_lanes = 128
    num_sublanes = 8


class FlashAttentionVmemEstimationTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        # Mock get_tpu_info to return predictable values and allow CPU testing
        self.mock_get_tpu_info = self.enter_context(
            mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                       return_value=MockTpuInfo()))

    def test_calculate_vmem_usage_bytes_base(self):
        """Tests the base memory calculation with standard bf16 types."""
        block_sizes = BlockSizes(block_b=1,
                                 block_q=128,
                                 block_k_major=128,
                                 block_k=128)

        # d_model=64 should be padded to 128 lanes
        # q_o_elements = 1 * 128 * 128 = 16384
        # q_o_vmem = 16384 * 2 (bf16) * 2 (Q & O) = 65536
        # kv_elements = 1 * 128 * 128 = 16384
        # kv_vmem = 16384 * 2 (bf16) * 2 (K & V) = 65536
        # logits_elements = 128 * 128 = 16384
        # logits_vmem = 16384 * 4 (fp32) * 2 (S & P) = 131072
        # Expected = 65536 + 65536 + 131072 = 262144

        total_bytes = calculate_vmem_usage_bytes(block_sizes,
                                                 q_dtype=jnp.bfloat16,
                                                 kv_dtype=jnp.bfloat16,
                                                 d_model=64,
                                                 kv_seq_len=128)
        self.assertEqual(total_bytes, 262144)

    def test_calculate_vmem_usage_bytes_alignment(self):
        """Verifies that d_model is correctly padded to lane count (128)."""
        block_sizes = BlockSizes(block_b=1,
                                 block_q=128,
                                 block_k_major=128,
                                 block_k=128)

        # d_model=64 and d_model=128 should yield the same VMEM because both pad to 128
        total_bytes_64 = calculate_vmem_usage_bytes(block_sizes,
                                                    q_dtype=jnp.bfloat16,
                                                    kv_dtype=jnp.bfloat16,
                                                    d_model=64,
                                                    kv_seq_len=128)

        total_bytes_128 = calculate_vmem_usage_bytes(block_sizes,
                                                     q_dtype=jnp.bfloat16,
                                                     kv_dtype=jnp.bfloat16,
                                                     d_model=128,
                                                     kv_seq_len=128)

        self.assertEqual(total_bytes_64, total_bytes_128)

        # d_model=129 should pad to 256, increasing memory
        total_bytes_129 = calculate_vmem_usage_bytes(block_sizes,
                                                     q_dtype=jnp.bfloat16,
                                                     kv_dtype=jnp.bfloat16,
                                                     d_model=129,
                                                     kv_seq_len=128)
        self.assertGreater(total_bytes_129, total_bytes_128)

    def test_calculate_vmem_usage_bytes_dtypes(self):
        """Tests that memory estimation adapts to different data types."""
        block_sizes = BlockSizes(block_b=1,
                                 block_q=128,
                                 block_k_major=128,
                                 block_k=128)

        # Q as fp32 (4 bytes) instead of bf16 (2 bytes)
        # q_o_vmem should double from 65536 to 131072
        # Total = 131072 (Q&O) + 65536 (K&V) + 131072 (Logits) = 327680
        total_bytes_fp32 = calculate_vmem_usage_bytes(block_sizes,
                                                      q_dtype=jnp.float32,
                                                      kv_dtype=jnp.bfloat16,
                                                      d_model=64,
                                                      kv_seq_len=128)
        self.assertEqual(total_bytes_fp32, 327680)

    def test_calculate_vmem_usage_bytes_optional_ab(self):
        """Tests memory calculation with Attention Bias."""
        block_sizes = BlockSizes(block_b=1,
                                 block_q=128,
                                 block_k_major=128,
                                 block_k=128)

        base_bytes = calculate_vmem_usage_bytes(block_sizes,
                                                q_dtype=jnp.bfloat16,
                                                kv_dtype=jnp.bfloat16,
                                                d_model=64,
                                                kv_seq_len=128)

        # Mock Attention Bias tensor
        ab = jnp.zeros((1, 1, 128, 128), dtype=jnp.bfloat16)
        # ab_vmem = 1 * 128 * 128 * 2 (bf16) = 32768

        total_bytes_ab = calculate_vmem_usage_bytes(block_sizes,
                                                    q_dtype=jnp.bfloat16,
                                                    kv_dtype=jnp.bfloat16,
                                                    d_model=64,
                                                    kv_seq_len=128,
                                                    ab=ab)

        self.assertEqual(total_bytes_ab, base_bytes + 32768)

    def test_calculate_vmem_usage_bytes_optional_segment_ids(self):
        """Tests memory calculation with Segment IDs."""
        block_sizes = BlockSizes(block_b=1,
                                 block_q=128,
                                 block_k_major=128,
                                 block_k=128)

        base_bytes = calculate_vmem_usage_bytes(block_sizes,
                                                q_dtype=jnp.bfloat16,
                                                kv_dtype=jnp.bfloat16,
                                                d_model=64,
                                                kv_seq_len=128)

        # segment_vmem = (1 * 128 * 128 * 4) + (1 * 8 * 128 * 4) = 65536 + 4096 = 69632
        total_bytes_seg = calculate_vmem_usage_bytes(
            block_sizes,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            d_model=64,
            kv_seq_len=128,
            segment_ids=True)  # Non-None triggers calculation

        self.assertEqual(total_bytes_seg, base_bytes + 69632)


if __name__ == "__main__":
    absltest.main()
