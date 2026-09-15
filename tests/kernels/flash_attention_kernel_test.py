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
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.flash_attention.kernel import (
    BlockSizes, calculate_vmem_usage_bytes, flash_attention)

jax.config.parse_flags_with_absl()


class MockTpuInfo:
    num_lanes = 128
    num_sublanes = 8
    vmem_capacity_bytes = 32 * 1024 * 1024


class FlashAttentionVmemEstimationTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.enter_context(jax.disable_jit())
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

    def test_vmem_estimation_sequence_length_guard(self):
        """Tests that long sequences exceeding VMEM capacity do not override block_k."""
        block_sizes = BlockSizes.get_default(1, 16, 30976, 30976, 72)

        # Calculate VMEM for MMMU 30,976 visual tokens
        total_bytes_30k = calculate_vmem_usage_bytes(
            block_sizes,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            d_model=72,
            kv_seq_len=30976,
        )

        vmem_capacity = pltpu.get_tpu_info().vmem_capacity_bytes
        # 30,976 tokens require ~46MB, which exceeds the physical VMEM budget
        self.assertGreater(total_bytes_30k, vmem_capacity)

    def test_vmem_estimation_short_sequence_fits(self):
        """Tests that short sequences (e.g. 512, 1024) require well below 32MB VMEM."""
        block_sizes = BlockSizes.get_default(1, 16, 1024, 1024, 128)
        total_bytes_1k = calculate_vmem_usage_bytes(
            block_sizes,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            d_model=128,
            kv_seq_len=1024,
        )
        # 1024 tokens require ~2MB, well within budget
        self.assertLess(total_bytes_1k, 5 * 1024 * 1024)

    def test_dynamic_block_sizing_override_for_short_sequence(self):
        """Verifies that short sequences override block_k to kv_seq_len when VMEM allows."""
        q = jnp.zeros((1, 16, 1024, 64), dtype=jnp.bfloat16)
        k = jnp.zeros((1, 16, 1024, 64), dtype=jnp.bfloat16)
        v = jnp.zeros((1, 16, 1024, 64), dtype=jnp.bfloat16)

        with mock.patch(
                "tpu_inference.kernels.flash_attention.kernel._flash_attention"
        ) as mock_flash:
            flash_attention(q, k, v)
            mock_flash.assert_called_once()
            _, kwargs = mock_flash.call_args
            # For 1024 tokens with d_model=64, estimated VMEM is ~2MB, well below 32MB * 0.9.
            # Thus block_k and block_k_major should be overridden to 1024.
            passed_block_sizes = kwargs.get(
                "block_sizes") or mock_flash.call_args[0][8]
            self.assertEqual(passed_block_sizes.block_k, 1024)
            self.assertEqual(passed_block_sizes.block_k_major, 1024)

    def test_dynamic_block_sizing_guard_for_large_sequence(self):
        """Verifies that large sequences (e.g. 30k tokens) do NOT override block_k to prevent VMEM OOM."""
        q = jnp.zeros((1, 16, 128, 72), dtype=jnp.bfloat16)
        k = jnp.zeros((1, 16, 30976, 72), dtype=jnp.bfloat16)
        v = jnp.zeros((1, 16, 30976, 72), dtype=jnp.bfloat16)

        with mock.patch(
                "tpu_inference.kernels.flash_attention.kernel._flash_attention"
        ) as mock_flash:
            flash_attention(q, k, v)
            mock_flash.assert_called_once()
            passed_block_sizes = mock_flash.call_args[0][8]
            # 30,976 tokens require ~47.6MB, exceeding 32MB * 0.9 (28.8MB).
            # BlockSizes should NOT override block_k to 30976; it must retain default block size (128).
            self.assertNotEqual(passed_block_sizes.block_k, 30976)
            self.assertEqual(passed_block_sizes.block_k, 128)

    def test_dynamic_block_sizing_custom_vmem_limit(self):
        """Verifies that an explicitly passed vmem_limit_bytes is strictly respected."""
        q = jnp.zeros((1, 16, 128, 128), dtype=jnp.bfloat16)
        k = jnp.zeros((1, 16, 4096, 128), dtype=jnp.bfloat16)
        v = jnp.zeros((1, 16, 4096, 128), dtype=jnp.bfloat16)

        # 4096 tokens with d_model=128 requires ~6.06MB.
        # Under default 32MB limit (32 * 0.9 = 28.8MB), it fits and overrides block_k to 4096.
        # But with a constrained 4MB limit (4 * 0.9 = 3.6MB), it exceeds the limit and does NOT override.
        with mock.patch(
                "tpu_inference.kernels.flash_attention.kernel._flash_attention"
        ) as mock_flash:
            flash_attention(q, k, v, vmem_limit_bytes=4 * 1024 * 1024)
            mock_flash.assert_called_once()
            passed_block_sizes = mock_flash.call_args[0][8]
            self.assertNotEqual(passed_block_sizes.block_k, 4096)
            self.assertEqual(passed_block_sizes.block_k, 128)


if __name__ == "__main__":
    absltest.main()
