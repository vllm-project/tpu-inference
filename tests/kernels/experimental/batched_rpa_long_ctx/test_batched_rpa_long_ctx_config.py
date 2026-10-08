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
"""Tests for batched RPA config and block size calculations across TPU generations."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from tpu_inference.kernels.experimental.batched_rpa_long_ctx import configs, wrapper


def _create_mock_tpu_info(
    generation: int,
    chip_version: str = "v8i",
    num_lanes: int = 128,
    mxu_column_size: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    fp8_ops_per_second: int = 2,
    bf16_ops_per_second: int = 1,
):
    mock = MagicMock()
    mock.generation = generation
    mock.chip_version = chip_version
    mock.num_lanes = num_lanes
    mock.mxu_column_size = mxu_column_size
    mock.vmem_capacity_bytes = vmem_capacity_bytes
    mock.fp8_ops_per_second = fp8_ops_per_second
    mock.bf16_ops_per_second = bf16_ops_per_second
    return mock


@pytest.mark.parametrize(
    "generation,chip_version,is_8bit,expected_bq_c_min",
    [
        (8, "v8i", True, 1),
        (8, "v8i", False, 1),
        (8, "v8t", True, 1),
        (8, "v8t", False, 1),
        (7, "v7x", True, 1),
        (7, "v7x", False, 1),
        (5, "v5p", False, 1),
    ],
)
def test_calculate_block_sizes_generation(
    generation, chip_version, is_8bit, expected_bq_c_min
):
    dtype = jnp.float8_e4m3fn if is_8bit else jnp.bfloat16
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=16,
        total_q_tokens=8,
        num_page_indices=128,
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(generation, chip_version=chip_version),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
        assert decode_blocks.bq_sz >= decode_blocks.bq_c_sz >= expected_bq_c_min
        assert prefill_blocks.bq_sz >= prefill_blocks.bq_c_sz >= expected_bq_c_min


def test_calculate_block_sizes_cp_pcp():
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=2,
        page_size=128,
        total_q_tokens=1024,
        num_page_indices=128,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        cp_group_size=4,
        pcp_ring_axis_name="pcp",
        return_lse=True,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x"),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
        assert prefill_blocks.n_buffer == 3
        assert prefill_blocks.batch_size == 1
        assert prefill_blocks.bq_sz >= 128
        assert prefill_blocks.bkv_sz >= 128

