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

import math
import jax
import numpy as np
import pytest
import torch

from tpu_inference.models.vllm.experimental.vision_tower_jit import GridTHW


def test_grid_thw_basic_and_slicing():
    # Test creation and representation
    grid = GridTHW([(1, 28, 28), (2, 14, 14), (4, 7, 7)])
    assert len(grid) == 3
    assert grid.shape == (3, 3)
    assert grid.ndim == 2
    assert grid.tolist() == [[1, 28, 28], [2, 14, 14], [4, 7, 7]]
    assert np.array_equal(grid.prod(), np.array([1 * 28 * 28, 2 * 14 * 14, 4 * 7 * 7]))

    # Test indexing
    assert grid[0] == (1, 28, 28)

    # Test slicing preserves GridTHW type
    sliced = grid[1:]
    assert isinstance(sliced, GridTHW)
    assert len(sliced) == 2
    assert sliced.tolist() == [[2, 14, 14], [4, 7, 7]]


def test_grid_thw_pytree_roundtrip():
    grid = GridTHW([(1, 16, 16), (2, 32, 32)])
    leaves, treedef = jax.tree_util.tree_flatten(grid)
    assert leaves == []

    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, GridTHW)
    assert restored.tolist() == [[1, 16, 16], [2, 32, 32]]


def test_padding_factorization_math():
    # Simulate various TP sizes and merge sizes
    test_cases = [
        # (tp_size, spatial_merge_size, seq_len)
        (8, 2, 100),   # merge_factor = 4, pad_factor = lcm(8, 4) = 8. 100 -> 104 (pad 4)
        (4, 2, 104),   # merge_factor = 4, pad_factor = lcm(4, 4) = 4. 104 -> 104 (pad 0)
        (8, 1, 15),    # merge_factor = 1, pad_factor = lcm(8, 1) = 8. 15 -> 16 (pad 1)
        (16, 2, 252),  # merge_factor = 4, pad_factor = lcm(16, 4) = 16. 252 -> 256 (pad 4)
    ]

    for tp_size, merge_size, seq_len in test_cases:
        merge_factor = merge_size * merge_size
        pad_factor = math.lcm(tp_size, merge_factor)

        if seq_len % pad_factor != 0:
            pad_seq = pad_factor - (seq_len % pad_factor)
        else:
            pad_seq = 0

        total_seq = seq_len + pad_seq
        assert total_seq % tp_size == 0
        assert total_seq % merge_factor == 0

        if pad_seq > 0:
            dummy_grid_thw = (pad_seq // merge_factor, merge_size, merge_size)
            # Spatial dimensions must be divisible by merge_size
            assert dummy_grid_thw[1] % merge_size == 0
            assert dummy_grid_thw[2] % merge_size == 0
            # Total patches in dummy grid equals pad_seq
            assert dummy_grid_thw[0] * dummy_grid_thw[1] * dummy_grid_thw[2] == pad_seq


def test_multimodal_unpadding_stripping():
    # Verify exact numerical unpadding logic
    original_pixels_len = 100
    merge_size = 2
    merge_factor = merge_size * merge_size
    tp_size = 8
    pad_factor = math.lcm(tp_size, merge_factor)

    pad_seq = pad_factor - (original_pixels_len % pad_factor)
    total_pixels_len = original_pixels_len + pad_seq

    # Simulated vision tower output: (total_patches // merge_factor, hidden_dim)
    hidden_dim = 64
    total_tokens = total_pixels_len // merge_factor
    simulated_output = torch.randn((total_tokens, hidden_dim))

    # Strip logic
    original_tokens_len = original_pixels_len // merge_factor
    stripped_output = simulated_output[:original_tokens_len, :]

    assert stripped_output.shape == (original_pixels_len // merge_factor, hidden_dim)


def test_get_model_tp_size_helper():
    from unittest.mock import MagicMock
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

    wrapper = MagicMock(spec=VllmModelWrapper)
    wrapper._get_model_tp_size = VllmModelWrapper._get_model_tp_size.__get__(wrapper)

    # Case 1: Mesh with "model" axis present
    mock_mesh = MagicMock()
    mock_mesh.shape = {"data": 1, "model": 4}
    wrapper.mesh = mock_mesh
    assert wrapper._get_model_tp_size() == 4

    # Case 2: Mesh without "model" axis, fallback to parallel_config
    mock_mesh.shape = {"data": 2}
    wrapper.vllm_config.parallel_config.tensor_parallel_size = 2
    assert wrapper._get_model_tp_size() == 2

    # Case 3: Mesh is None, fallback to parallel_config
    wrapper.mesh = None
    wrapper.vllm_config.parallel_config.tensor_parallel_size = 8
    assert wrapper._get_model_tp_size() == 8


def test_get_activation_sharding_divisor():
    from unittest.mock import MagicMock
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

    wrapper = MagicMock(spec=VllmModelWrapper)
    wrapper._get_activation_sharding_divisor = (
        VllmModelWrapper._get_activation_sharding_divisor.__get__(wrapper)
    )

    # Case 1: DP attention mesh ('attn_dp': 8, 'model': 1)
    mock_mesh = MagicMock()
    mock_mesh.shape = {"data": 1, "attn_dp": 8, "model": 1}
    wrapper.mesh = mock_mesh
    assert wrapper._get_activation_sharding_divisor() == 8

    # Case 2: Hybrid PCP + DP attention ('pcp': 2, 'attn_dp': 4)
    mock_mesh.shape = {"data": 1, "pcp": 2, "attn_dp": 4, "model": 1}
    assert wrapper._get_activation_sharding_divisor() == 4  # lcm(1, 2, 4, 1) = 4

    # Case 3: Fallback when mesh is None
    wrapper.mesh = None
    wrapper.vllm_config.parallel_config.tensor_parallel_size = 8
    assert wrapper._get_activation_sharding_divisor() == 8


def test_padding_metadata_synchronization():
    # Initial tensors for 1 video (100 patches, 1 timestamp entry)
    grid = GridTHW([(1, 10, 10)])
    second_per_grid_ts_tensor = torch.tensor([1.5])
    second_per_grid_ts_list = [1.5]
    timestamps_tensor = torch.tensor([[0.0, 1.0]])
    timestamps_list = [[0.0, 1.0]]
    pixels = torch.randn((100, 16))

    tp_size = 8
    spatial_merge_size = 2
    merge_factor = spatial_merge_size * spatial_merge_size  # 4
    pad_factor = math.lcm(tp_size, merge_factor)  # 8

    # Sequence padding
    seq_len = pixels.shape[0]
    pad_seq = pad_factor - (seq_len % pad_factor)  # 4
    dummy_grid_thw = (max(1, pad_seq // merge_factor), spatial_merge_size, spatial_merge_size)  # (1, 2, 2)

    pad_pixels_seq = torch.zeros((pad_seq, *pixels.shape[1:]))
    pixels = torch.cat([pixels, pad_pixels_seq], dim=0)

    grid_list = list(grid)
    grid_list.append(dummy_grid_thw)
    grid = GridTHW(grid_list)

    # Tensor padding
    pad_ts = torch.full((1,), second_per_grid_ts_tensor[-1].item())
    second_per_grid_ts_tensor = torch.cat([second_per_grid_ts_tensor, pad_ts], dim=0)
    pad_ts_vals = timestamps_tensor[-1].unsqueeze(0)
    timestamps_tensor = torch.cat([timestamps_tensor, pad_ts_vals], dim=0)

    # List padding
    second_per_grid_ts_list = second_per_grid_ts_list + [second_per_grid_ts_list[-1]]
    timestamps_list = timestamps_list + [timestamps_list[-1]]

    # All arrays must match in length after sequence padding
    assert len(grid) == 2
    assert len(second_per_grid_ts_tensor) == 2
    assert len(timestamps_tensor) == 2
    assert len(second_per_grid_ts_list) == 2
    assert len(timestamps_list) == 2
    assert pixels.shape[0] % tp_size == 0



