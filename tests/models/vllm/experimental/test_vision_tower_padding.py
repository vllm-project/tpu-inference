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
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from vllm.model_executor.models.qwen3_omni_moe_thinker import \
    Qwen3OmniMoeThinkerForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from tpu_inference.models.vllm.experimental.vision_tower_jit import (
    JITTABLE_ARCHS, GridTHW, is_jittable_architecture)
from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper


def _make_stub_wrapper(tp_size: int = 8,
                       spatial_merge_size: int = 2,
                       mesh_shape: dict | None = None) -> SimpleNamespace:
    wrapper = SimpleNamespace()
    wrapper.mesh = SimpleNamespace(
        shape=mesh_shape if mesh_shape is not None else {
            "data": 1,
            "model": tp_size,
        })
    wrapper.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
        tensor_parallel_size=tp_size))
    wrapper.model = SimpleNamespace(vllm_model=SimpleNamespace(
        visual=SimpleNamespace(spatial_merge_size=spatial_merge_size)))
    wrapper._get_model_tp_size = VllmModelWrapper._get_model_tp_size.__get__(
        wrapper)
    wrapper._get_activation_sharding_divisor = (
        VllmModelWrapper._get_activation_sharding_divisor.__get__(wrapper))
    wrapper._get_spatial_merge_size = (
        VllmModelWrapper._get_spatial_merge_size.__get__(wrapper))
    wrapper._maybe_pad_multimodal_kwargs = (
        VllmModelWrapper._maybe_pad_multimodal_kwargs.__get__(wrapper))
    wrapper._maybe_unpad_multimodal_output = (
        VllmModelWrapper._maybe_unpad_multimodal_output)
    wrapper._make_multimodal_move_fn = (
        VllmModelWrapper._make_multimodal_move_fn.__get__(wrapper))
    return wrapper


def test_grid_thw_basic_and_slicing():
    grid = GridTHW([(1, 28, 28), (2, 14, 14), (4, 7, 7)])
    assert len(grid) == 3
    assert grid.shape == (3, 3)
    assert grid.ndim == 2
    assert grid.tolist() == [[1, 28, 28], [2, 14, 14], [4, 7, 7]]
    assert np.array_equal(grid.prod(),
                          np.array([1 * 28 * 28, 2 * 14 * 14, 4 * 7 * 7]))
    assert grid[0] == (1, 28, 28)

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


@pytest.mark.parametrize(
    "tp_size,merge_size,seq_len,expected_padded",
    [
        (8, 2, 100, True),  # lcm(8, 4) = 8 -> 100 pads to 104 (+4)
        (4, 2, 104, False),  # lcm(4, 4) = 4 -> 104 already aligned
        (8, 1, 15, True),  # lcm(8, 1) = 8 -> 15 pads to 16 (+1)
        (16, 2, 252, True),  # lcm(16, 4) = 16 -> 252 pads to 256 (+4)
    ],
)
def test_maybe_pad_multimodal_kwargs_image(tp_size, merge_size, seq_len,
                                           expected_padded):
    wrapper = _make_stub_wrapper(tp_size=tp_size,
                                 spatial_merge_size=merge_size)
    kwargs = {
        "pixel_values": torch.ones((seq_len, 64)),
        "image_grid_thw": GridTHW([(1, merge_size, seq_len // merge_size)]),
    }
    padded_kwargs, padded, orig_batch, orig_pixels, merge_factor = (
        wrapper._maybe_pad_multimodal_kwargs(kwargs))

    assert padded is expected_padded
    assert orig_batch == 1
    assert orig_pixels == seq_len
    assert merge_factor == merge_size * merge_size

    pad_factor = math.lcm(tp_size, merge_factor)
    assert padded_kwargs["pixel_values"].shape[0] % pad_factor == 0
    if expected_padded:
        assert len(padded_kwargs["image_grid_thw"]) == 2
        dummy_t, dummy_h, dummy_w = padded_kwargs["image_grid_thw"][-1]
        assert dummy_h == merge_size and dummy_w == merge_size
        assert (dummy_t * dummy_h *
                dummy_w == padded_kwargs["pixel_values"].shape[0] - seq_len)
    else:
        assert len(padded_kwargs["image_grid_thw"]) == 1


def test_maybe_pad_multimodal_kwargs_video_metadata_sync():
    wrapper = _make_stub_wrapper(tp_size=8, spatial_merge_size=2)
    kwargs_tensor = {
        "pixel_values_videos": torch.randn((100, 16)),
        "video_grid_thw": GridTHW([(1, 10, 10)]),
        "second_per_grid_ts": torch.tensor([1.5]),
        "timestamps": torch.tensor([[0.0, 1.0]]),
    }
    out_kwargs, padded, orig_batch, orig_pixels, merge_factor = (
        wrapper._maybe_pad_multimodal_kwargs(kwargs_tensor))

    assert padded is True
    assert orig_batch == 1
    assert orig_pixels == 100
    assert merge_factor == 4
    assert out_kwargs["pixel_values_videos"].shape == (104, 16)
    assert len(out_kwargs["video_grid_thw"]) == 2
    assert out_kwargs["second_per_grid_ts"].tolist() == [1.5, 1.5]
    assert out_kwargs["timestamps"].tolist() == [[0.0, 1.0], [0.0, 1.0]]

    # Verify list-based metadata sync and empty fallback
    kwargs_list = {
        "pixel_values_videos": torch.randn((100, 16)),
        "video_grid_thw": GridTHW([(1, 10, 10)]),
        "second_per_grid_ts": [],
        "timestamps": [],
    }
    out_list, padded_list, _, _, _ = wrapper._maybe_pad_multimodal_kwargs(
        kwargs_list)
    assert padded_list is True
    assert out_list["second_per_grid_ts"] == [0.0]
    assert out_list["timestamps"] == [[0.0, 0.0]]


def test_maybe_unpad_multimodal_output_all_formats():
    # 1. 2D tensor output: (total_tokens, hidden_dim) -> sliced to original_pixels_len // merge_factor
    out_2d = torch.randn((26, 64))  # 104 padded pixels // 4 = 26 tokens
    unpadded_2d = VllmModelWrapper._maybe_unpad_multimodal_output(
        out_2d,
        padded_anything=True,
        original_batch_len=1,
        original_pixels_len=100,
        merge_factor=4,
    )
    assert unpadded_2d.shape == (25, 64)

    # 2. 3D tensor output: (batch, tokens, hidden_dim) -> sliced to original_batch_len
    out_3d = torch.randn((3, 16, 64))
    unpadded_3d = VllmModelWrapper._maybe_unpad_multimodal_output(
        out_3d,
        padded_anything=True,
        original_batch_len=2,
        original_pixels_len=100,
        merge_factor=4,
    )
    assert unpadded_3d.shape == (2, 16, 64)

    # 3. Tuple and list outputs (vLLM standard per-item embedding tuple)
    out_tuple = (torch.randn(10, 64), torch.randn(12, 64), torch.randn(1, 64))
    unpadded_tuple = VllmModelWrapper._maybe_unpad_multimodal_output(
        out_tuple,
        padded_anything=True,
        original_batch_len=2,
        original_pixels_len=88,
        merge_factor=4,
    )
    assert isinstance(unpadded_tuple, tuple)
    assert len(unpadded_tuple) == 2

    # 4. No-op when padded_anything is False
    assert (VllmModelWrapper._maybe_unpad_multimodal_output(
        out_tuple,
        padded_anything=False,
        original_batch_len=2,
        original_pixels_len=88,
        merge_factor=4,
    ) is out_tuple)


def test_get_model_tp_size_helper():
    wrapper = _make_stub_wrapper(tp_size=4)
    assert wrapper._get_model_tp_size() == 4

    wrapper.mesh = SimpleNamespace(shape={"data": 2})
    wrapper.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
        tensor_parallel_size=2))
    assert wrapper._get_model_tp_size() == 2

    wrapper.mesh = None
    wrapper.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
        tensor_parallel_size=8))
    assert wrapper._get_model_tp_size() == 8


def test_get_activation_sharding_divisor():
    wrapper = _make_stub_wrapper(mesh_shape={
        "data": 1,
        "attn_dp": 8,
        "model": 1
    })
    assert wrapper._get_activation_sharding_divisor() == 8

    wrapper.mesh = SimpleNamespace(shape={
        "data": 1,
        "pcp": 2,
        "attn_dp": 4,
        "model": 1
    })
    assert wrapper._get_activation_sharding_divisor() == 4

    wrapper.mesh = SimpleNamespace(shape={"data": 4, "model": 2})
    assert wrapper._get_activation_sharding_divisor() == 4

    wrapper.mesh = None
    wrapper.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
        tensor_parallel_size=8))
    assert wrapper._get_activation_sharding_divisor() == 8


def test_jittable_architectures_isinstance():

    class NonJittableModel:
        pass

    assert not is_jittable_architecture(NonJittableModel())

    for base_cls in JITTABLE_ARCHS:
        dummy_instance = object.__new__(base_cls)
        assert is_jittable_architecture(dummy_instance)

    # Verify subclassing (e.g. Qwen3.5 / Qwen3-VL-MoE inheriting from Qwen3VLForConditionalGeneration)
    class DummyQwen35Subclass(Qwen3VLForConditionalGeneration):

        def __init__(self):
            pass

    assert is_jittable_architecture(DummyQwen35Subclass())


def test_grid_thw_extended_methods_and_errors():
    grid = GridTHW([(2, 14, 14), (1, 28, 28)])
    assert grid.prod(dim=-1).tolist() == [392, 784]
    assert grid.prod(dim=1).tolist() == [392, 784]

    with pytest.raises(NotImplementedError):
        grid.prod(dim=0)
