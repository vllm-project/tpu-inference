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

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tpu_inference.models.vllm.experimental import vision_tower_jit
from tpu_inference.models.vllm.experimental.vision_tower_jit import (
    GridTHW, is_video_supported_model, maybe_precompile_vision_encoder_fn,
    maybe_prepare_for_jit)


def test_is_video_supported_model():
    # Video models
    for model_type in [
            "qwen2_5_vl", "qwen3_vl", "qwen2_vl", "qwen3_5", "qwen3_5_vl"
    ]:
        config = SimpleNamespace(model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)))
        assert is_video_supported_model(None, config) is True

    # Omni / image-only models
    for model_type in ["qwen3_omni_moe", "llama", "gemma"]:
        config = SimpleNamespace(model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)))
        assert is_video_supported_model(None, config) is False


def test_omni_model_single_frame_warmup():
    # Verify that Omni / non-video models ONLY warm up t=1 (no fan-out bloat)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_omni_moe"),
            dtype="bfloat16",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
    )

    vc = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )

    with patch.object(vision_tower_jit, "has_jittable_vision", return_value=True), \
         patch.object(vision_tower_jit, "get_vision_config", return_value=vc), \
         patch.object(vision_tower_jit, "to_jax_dtype", return_value=None):

        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=SimpleNamespace(),
            vllm_config=config,
        )
        assert precompile_fn is not None

        compilation_calls = []

        def record_compilation(name, fn, params, call_kwargs, num_patches):
            compilation_calls.append((name, call_kwargs))

        precompile_fn(record_compilation)

        # Ensure all calls were image calls (t=1, image_grid_thw)
        assert len(compilation_calls) > 0
        for name, kwargs in compilation_calls:
            assert "image_grid_thw" in kwargs
            assert "video_grid_thw" not in kwargs
            assert "pixel_values" in kwargs
            assert "pixel_values_videos" not in kwargs


def test_video_model_multi_frame_warmup():
    # Verify that video models warm up multi-frame counts [1, 2, 4, 8, 16]
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl"),
            dtype="bfloat16",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )

    vc = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )

    with patch.object(vision_tower_jit, "has_jittable_vision", return_value=True), \
         patch.object(vision_tower_jit, "get_vision_config", return_value=vc), \
         patch.object(vision_tower_jit, "to_jax_dtype", return_value=None):

        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=SimpleNamespace(),
            vllm_config=config,
        )
        assert precompile_fn is not None

        compilation_calls = []

        def record_compilation(name, fn, params, call_kwargs, num_patches):
            compilation_calls.append((name, call_kwargs))

        precompile_fn(record_compilation)

        # Ensure both image (t=1) and video (t>1) calls were made
        has_image = any("image_grid_thw" in kwargs
                        for _, kwargs in compilation_calls)
        has_video = any("video_grid_thw" in kwargs
                        for _, kwargs in compilation_calls)
        assert has_image is True
        assert has_video is True


def test_env_var_overrides():
    # Verify custom VISION_PRECOMPILE_FRAMES and VISION_MIN_SHIFT
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl"),
            dtype="bfloat16",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )

    vc = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )

    with patch.dict(os.environ, {"VISION_PRECOMPILE_FRAMES": "2,8", "VISION_MIN_SHIFT": "5"}), \
         patch.object(vision_tower_jit, "has_jittable_vision", return_value=True), \
         patch.object(vision_tower_jit, "get_vision_config", return_value=vc), \
         patch.object(vision_tower_jit, "to_jax_dtype", return_value=None):

        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=SimpleNamespace(),
            vllm_config=config,
        )
        assert precompile_fn is not None

        compilation_calls = []

        def record_compilation(name, fn, params, call_kwargs, num_patches):
            compilation_calls.append((name, call_kwargs))

        precompile_fn(record_compilation)

        # Verify image calls respect VISION_MIN_SHIFT=5 (first bucket 1<<5 = 32 patches)
        image_calls = [
            c for c in compilation_calls if "image_grid_thw" in c[1]
        ]
        assert len(image_calls) > 0
        assert image_calls[0][1]["pixel_values"].shape[0] >= 32

        # Verify video calls respect VISION_PRECOMPILE_FRAMES="2,8"
        video_calls = [
            c for c in compilation_calls if "video_grid_thw" in c[1]
        ]
        assert len(video_calls) == 2
        video_frames = [c[1]["video_grid_thw"][0][0] for c in video_calls]
        assert video_frames == [2, 8]


def test_maybe_prepare_for_jit_conversions():
    # Verify conversion of grid_thw tensors to GridTHW and audio tensors to tuples
    mock_model = SimpleNamespace()
    with patch.object(vision_tower_jit,
                      "has_jittable_vision",
                      return_value=True):
        kwargs = {
            "image_grid_thw": torch.tensor([[1, 28, 28]]),
            "video_grid_thw": torch.tensor([[2, 14, 14]]),
            "grid_thw": torch.tensor([[1, 14, 14]]),
            "audio_feature_lengths": torch.tensor([100, 200]),
            "other_param": "untouched",
        }
        res = maybe_prepare_for_jit(kwargs, mock_model)
        assert isinstance(res["image_grid_thw"], GridTHW)
        assert res["image_grid_thw"].tolist() == [[1, 28, 28]]
        assert isinstance(res["video_grid_thw"], GridTHW)
        assert res["video_grid_thw"].tolist() == [[2, 14, 14]]
        assert isinstance(res["grid_thw"], GridTHW)
        assert res["grid_thw"].tolist() == [[1, 14, 14]]
        assert res["audio_feature_lengths"] == (100, 200)
        assert res["other_param"] == "untouched"

    # Non-jittable vision model should return kwargs untouched
    with patch.object(vision_tower_jit,
                      "has_jittable_vision",
                      return_value=False):
        raw_kwargs = {"image_grid_thw": torch.tensor([[1, 28, 28]])}
        res_non_jit = maybe_prepare_for_jit(raw_kwargs, mock_model)
        assert isinstance(res_non_jit["image_grid_thw"], torch.Tensor)


def test_is_video_supported_model_edge_cases():
    # None configs
    assert is_video_supported_model(None, None) is False
    assert is_video_supported_model(
        None, SimpleNamespace(model_config=None)) is False
    assert is_video_supported_model(
        None,
        SimpleNamespace(model_config=SimpleNamespace(hf_config=None))) is False

    # Legacy qwen_vl returns False
    cfg_legacy = SimpleNamespace(model_config=SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen_vl")))
    assert is_video_supported_model(None, cfg_legacy) is False

    # Mixed-case model type
    cfg_upper = SimpleNamespace(model_config=SimpleNamespace(
        hf_config=SimpleNamespace(model_type="QWEN2_5_VL")))
    assert is_video_supported_model(None, cfg_upper) is True


def test_env_var_malformed_and_negative_fallbacks():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl"),
            dtype="bfloat16",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )
    vc = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )

    with patch.dict(os.environ, {
            "VISION_PRECOMPILE_FRAMES": "invalid,abc,,-1,0",
            "VISION_MIN_SHIFT": "not_an_int"
    }), \
         patch.object(vision_tower_jit, "has_jittable_vision", return_value=True), \
         patch.object(vision_tower_jit, "get_vision_config", return_value=vc), \
         patch.object(vision_tower_jit, "to_jax_dtype", return_value=None):

        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=SimpleNamespace(),
            vllm_config=config,
        )
        assert precompile_fn is not None

        compilation_calls = []
        precompile_fn(lambda name, fn, params, call_kwargs, num_patches:
                      compilation_calls.append(call_kwargs))
        # Malformed frame string with no valid positive frame counts falls back to [1]
        assert len(compilation_calls) > 0


def test_max_patches_budget_skips_large_video_frames():
    # Low max_num_batched_tokens = 512 -> max_patches = 128 (512 // 4)
    # Video grid 16x16 = 256 patches per frame -> even t=2 (512 patches) exceeds 128
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl"),
            dtype="bfloat16",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
    )
    vc = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )

    with patch.object(vision_tower_jit, "has_jittable_vision", return_value=True), \
         patch.object(vision_tower_jit, "get_vision_config", return_value=vc), \
         patch.object(vision_tower_jit, "to_jax_dtype", return_value=None):

        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=SimpleNamespace(),
            vllm_config=config,
        )
        assert precompile_fn is not None

        compilation_calls = []
        precompile_fn(lambda name, fn, params, call_kwargs, num_patches:
                      compilation_calls.append(call_kwargs))

        # Video calls should have been skipped because total_patches (>= 512) exceeded max_patches (128)
        has_video = any("video_grid_thw" in kwargs
                        for kwargs in compilation_calls)
        assert has_video is False


def test_maybe_precompile_returns_none_for_unsupported_model():
    # Case 1: embed_multimodal_fn is None
    res1 = maybe_precompile_vision_encoder_fn(
        params={},
        embed_multimodal_fn=None,
        vllm_model=SimpleNamespace(),
        vllm_config=SimpleNamespace(),
    )
    assert res1 is None

    # Case 2: has_jittable_vision is False
    with patch.object(vision_tower_jit,
                      "has_jittable_vision",
                      return_value=False):
        res2 = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=SimpleNamespace(),
            vllm_config=SimpleNamespace(),
        )
        assert res2 is None
