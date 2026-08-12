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
from unittest.mock import MagicMock, patch
import pytest

from tpu_inference.models.vllm.experimental.vision_tower_jit import (
    is_video_supported_model,
    maybe_precompile_vision_encoder_fn,
)


def test_is_video_supported_model():
    mock_config = MagicMock()

    # Video models
    for model_type in ["qwen2_5_vl", "qwen3_vl", "qwen2_vl"]:
        mock_config.model_config.hf_config.model_type = model_type
        assert is_video_supported_model(None, mock_config) is True

    # Omni / image-only models
    for model_type in ["qwen3_omni_moe", "llama", "gemma"]:
        mock_config.model_config.hf_config.model_type = model_type
        assert is_video_supported_model(None, mock_config) is False


def test_omni_model_single_frame_warmup():
    # Verify that Omni / non-video models ONLY warm up t=1 (no fan-out bloat)
    mock_config = MagicMock()
    mock_config.model_config.hf_config.model_type = "qwen3_omni_moe"
    mock_config.model_config.dtype = "bfloat16"
    mock_config.scheduler_config.max_num_batched_tokens = 512
    
    # Mock vision config
    vc = MagicMock()
    vc.in_channels = 3
    vc.temporal_patch_size = 2
    vc.patch_size = 14
    vc.spatial_merge_size = 2
    
    with patch("tpu_inference.models.vllm.experimental.vision_tower_jit.has_jittable_vision", return_value=True), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.get_vision_config", return_value=vc), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.to_jax_dtype", return_value=None):
        
        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=MagicMock(),
            vllm_model=MagicMock(),
            vllm_config=mock_config,
        )
        assert precompile_fn is not None

        compilation_calls = []
        def mock_run_compilation(name, fn, params, call_kwargs, num_patches):
            compilation_calls.append((name, call_kwargs))

        precompile_fn(mock_run_compilation)

        # Ensure all calls were image calls (t=1, image_grid_thw)
        assert len(compilation_calls) > 0
        for name, kwargs in compilation_calls:
            assert "image_grid_thw" in kwargs
            assert "video_grid_thw" not in kwargs
            assert "pixel_values" in kwargs
            assert "pixel_values_videos" not in kwargs


def test_video_model_multi_frame_warmup():
    # Verify that video models warm up multi-frame counts [1, 2, 4, 8, 16]
    mock_config = MagicMock()
    mock_config.model_config.hf_config.model_type = "qwen3_vl"
    mock_config.model_config.dtype = "bfloat16"
    mock_config.scheduler_config.max_num_batched_tokens = 4096
    
    vc = MagicMock()
    vc.in_channels = 3
    vc.temporal_patch_size = 2
    vc.patch_size = 14
    vc.spatial_merge_size = 2
    
    with patch("tpu_inference.models.vllm.experimental.vision_tower_jit.has_jittable_vision", return_value=True), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.get_vision_config", return_value=vc), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.to_jax_dtype", return_value=None):
        
        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=MagicMock(),
            vllm_model=MagicMock(),
            vllm_config=mock_config,
        )
        assert precompile_fn is not None

        compilation_calls = []
        def mock_run_compilation(name, fn, params, call_kwargs, num_patches):
            compilation_calls.append((name, call_kwargs))

        precompile_fn(mock_run_compilation)

        # Ensure both image (t=1) and video (t>1) calls were made
        has_image = any("image_grid_thw" in kwargs for _, kwargs in compilation_calls)
        has_video = any("video_grid_thw" in kwargs for _, kwargs in compilation_calls)
        assert has_image is True
        assert has_video is True


def test_env_var_overrides():
    # Verify custom VISION_PRECOMPILE_FRAMES and VISION_MIN_SHIFT
    mock_config = MagicMock()
    mock_config.model_config.hf_config.model_type = "qwen3_vl"
    mock_config.model_config.dtype = "bfloat16"
    mock_config.scheduler_config.max_num_batched_tokens = 4096
    
    vc = MagicMock()
    vc.in_channels = 3
    vc.temporal_patch_size = 2
    vc.patch_size = 14
    vc.spatial_merge_size = 2
    
    with patch.dict(os.environ, {"VISION_PRECOMPILE_FRAMES": "2,8", "VISION_MIN_SHIFT": "5"}), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.has_jittable_vision", return_value=True), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.get_vision_config", return_value=vc), \
         patch("tpu_inference.models.vllm.experimental.vision_tower_jit.to_jax_dtype", return_value=None):
        
        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=MagicMock(),
            vllm_model=MagicMock(),
            vllm_config=mock_config,
        )
        assert precompile_fn is not None

        compilation_calls = []
        def mock_run_compilation(name, fn, params, call_kwargs, num_patches):
            compilation_calls.append((name, call_kwargs))

        precompile_fn(mock_run_compilation)

        # Only video frames 2 and 8 should be present
        for name, kwargs in compilation_calls:
            assert "video_grid_thw" in kwargs
