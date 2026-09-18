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
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.models.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.model_executor.models.qwen3_omni_moe_thinker import \
    Qwen3OmniMoeThinkerForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from tpu_inference.models.vllm.experimental import vision_tower_jit
from tpu_inference.models.vllm.experimental.vision_tower_jit import (
    GridTHW, is_video_supported_model, maybe_precompile_vision_encoder_fn,
    maybe_prepare_for_jit)

# Qwen3-VL-ish vision config: 16px patches, 2-frame temporal patches, 2x2 merge.
VISION_CONFIG = SimpleNamespace(
    in_channels=3,
    temporal_patch_size=2,
    patch_size=16,
    spatial_merge_size=2,
)


@pytest.fixture(autouse=True)
def clear_warmed_grids():
    """`_WARMED_GRIDS` is module-level state shared across tests."""
    vision_tower_jit._WARMED_GRIDS.clear()
    yield
    vision_tower_jit._WARMED_GRIDS.clear()


def make_config(max_num_batched_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype="bfloat16", hf_config=None),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens),
    )


def run_warmup(vllm_model,
               max_num_batched_tokens=8192,
               video_grid=(16, 16),
               env=None):
    """Run the precompile fn and return the recorded call_kwargs."""
    with patch.dict(os.environ, env or {}), \
         patch.object(vision_tower_jit, "has_jittable_vision", return_value=True), \
         patch.object(vision_tower_jit, "get_vision_config", return_value=VISION_CONFIG), \
         patch.object(vision_tower_jit, "_derive_video_spatial_grid", return_value=video_grid), \
         patch.object(vision_tower_jit, "to_jax_dtype", return_value=None):

        precompile_fn = maybe_precompile_vision_encoder_fn(
            params={},
            embed_multimodal_fn=lambda *args, **kwargs: None,
            vllm_model=vllm_model,
            vllm_config=make_config(max_num_batched_tokens),
        )
        assert precompile_fn is not None

        calls = []
        precompile_fn(lambda name, fn, params, call_kwargs, num_patches: calls.
                      append(call_kwargs))
        return calls


def test_is_video_supported_model():
    for cls in (Qwen2VLForConditionalGeneration,
                Qwen2_5_VLForConditionalGeneration,
                Qwen3VLForConditionalGeneration):
        assert is_video_supported_model(MagicMock(spec=cls)) is True

    # Omni is jittable but is not a video model.
    assert is_video_supported_model(
        MagicMock(spec=Qwen3OmniMoeThinkerForConditionalGeneration)) is False
    assert is_video_supported_model(SimpleNamespace()) is False


def test_omni_model_single_frame_warmup():
    # Non-video models must only warm t=1 image shapes (no fan-out bloat).
    calls = run_warmup(
        MagicMock(spec=Qwen3OmniMoeThinkerForConditionalGeneration))

    assert len(calls) > 0
    for kwargs in calls:
        assert "image_grid_thw" in kwargs
        assert "pixel_values" in kwargs
        assert "video_grid_thw" not in kwargs
        assert "pixel_values_videos" not in kwargs


def test_video_model_warms_both_image_and_video():
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration))

    assert any("image_grid_thw" in kwargs for kwargs in calls)
    assert any("video_grid_thw" in kwargs for kwargs in calls)


def test_video_warmup_uses_derived_spatial_grid():
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
                       video_grid=(34, 46))

    video_grids = [
        kwargs["video_grid_thw"][0] for kwargs in calls
        if "video_grid_thw" in kwargs
    ]
    assert video_grids
    for _, h, w in video_grids:
        assert (h, w) == (34, 46)


def test_frame_counts_are_converted_to_temporal_grid():
    # temporal_patch_size=2, so 4 and 16 decoded frames are grid_t 2 and 8.
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
                       env={"VISION_PRECOMPILE_FRAMES": "4,16"})

    grid_ts = [
        kwargs["video_grid_thw"][0][0] for kwargs in calls
        if "video_grid_thw" in kwargs
    ]
    assert grid_ts == [2, 8]


def test_frame_count_below_temporal_patch_size_is_not_a_video_shape():
    # 2 decoded frames collapse to grid_t=1, which is an image-shaped signature.
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
                       env={"VISION_PRECOMPILE_FRAMES": "2"})

    assert not any("video_grid_thw" in kwargs for kwargs in calls)


def test_min_shift_env_override():
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
                       env={"VISION_MIN_SHIFT": "5"})

    image_calls = [kwargs for kwargs in calls if "image_grid_thw" in kwargs]
    assert image_calls
    # First bucket is 1 << 5 = 32 patches.
    assert image_calls[0]["pixel_values"].shape[0] == 32


def test_malformed_env_vars_raise():
    model = MagicMock(spec=Qwen3VLForConditionalGeneration)

    with pytest.raises(ValueError):
        run_warmup(model, env={"VISION_MIN_SHIFT": "not_an_int"})

    with pytest.raises(ValueError):
        run_warmup(model, env={"VISION_PRECOMPILE_FRAMES": "abc"})


def test_patch_budget_scales_up_by_the_merge_unit():
    # max_patches = max_num_batched_tokens * spatial_merge_size**2 = 512 * 4.
    # A 16x16 grid at grid_t=8 is 2048 patches, which exactly fits.
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
                       max_num_batched_tokens=512,
                       env={"VISION_PRECOMPILE_FRAMES": "16"})

    grid_ts = [
        kwargs["video_grid_thw"][0][0] for kwargs in calls
        if "video_grid_thw" in kwargs
    ]
    assert grid_ts == [8]


def test_video_grid_env_override():
    with patch.dict(os.environ, {"VISION_PRECOMPILE_VIDEO_GRID": "16,16"}):
        assert vision_tower_jit._derive_video_spatial_grid(
            4096, 8, VISION_CONFIG) == (16, 16)


def test_malformed_video_grid_override_raises():
    with patch.dict(os.environ, {"VISION_PRECOMPILE_VIDEO_GRID": "16"}):
        with pytest.raises(ValueError):
            vision_tower_jit._derive_video_spatial_grid(4096, 8, VISION_CONFIG)


def test_derived_grid_fallback_always_fits_the_budget():
    # The fallback must never emit a grid that warmup would then skip.
    for max_patches, max_grid_t in ((4096, 8), (2048, 16), (256, 2), (64, 1)):
        h, w = vision_tower_jit._derive_video_spatial_grid(
            max_patches, max_grid_t, VISION_CONFIG)
        assert h % VISION_CONFIG.spatial_merge_size == 0
        assert w % VISION_CONFIG.spatial_merge_size == 0
        assert h * w * max_grid_t <= max_patches


def test_video_shapes_over_the_patch_budget_are_skipped():
    # 64x64 = 4096 patches per temporal step, well over the 512 * 4 budget.
    calls = run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
                       max_num_batched_tokens=512,
                       video_grid=(64, 64))

    assert not any("video_grid_thw" in kwargs for kwargs in calls)


def test_warmed_grids_are_recorded():
    run_warmup(MagicMock(spec=Qwen3VLForConditionalGeneration),
               env={"VISION_PRECOMPILE_FRAMES": "4"})

    assert ((2, 16, 16), ) in vision_tower_jit._WARMED_GRIDS
    assert ((1, 16, 16), ) in vision_tower_jit._WARMED_GRIDS


def test_warmup_miss_is_logged():
    vision_tower_jit._WARMED_GRIDS.add(((2, 16, 16), ))

    with patch.object(vision_tower_jit.logger, "warning_once") as warn:
        vision_tower_jit._warn_if_grid_not_warmed("video_grid_thw",
                                                  GridTHW([(2, 16, 16)]))
        warn.assert_not_called()

        vision_tower_jit._warn_if_grid_not_warmed("video_grid_thw",
                                                  GridTHW([(3, 34, 46)]))
        warn.assert_called_once()


def test_warmup_miss_is_silent_when_warmup_never_ran():
    with patch.object(vision_tower_jit.logger, "warning_once") as warn:
        vision_tower_jit._warn_if_grid_not_warmed("video_grid_thw",
                                                  GridTHW([(3, 34, 46)]))
        warn.assert_not_called()


def test_maybe_prepare_for_jit_conversions():
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


def test_video_batch_sizes_env_override():
    calls = run_warmup(
        MagicMock(spec=Qwen3VLForConditionalGeneration),
        max_num_batched_tokens=1024,
        env={
            "VISION_PRECOMPILE_FRAMES": "4",
            "VISION_PRECOMPILE_VIDEO_BATCH_SIZES": "1,2,4",
        },
        video_grid=(16, 16),
    )

    video_calls = [kwargs for kwargs in calls if "video_grid_thw" in kwargs]
    assert len(video_calls) == 3
    assert [len(c["video_grid_thw"]) for c in video_calls] == [1, 2, 4]
    assert video_calls[0]["video_grid_thw"] == GridTHW([(2, 16, 16)])
    assert video_calls[1]["video_grid_thw"] == GridTHW([(2, 16, 16)] * 2)
    assert video_calls[2]["video_grid_thw"] == GridTHW([(2, 16, 16)] * 4)
    assert video_calls[2]["second_per_grid_ts"].shape == (4, )
    assert video_calls[2]["timestamps"].shape == (4, 2)
