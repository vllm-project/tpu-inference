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

# Utilities to support JIT compilation of VisionTower.

import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import \
    Qwen3OmniMoeConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.model_executor.models.qwen3_omni_moe_thinker import \
    Qwen3OmniMoeThinkerForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import is_qwen3_vl
from tpu_inference.utils import to_jax_dtype

logger = init_logger(__name__)

# Architectures whose embed_multimodal function is safe to wrap with jax.jit.
JITTABLE_ARCHS = {
    Qwen3OmniMoeThinkerForConditionalGeneration,
}


def is_jittable_architecture(vllm_model) -> bool:
    """Check if the given vLLM model is of an architecture that supports JIT compilation."""
    is_jittable = any(isinstance(vllm_model, arch) for arch in JITTABLE_ARCHS)
    if is_jittable:
        logger.info_once(
            f"{type(vllm_model)}'s vision tower supports JIT compilation.")
    else:
        logger.warning_once(
            f"{type(vllm_model)}'s vision tower does NOT support JIT compilation."
        )
    return is_jittable


def has_jittable_vision(vllm_model) -> bool:
    """Check if the model has any JIT-compiled vision component (either whole or submodule)."""
    return is_jittable_architecture(vllm_model) or is_qwen3_vl(vllm_model)


def get_vision_config(hf_config: Any) -> Any:
    """Extract vision configuration from hf_config, supporting nested/thinker wrappers."""

    if isinstance(hf_config, Qwen3OmniMoeConfig):
        return hf_config.thinker_config.vision_config
    return hf_config.vision_config


def maybe_jit_embed_multimodal_func(embed_multimodal_func_jax: Callable,
                                    vllm_model) -> Callable:
    """Conditionally wrap `embed_multimodal_func_jax` with jax.jit based on the VllmConfig.

    Args:
        embed_multimodal_func_jax: The JAX function to be potentially JIT-compiled.
        vllm_model: The Vllm model instance containing the configuration.
    """
    if is_jittable_architecture(vllm_model):
        return jax.jit(static_argnames=(
            "image_grid_thw", "video_grid_thw", "grid_thw",
            "audio_feature_lengths"))(embed_multimodal_func_jax)
    else:
        return embed_multimodal_func_jax


class GridTHW(tuple):
    """Tensor-like wrapper for image/video grid_thw arguments.

    - tuple subclass so isinstance(x, tuple) is True — passes vLLM's
    tensor_schema type check (e.g. https://github.com/vllm-project/vllm/blob/9744b699bafed423909ed10da96b80eb0542424b/vllm/model_executor/models/qwen3_vl.py#L2026).
    - Implements a minimal tensor-like API (ndim, shape, tolist, prod) expected by vLLM's
    _process_image_input (https://github.com/vllm-project/vllm/blob/9744b699bafed423909ed10da96b80eb0542424b/vllm/model_executor/models/qwen3_vl.py#L2072)

    We cannot use torch.Tensor[tuple] because jax.jit would complain.
    """

    def __new__(cls, values):

        def _nested_to_tuple(v):
            if isinstance(v, (list, tuple)):
                return tuple(_nested_to_tuple(x) for x in v)
            return int(v)

        flat: tuple = _nested_to_tuple(values)
        return super().__new__(cls, flat)

    # ---- tensor-like API expected by _process_image_input ----

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return (len(self), 3)

    def tolist(self):
        return [list(row) for row in self]

    def prod(self, dim=-1):
        if dim in (-1, 1):
            return np.array([row[0] * row[1] * row[2] for row in self])
        raise NotImplementedError(f"GridTHW.prod({dim}) not supported")

    def __repr__(self):
        return f"GridTHW({tuple(self)})"


def is_video_supported_model(vllm_model) -> bool:
    """Check if the model architecture supports video multimodal inputs.

    Qwen3VLMoe, Qwen3_5 and Qwen3_5Moe all subclass
    Qwen3VLForConditionalGeneration, so one entry covers the whole family.
    """
    return isinstance(vllm_model, (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    ))


# Grid signatures that warmup actually precompiled. Populated by
# `maybe_precompile_vision_encoder_fn` and read by `maybe_prepare_for_jit` so
# that a warmup miss -- which costs a blocking XLA recompile -- is visible in
# the logs instead of being silent. Empty means warmup never ran.
_WARMED_GRIDS: set[tuple] = set()


def _record_warmed_grid(grid: "GridTHW") -> None:
    _WARMED_GRIDS.add(tuple(grid))


def _warn_if_grid_not_warmed(key: str, grid: "GridTHW") -> None:
    """Log once per grid signature that warmup did not cover.

    `grid_thw` is a jax.jit static argument, so a signature we did not
    precompile triggers a blocking recompile on its first request.
    """
    if not _WARMED_GRIDS or tuple(grid) in _WARMED_GRIDS:
        return
    logger.warning_once(
        f"Vision warmup miss: {key}={grid} was not precompiled, so XLA will "
        f"recompile the vision encoder for this signature. Precompiled "
        f"signatures: {sorted(_WARMED_GRIDS)}.")


def _derive_video_spatial_grid(max_patches: int, max_grid_t: int,
                               vision_config: Any) -> tuple[int, int]:
    """Return the (height, width) patch grid that video warmup should target.

    There is no general way to predict the spatial grid: `smart_resize` only
    downscales videos that exceed the processor's pixel budget, and that
    budget is effectively unbounded for these models (25M pixels for
    Qwen3.5), so real videos keep their native resolution. Set
    VISION_PRECOMPILE_VIDEO_GRID to the grid your traffic actually produces.

    Absent that, fall back to the largest near-square grid that still fits the
    patch budget at the largest warmed frame count, so warmup at least emits a
    usable signature rather than one that is skipped as over-budget.
    """
    override = envs.VISION_PRECOMPILE_VIDEO_GRID
    if override:
        if len(override) != 2:
            raise ValueError(
                f"VISION_PRECOMPILE_VIDEO_GRID must be 'height,width' in "
                f"patches, got {override}.")
        return override[0], override[1]

    merge = vision_config.spatial_merge_size
    spatial_budget = max(merge * merge, max_patches // max(max_grid_t, 1))
    side = math.isqrt(spatial_budget)
    height = max(merge, (side // merge) * merge)
    width = max(merge, (spatial_budget // height // merge) * merge)
    return height, width


def maybe_precompile_vision_encoder_fn(
        params: Any, embed_multimodal_fn: Optional[Callable], vllm_model,
        vllm_config: VllmConfig) -> Optional[Callable]:
    """Return a precompile function for jittable vision encoders, or None.

    The returned function accepts a single argument (run_compilation_fn) and
    calls embed_multimodal_fn with dummy pixel_value tensors of various sizes
    so that JAX/XLA compilation is done upfront rather than at first inference.
    Only architectures listed in JITTABLE_ARCHS are supported.

    By default, warmup covers single-item batches (batch size 1). Under
    concurrent traffic, multiple videos may be batched into a single
    `video_grid_thw` tuple; set `VISION_PRECOMPILE_VIDEO_BATCH_SIZES` to
    precompile multi-item video batch sizes up front. Any un-warmed signatures
    are reported at runtime by `_warn_if_grid_not_warmed`.
    """
    if embed_multimodal_fn is None:
        return None

    if not has_jittable_vision(vllm_model):
        return None

    # patch_input_dim is the flattened input feature dimension per raw patch:
    #   in_channels * temporal_patch_size * patch_size * patch_size
    # e.g. for Qwen3.5: 3 * 2 * 16 * 16 = 1536
    # Ref: https://github.com/vllm-project/vllm/blob/eb6661d52/vllm/model_executor/models/qwen3_vl.py#L1941
    vc = get_vision_config(vllm_config.model_config.hf_config)
    patch_input_dim = (vc.in_channels * vc.temporal_patch_size *
                       vc.patch_size * vc.patch_size)
    # Each output token is produced by `spatial_merge_size ** 2` input patches,
    # so the patch budget is the token budget scaled *up* by the merge unit.
    spatial_merge_unit = vc.spatial_merge_size**2
    max_patches = (vllm_config.scheduler_config.max_num_batched_tokens *
                   spatial_merge_unit)

    min_shift = envs.VISION_MIN_SHIFT
    max_shift = max(min_shift, (max(max_patches, 1) - 1).bit_length())
    num_patches_paddings = [1 << i for i in range(min_shift, max_shift + 1)]

    # Decoded frame counts. For video-capable models, warm a representative
    # spread; for image-only and Omni models, default strictly to [1] to avoid
    # startup compilation bloat.
    video_supported = is_video_supported_model(vllm_model)
    frame_counts = envs.VISION_PRECOMPILE_FRAMES
    if not frame_counts:
        frame_counts = [2, 4, 8, 16, 32] if video_supported else [1]

    # The processor emits `grid_t = num_frames // temporal_patch_size`, so
    # convert the frame counts into the temporal grid values that actually key
    # the compilation cache.
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py#L317
    grid_ts = sorted(
        {max(1, f // vc.temporal_patch_size)
         for f in frame_counts if f >= 1})
    video_batch_sizes = sorted({
        b
        for b in (envs.VISION_PRECOMPILE_VIDEO_BATCH_SIZES or [1]) if b >= 1
    })

    jax_dtype = to_jax_dtype(vllm_config.model_config.dtype)

    def precompile_fn(run_compilation_fn: Callable) -> None:
        # 1. Precompile standard single-frame image shapes across patch budgets
        for num_patches in num_patches_paddings:
            k = int(round(math.log2(num_patches)))
            h = 1 << (k // 2)
            w = 1 << (k - k // 2)

            dummy_pixel_values = jnp.ones((num_patches, patch_input_dim),
                                          dtype=jax_dtype)
            dummy_image_grid_thw = GridTHW([(1, h, w)])
            run_compilation_fn(
                f"vllm embed_multimodal image {dummy_image_grid_thw}",
                embed_multimodal_fn,
                params,
                call_kwargs={
                    "pixel_values": dummy_pixel_values,
                    "image_grid_thw": dummy_image_grid_thw,
                },
                num_patches=num_patches,
            )
            _record_warmed_grid(dummy_image_grid_thw)

        # 2. Precompile multi-frame video shapes for video-supported models
        video_grid_ts = [t for t in grid_ts if t > 1]
        if not video_grid_ts:
            return

        rep_h, rep_w = _derive_video_spatial_grid(max_patches,
                                                  max(video_grid_ts), vc)
        rep_spatial_patches = rep_h * rep_w
        logger.info(f"Precompiling video vision encoder for spatial grid "
                    f"({rep_h}, {rep_w}) at temporal grids {video_grid_ts} "
                    f"and batch sizes {video_batch_sizes}.")

        for t in video_grid_ts:
            for b in video_batch_sizes:
                total_patches = b * t * rep_spatial_patches
                if total_patches > max_patches:
                    logger.warning(
                        f"Skipping video warmup for grid ({t}, {rep_h}, "
                        f"{rep_w}) x{b}: {total_patches} patches exceeds the "
                        f"budget of {max_patches}.")
                    continue

                dummy_pixel_values = jnp.ones((total_patches, patch_input_dim),
                                              dtype=jax_dtype)
                dummy_video_grid_thw = GridTHW([(t, rep_h, rep_w)] * b)
                run_compilation_fn(
                    f"vllm embed_multimodal video {dummy_video_grid_thw}",
                    embed_multimodal_fn,
                    params,
                    call_kwargs={
                        "pixel_values_videos": dummy_pixel_values,
                        "video_grid_thw": dummy_video_grid_thw,
                        "second_per_grid_ts": jnp.zeros((b, ),
                                                        dtype=jnp.float32),
                        "timestamps": jnp.zeros((b, 2), dtype=jnp.float32),
                    },
                    num_patches=total_patches,
                )
                _record_warmed_grid(dummy_video_grid_thw)

    return precompile_fn


def maybe_prepare_for_jit(kwargs: dict, vllm_model) -> dict:
    """Convert certain kwargs to JIT-friendly formats, if needed.

    Specifically, convert "image_grid_thw", "video_grid_thw", and "grid_thw" to
    GridTHW instances, which are tuple subclasses that can be hashed in jax.jit.
    """
    if not has_jittable_vision(vllm_model):
        return kwargs

    for k, v in kwargs.items():
        if k in ("image_grid_thw", "video_grid_thw", "grid_thw"):
            kwargs[k] = GridTHW(v.tolist())
            _warn_if_grid_not_warmed(k, kwargs[k])

        elif k == "audio_feature_lengths" and isinstance(v, torch.Tensor):
            kwargs[k] = tuple(v.tolist())

    return kwargs
