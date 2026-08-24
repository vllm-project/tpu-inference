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
import os
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import \
    Qwen3OmniMoeConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_omni_moe_thinker import \
    Qwen3OmniMoeThinkerForConditionalGeneration

from tpu_inference.logger import init_logger
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
    from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import \
        is_qwen3_vl
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


def is_video_supported_model(vllm_model, vllm_config: VllmConfig) -> bool:
    """Check if the model architecture supports video multimodal inputs."""
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type",
                         "").lower() if hf_config else ""
    return any(v_arch in model_type for v_arch in (
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_vl",
        "qwen_vl",
    ))


def maybe_precompile_vision_encoder_fn(
        params: Any, embed_multimodal_fn: Optional[Callable], vllm_model,
        vllm_config: VllmConfig) -> Optional[Callable]:
    """Return a precompile function for jittable vision encoders, or None.

    The returned function accepts a single argument (run_compilation_fn) and
    calls embed_multimodal_fn with dummy pixel_value tensors of various sizes
    so that JAX/XLA compilation is done upfront rather than at first inference.
    Only architectures listed in JITTABLE_ARCHS are supported.
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
    spatial_merge_unit = vc.spatial_merge_size**2
    max_patches = (vllm_config.scheduler_config.max_num_batched_tokens //
                   spatial_merge_unit)

    env_min_shift = os.environ.get("VISION_MIN_SHIFT", None)
    if env_min_shift:
        try:
            min_shift = max(1, int(env_min_shift.strip()))
        except ValueError:
            min_shift = 4
    else:
        min_shift = 4  # 1 << 4 = 16 patches minimum

    max_shift = max(min_shift, (max(max_patches, 1) - 1).bit_length())
    num_patches_paddings = [1 << i for i in range(min_shift, max_shift + 1)]

    # Frame counts:
    # For video-capable models, warm representative video frame counts [2, 4, 8, 16].
    # For image-only and Omni models, default strictly to [1] to avoid startup compilation bloat.
    env_frames = os.environ.get("VISION_PRECOMPILE_FRAMES", None)
    if env_frames:
        try:
            parsed = [
                int(f.strip()) for f in env_frames.split(",") if f.strip()
            ]
            frame_counts = [f for f in parsed if f >= 1]
            if not frame_counts:
                frame_counts = [1]
        except ValueError:
            logger.warning(
                f"Invalid VISION_PRECOMPILE_FRAMES '{env_frames}', using defaults."
            )
            frame_counts = [1, 2, 4, 8, 16] if is_video_supported_model(
                vllm_model, vllm_config) else [1]
    elif is_video_supported_model(vllm_model, vllm_config):
        frame_counts = [1, 2, 4, 8, 16]
    else:
        frame_counts = [1]

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

        # 2. Precompile multi-frame video shapes for video-supported models
        video_frames = [t for t in frame_counts if t > 1]
        if video_frames:
            # Representative spatial grid (e.g. 14x14 or 16x16)
            rep_h, rep_w = 16, 16
            rep_spatial_patches = rep_h * rep_w
            for t in video_frames:
                total_patches = t * rep_spatial_patches
                if total_patches > max_patches:
                    continue

                dummy_pixel_values = jnp.ones((total_patches, patch_input_dim),
                                              dtype=jax_dtype)
                dummy_video_grid_thw = GridTHW([(t, rep_h, rep_w)])
                run_compilation_fn(
                    f"vllm embed_multimodal video {dummy_video_grid_thw}",
                    embed_multimodal_fn,
                    params,
                    call_kwargs={
                        "pixel_values_videos": dummy_pixel_values,
                        "video_grid_thw": dummy_video_grid_thw,
                        "second_per_grid_ts": jnp.zeros((1, ),
                                                        dtype=jnp.float32),
                        "timestamps": jnp.zeros((1, 2), dtype=jnp.float32),
                    },
                    num_patches=total_patches,
                )

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

        elif k == "audio_feature_lengths" and isinstance(v, torch.Tensor):
            kwargs[k] = tuple(v.tolist())

    return kwargs
