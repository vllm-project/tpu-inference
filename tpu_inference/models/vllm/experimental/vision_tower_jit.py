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
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)
from vllm.model_executor.models.qwen3_omni_moe_thinker import \
    Qwen3OmniMoeThinkerForConditionalGeneration

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_jax_dtype

logger = init_logger(__name__)

# Architectures whose embed_multimodal function is safe to wrap with jax.jit.
JITTABLE_ARCHS = {
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
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


@jax.tree_util.register_pytree_node_class
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

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if isinstance(key, slice):
            return type(self)(val)
        return val

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

    def tree_flatten(self):
        return (), tuple(self)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


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
    min_shift = envs.VISION_MIN_SHIFT
    if min_shift < 6:
        logger.warning(
            f"VISION_MIN_SHIFT is set to {min_shift} (< 6). "
            "This may cause JAX divisibility errors on TPUs with 8+ devices "
            "if spatial merging is active.")
    max_shift = max(min_shift, (max(max_patches, 1) - 1).bit_length())
    num_patches_paddings = [1 << i for i in range(min_shift, max_shift + 1)]

    jax_dtype = to_jax_dtype(vllm_config.model_config.dtype)

    def precompile_fn(run_compilation_fn: Callable) -> None:
        for num_patches in num_patches_paddings:
            # Split num_patches into (h, w) by distributing bits evenly.
            # For any power-of-2 num_patches = 2^k: h=2^(k//2), w=2^(k-k//2).
            k = int(round(math.log2(num_patches)))
            h = 1 << (k // 2)
            w = 1 << (k - k // 2)

            # By default, we precompile for common small frame counts to balance startup time.
            # Users can override this via the VISION_PRECOMPILE_FRAMES environment variable
            # (e.g., VISION_PRECOMPILE_FRAMES="1,2,4,8,16,64") to support specific video lengths.
            #
            # ⚠️ WARNING: Adding more frames or larger buckets here will significantly increase
            # server startup time (XLA compilation) and can cause Host CPU OOMs during boot.
            frame_counts = [1, 2, 4, 8, 16]
            if envs.VISION_PRECOMPILE_FRAMES:
                frame_counts = envs.VISION_PRECOMPILE_FRAMES
                logger.info(
                    f"Using custom vision precompile frames: {frame_counts}")

            for t_val in frame_counts:
                # Limit batch sizes to prevent astronomical compilation time and host OOMs.
                # If users submit larger batches, they will incur a one-time compilation cost at runtime.
                batch_sizes = [1, 2] if t_val == 1 else [1]
                for b in batch_sizes:
                    dummy_pixel_values = jnp.ones(
                        (b * t_val * num_patches, patch_input_dim),
                        dtype=jax_dtype)
                    dummy_image_grid_thw = GridTHW([(t_val, h, w)] * b)

                    model_type = getattr(vllm_config.model_config.hf_config,
                                         "model_type", "")
                    if model_type in ("qwen2_vl", "qwen2_5_vl", "qwen",
                                      "qwen3_5_moe", "qwen3_5", "qwen3_vl"):
                        grid_keys = ("image_grid_thw", "video_grid_thw")
                    else:
                        grid_keys = ("image_grid_thw", "video_grid_thw",
                                     "grid_thw")

                    for grid_key in grid_keys:
                        pixel_key = "pixel_values_videos" if grid_key == "video_grid_thw" else "pixel_values"
                        run_compilation_fn(
                            f"vllm embed_multimodal {grid_key}={dummy_image_grid_thw}",
                            embed_multimodal_fn,
                            params,
                            call_kwargs={
                                pixel_key: dummy_pixel_values,
                                grid_key: dummy_image_grid_thw,
                            },
                            num_patches=num_patches,
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

        elif k == "timestamps":
            if isinstance(v, list):
                kwargs[k] = torch.tensor(v, dtype=torch.float32)
            elif isinstance(v, (float, int)):
                kwargs[k] = torch.tensor([v], dtype=torch.float32)

    return kwargs
