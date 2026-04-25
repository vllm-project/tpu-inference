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
from vllm.config import VllmConfig

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Architectures whose embed_multimodal function is safe to wrap with jax.jit.
JITTABLE_ARCHS = {
    "Qwen3_5MoeForConditionalGeneration",
}


def maybe_jit_embed_multimodal_func(embed_multimodal_func_jax: Callable,
                                    vllm_config: VllmConfig):
    """Conditionally wrap `embed_multimodal_func_jax` with jax.jit based on the VllmConfig.

    Args:
        embed_multimodal_func_jax: The JAX function to be potentially JIT-compiled.
        vllm_config: The VllmConfig instance containing the configuration.
    """
    archs = set(vllm_config.model_config.hf_config.architectures)

    if archs & JITTABLE_ARCHS:
        logger.info_once(
            f"JIT-compiling embed_multimodal_func_jax for architectures: {archs & JITTABLE_ARCHS}"
        )
        return jax.jit(static_argnames=("image_grid_thw", "video_grid_thw",
                                        "grid_thw"))(embed_multimodal_func_jax)
    else:
        return embed_multimodal_func_jax


def maybe_precompile_vision_encoder_fn(
    params: Any,
    embed_multimodal_fn: Optional[Callable],
    vllm_config: VllmConfig,
) -> Optional[Callable]:
    """Return a precompile function for jittable vision encoders, or None.

    The returned function accepts a single argument (run_compilation_fn) and
    calls embed_multimodal_fn with dummy pixel_value tensors of various sizes
    so that JAX/XLA compilation is done upfront rather than at first inference.
    Only architectures listed in JITTABLE_ARCHS are supported.
    """
    if embed_multimodal_fn is None:
        return None

    archs = set(vllm_config.model_config.hf_config.architectures)
    if not (archs & JITTABLE_ARCHS):
        return None

    vc = vllm_config.model_config.hf_config.vision_config
    patch_input_dim = (vc.in_channels * vc.temporal_patch_size *
                       vc.patch_size * vc.patch_size)
    spatial_merge_unit = vc.spatial_merge_size**2
    max_patches = (vllm_config.scheduler_config.max_num_batched_tokens *
                   spatial_merge_unit)
    min_shift = 4  # 1 << 4 = 16 patches minimum
    max_shift = max(min_shift, (max(max_patches, 1) - 1).bit_length())
    num_patches_paddings = [1 << i for i in range(min_shift, max_shift + 1)]

    from torchax.ops.mappings import TORCH_DTYPE_TO_JAX
    jax_dtype = TORCH_DTYPE_TO_JAX[vllm_config.model_config.dtype]

    def precompile_fn(run_compilation_fn: Callable) -> None:
        # Deferred import avoids circular dependencies at module load time.
        from tpu_inference.runner.multimodal_manager import GridTHW

        for num_patches in num_patches_paddings:
            # Split num_patches into (h, w) by distributing bits evenly.
            # For any power-of-2 num_patches = 2^k: h=2^(k//2), w=2^(k-k//2).
            k = int(round(math.log2(num_patches)))
            h = 1 << (k // 2)
            w = 1 << (k - k // 2)

            dummy_pixel_values = jnp.ones((num_patches, patch_input_dim),
                                          dtype=jax_dtype)
            dummy_image_grid_thw = GridTHW([(1, h, w)])

            try:
                run_compilation_fn(
                    f"vllm embed_multimodal {dummy_image_grid_thw}",
                    embed_multimodal_fn,
                    params,
                    call_kwargs={
                        "pixel_values": dummy_pixel_values,
                        "image_grid_thw": dummy_image_grid_thw,
                    },
                    num_patches=num_patches,
                )
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "OOM" in str(e).upper():
                    logger.warning(
                        f"embed_multimodal precompile OOM at num_patches={num_patches}; "
                        f"skipping larger sizes. Error: {e}")
                    break
                raise

    return precompile_fn
