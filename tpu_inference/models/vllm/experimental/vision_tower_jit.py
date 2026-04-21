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

from typing import Callable

import jax
from vllm.config import VllmConfig

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def maybe_jit_embed_multimodal_func(embed_multimodal_func_jax: Callable,
                                    vllm_config: VllmConfig):
    """Conditionally wrap `embed_multimodal_func_jax` with jax.jit based on the VllmConfig.

    Args:
        embed_multimodal_func_jax: The JAX function to be potentially JIT-compiled.
        vllm_config: The VllmConfig instance containing the configuration.
    """
    archs = set(vllm_config.model_config.hf_config.architectures)

    JITTABLE_ARCHS = {
        "Qwen3_5MoeForConditionalGeneration",
    }

    if archs & JITTABLE_ARCHS:
        logger.info_once(
            f"JIT-compiling embed_multimodal_func_jax for architectures: {archs & JITTABLE_ARCHS}"
        )
        return jax.jit(embed_multimodal_func_jax)
    else:
        return embed_multimodal_func_jax
