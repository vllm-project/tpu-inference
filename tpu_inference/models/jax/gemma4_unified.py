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
"""Text-only JAX serving for ``Gemma4UnifiedForConditionalGeneration``.

Gemma 4 12B ships as ``gemma4_unified``: its language model is the same
decoder as the other dense Gemma 4 sizes (``model.language_model.*``, the same
tensor names and layer structure as 31B), paired with a lightweight vision
embedder (``model.vision_embedder.*``) and an audio projection
(``model.embed_audio.*``) in place of the ``vision_tower`` encoder that
``Gemma4ForConditionalGeneration`` implements. Without a JAX registration the
architecture falls back to the vLLM PyTorch path.

This class serves the decoder through ``Gemma4ForCausalLM``, whose loader
already skips every ``vision``/``audio`` tensor. It has no multimodal encoder,
so every modality the checkpoint declares (a ``<modality>_token_id`` in its
config: image, audio and video for 12B) must be listed with a limit of 0, e.g.
``--limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}'``. Listing is
required because vLLM treats an unlisted modality as unlimited, so a partial
list would still admit inputs this model cannot encode.
"""

from typing import Any

import jax
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference.models.jax.gemma4 import Gemma4ForCausalLM

_MODALITIES = ("image", "audio", "video")


def declared_modalities(hf_config: Any) -> list[str]:
    """Modalities the checkpoint has placeholder tokens for."""
    return [
        m for m in _MODALITIES
        if getattr(hf_config, f"{m}_token_id", None) is not None
    ]


class Gemma4UnifiedTextForCausalLM(Gemma4ForCausalLM):
    """Language model of a ``gemma4_unified`` checkpoint, text only."""

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        limits = getattr(model_config.multimodal_config, "limit_per_prompt",
                         None) or {}
        open_modalities = [
            m for m in declared_modalities(model_config.hf_config)
            if getattr(limits.get(m), "count", limits.get(m)) != 0
        ]
        if open_modalities:
            raise NotImplementedError(
                "Gemma4UnifiedForConditionalGeneration has no JAX multimodal "
                f"encoder, and {open_modalities} would still be accepted; "
                "serve it text-only by listing every modality with a limit of "
                "0, e.g. --limit-mm-per-prompt "
                "'{\"image\": 0, \"audio\": 0, \"video\": 0}'.")
        super().__init__(vllm_config, rng_key, mesh)
