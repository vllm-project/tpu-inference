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
"""Skips checkpoint tensors for decoder layers pruned via num_hidden_layers.

With hf_overrides={"num_hidden_layers": N} the model only instantiates layers
0..N-1, but the checkpoint still holds every layer. Both the vLLM and the JAX
weight loaders use these helpers to drop tensors (and whole safetensors
shards) that have no matching parameter.
"""

import json
import os

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import ModelConfig

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Optional prefixes in front of `layers.<idx>.` for decoder-stack tensors
# (`layers.N.`, `model.layers.N.`, `model.language_model.layers.N.`).
_DECODER_PREFIXES = ("model", "language_model")


def decoder_layer_index(name: str) -> int | None:
    """Returns the decoder layer index of a checkpoint tensor, or None.

    Only `[model.][language_model.]layers.<idx>.<rest>` qualifies. Embeddings,
    norms, lm_head, vision towers (`visual.blocks.N.`,
    `vision_tower.vision_model.encoder.layers.N.`) and MTP heads
    (`mtp.layers.N.`) return None: their depth is independent of
    `num_hidden_layers` and must never be pruned.
    """
    parts = name.split(".")
    while parts and parts[0] in _DECODER_PREFIXES:
        parts.pop(0)
    if len(parts) < 3 or parts[0] != "layers" or not parts[1].isdigit():
        return None
    return int(parts[1])


def should_skip_layer_weight(name: str, num_hidden_layers: int | None) -> bool:
    """True if `name` belongs to a decoder layer >= num_hidden_layers."""
    if num_hidden_layers is None:
        return False
    layer_idx = decoder_layer_index(name)
    return layer_idx is not None and layer_idx >= num_hidden_layers


def num_hidden_layers_override(model_config: ModelConfig) -> int | None:
    """Returns num_hidden_layers if set in hf_overrides, else None.

    Pruning is opt-in. Some full checkpoints keep extra modules above
    num_hidden_layers, so nothing is dropped unless the user
    explicitly truncated the model.
    """
    overrides = model_config.hf_overrides
    if not isinstance(overrides, dict):
        return None
    if "num_hidden_layers" not in overrides and "num_hidden_layers" not in (
            overrides.get("text_config") or {}):
        return None
    return model_config.hf_text_config.num_hidden_layers


def filter_safetensors_by_layer(hf_folder: str, hf_weights_files: list[str],
                                num_hidden_layers: int) -> list[str]:
    """Drops safetensors shards that only hold layers >= num_hidden_layers.

    Uses model.safetensors.index.json; returns the input unchanged when the
    checkpoint has no index (single shard or non-safetensors).
    """
    index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_path):
        return hf_weights_files
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    needed = {
        shard
        for name, shard in weight_map.items()
        if not should_skip_layer_weight(name, num_hidden_layers)
    }
    kept = [f for f in hf_weights_files if os.path.basename(f) in needed]
    logger.info(
        "num_hidden_layers=%d override: loading %d/%d safetensors shards",
        num_hidden_layers, len(kept), len(hf_weights_files))
    return kept
