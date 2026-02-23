# Copyright 2025 Google LLC
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

"""Expert selection tracking for MoE models.

This module provides utilities to capture and return per-layer expert routing
decisions (topk_weights and topk_ids) from MoE layers during the forward pass.
This is useful for maintaining consistent routing between inference and training.

Enable this feature by setting the environment variable:
    RETURN_EXPERT_SELECTION=1

When enabled, each MoE layer will return expert routing info alongside its
output. The model collects this per-layer info and returns it as part of
the model output tuple.

The expert selection data for each layer consists of:
    - topk_weights: shape (num_tokens, num_experts_per_tok) - routing weights
    - topk_ids: shape (num_tokens, num_experts_per_tok) - selected expert indices
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def is_expert_selection_enabled() -> bool:
    """Check if expert selection tracking is enabled via env var."""
    return envs.RETURN_EXPERT_SELECTION


@register_pytree_node_class
@dataclass
class ExpertSelection:
    """Container for per-layer expert routing decisions from all MoE layers.

    This is a JAX-pytree-compatible container that stores expert routing
    information from all MoE layers in a model for a single forward pass.

    Attributes:
        topk_weights: List of routing weights per MoE layer.
            Each element has shape (num_tokens, num_experts_per_tok).
        topk_ids: List of selected expert indices per MoE layer.
            Each element has shape (num_tokens, num_experts_per_tok).
        layer_indices: List of the original layer indices for the MoE layers.
    """
    topk_weights: List[jax.Array] = field(default_factory=list)
    topk_ids: List[jax.Array] = field(default_factory=list)
    layer_indices: List[int] = field(default_factory=list)

    def add_layer(self, layer_idx: int, weights: jax.Array,
                  ids: jax.Array) -> None:
        """Record expert selection for a MoE layer.

        Args:
            layer_idx: The index of the MoE layer in the model.
            weights: The routing weights, shape (num_tokens, num_experts_per_tok).
            ids: The selected expert indices, shape (num_tokens, num_experts_per_tok).
        """
        self.topk_weights.append(weights)
        self.topk_ids.append(ids)
        self.layer_indices.append(layer_idx)

    @property
    def num_layers(self) -> int:
        return len(self.topk_ids)

    def tree_flatten(self):
        children = tuple(self.topk_weights) + tuple(self.topk_ids)
        aux_data = (len(self.topk_weights), self.layer_indices)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_layers, layer_indices = aux_data
        obj = cls()
        obj.topk_weights = list(children[:num_layers])
        obj.topk_ids = list(children[num_layers:])
        obj.layer_indices = list(layer_indices)
        return obj

    def __repr__(self):
        if not self.topk_ids:
            return "ExpertSelection(empty)"
        shapes = {
            idx: (w.shape, i.shape)
            for idx, w, i in zip(self.layer_indices, self.topk_weights,
                                 self.topk_ids)
        }
        return f"ExpertSelection(layers={self.layer_indices}, shapes={shapes})"


@register_pytree_node_class
@dataclass
class LayerExpertSelection:
    """Expert selection for a single MoE layer.

    Returned from individual MoE layers when expert selection tracking is enabled.

    Attributes:
        topk_weights: Routing weights, shape (num_tokens, num_experts_per_tok).
        topk_ids: Selected expert indices, shape (num_tokens, num_experts_per_tok).
    """
    topk_weights: jax.Array
    topk_ids: jax.Array

    def tree_flatten(self):
        return ((self.topk_weights, self.topk_ids), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(topk_weights=children[0], topk_ids=children[1])
