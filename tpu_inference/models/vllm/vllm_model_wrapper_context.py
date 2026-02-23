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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import jax
from jax.sharding import Mesh

from tpu_inference.layers.common.expert_selection import (
    ExpertSelection, LayerExpertSelection)


@dataclass
class VllmModelWrapperContext:
    kv_caches: List[jax.Array]
    mesh: Mesh
    layer_name_to_kvcache_index: Dict[str, int]
    expert_selection: Optional[ExpertSelection] = None

    def record_expert_selection(self, layer_idx: int,
                                selection: LayerExpertSelection) -> None:
        """Record expert selection from a MoE layer during forward pass.

        Args:
            layer_idx: The index of the MoE layer.
            selection: The expert selection for this layer.
        """
        if self.expert_selection is None:
            self.expert_selection = ExpertSelection()
        self.expert_selection.add_layer(layer_idx, selection.topk_weights,
                                        selection.topk_ids)


_vllm_model_wrapper_context: Optional[VllmModelWrapperContext] = None


def get_vllm_model_wrapper_context() -> VllmModelWrapperContext:
    assert _vllm_model_wrapper_context is not None, (
        "VllmModelWrapperContext is not set. "
        "Please use `set_vllm_model_wrapper_context` to set the VllmModelWrapperContext."
    )
    return _vllm_model_wrapper_context


@contextmanager
def set_vllm_model_wrapper_context(
    *,
    kv_caches: List[jax.Array],
    mesh: Mesh,
    layer_name_to_kvcache_index: Dict[str, int] = None,
):
    global _vllm_model_wrapper_context
    prev_context = _vllm_model_wrapper_context
    _vllm_model_wrapper_context = VllmModelWrapperContext(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
    )

    try:
        yield
    finally:
        _vllm_model_wrapper_context = prev_context
