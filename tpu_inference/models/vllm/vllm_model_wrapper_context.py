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
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import \
    SharedAttentionMetadata


@dataclass
class VllmModelWrapperContext:
    kv_caches: List[jax.Array]
    mesh: Mesh
    layer_name_to_kvcache_index: Dict[str, int]
    vllm_config: Optional[VllmConfig] = None
    expert_indices_list: List[jax.Array] = field(default_factory=list)
    shared_attn_metadata: SharedAttentionMetadata = field(default_factory=None)
    # GLM-5.2/DSA sparse-indexer top-k indices, shared by reference across
    # every decoder layer within a single forward pass (written once by an
    # indexer-carrying layer, read by every layer's attention -- including
    # "skip" layers that reuse the most recently computed value; see
    # `layers/vllm/custom_ops/experimental/glm5/glm5_indexer.py` /
    # `layers/vllm/backends/flash_attn_mla_sparse.py`). Deliberately *not*
    # threaded through a persistent torch buffer object mutated in place
    # (`self.topk_indices_buffer[...].copy_(...)`) -- confirmed via a live
    # crash on the TPU VM (`jax.errors.UnexpectedTracerError`) that
    # reassigning a persistent (cross-jit-call) Python object's underlying
    # JAX value from *inside* a `jax.jit` trace leaks a tracer once the
    # trace exits. `VllmModelWrapperContext` is instead constructed fresh
    # per forward pass (see `set_vllm_model_wrapper_context`), exactly like
    # `kv_caches`'s existing "mutate this call's context object, discard it
    # after" idiom, which does not have this problem.
    topk_indices_buffer: Optional[jax.Array] = None


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
    vllm_config: Optional[VllmConfig] = None,
    shared_attn_metadata: SharedAttentionMetadata = None,
):
    global _vllm_model_wrapper_context
    prev_context = _vllm_model_wrapper_context
    _vllm_model_wrapper_context = VllmModelWrapperContext(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        vllm_config=vllm_config,
        shared_attn_metadata=shared_attn_metadata,
    )

    try:
        yield
    finally:
        _vllm_model_wrapper_context = prev_context
