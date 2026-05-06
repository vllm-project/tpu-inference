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

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import jax
from jax.sharding import Mesh
from vllm.config import VllmConfig


@dataclass
class VllmModelWrapperContext:
    kv_caches: List[jax.Array]
    mesh: Mesh
    layer_name_to_kvcache_index: Dict[str, int]
    vllm_config: Optional[VllmConfig] = None
    expert_indices_list: List[jax.Array] = field(default_factory=list)


_thread_local = threading.local()


def get_vllm_model_wrapper_context() -> VllmModelWrapperContext:
    ctx = getattr(_thread_local, "context", None)
    assert ctx is not None, (
        "VllmModelWrapperContext is not set. "
        "Please use `set_vllm_model_wrapper_context` to set the VllmModelWrapperContext."
    )
    return ctx


@contextmanager
def set_vllm_model_wrapper_context(
    *,
    kv_caches: List[jax.Array],
    mesh: Mesh,
    layer_name_to_kvcache_index: Dict[str, int] = None,
    vllm_config: Optional[VllmConfig] = None,
):
    prev_context = getattr(_thread_local, "context", None)
    _thread_local.context = VllmModelWrapperContext(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        vllm_config=vllm_config,
    )

    try:
        yield
    finally:
        _thread_local.context = prev_context
