from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
from jax.sharding import Mesh


@dataclass
class VllmModelWrapperContext:
    kv_caches: List[jax.Array]
    mesh: Mesh
    layer_name_to_kvcache_index: Dict[str, int]


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
