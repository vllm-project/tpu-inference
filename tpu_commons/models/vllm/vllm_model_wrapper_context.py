from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import jax

from tpu_commons.models.jax.attention_metadata import AttentionMetadata

KVCache = Tuple[jax.Array, jax.Array]


@dataclass
class VllmModelWrapperContext:
    kv_caches: List[KVCache]
    attention_metadata: AttentionMetadata
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
    kv_caches: List[KVCache],
    attention_metadata: AttentionMetadata,
    layer_name_to_kvcache_index: Dict[str, int],
):
    global _vllm_model_wrapper_context
    prev_context = _vllm_model_wrapper_context
    _vllm_model_wrapper_context = VllmModelWrapperContext(
        kv_caches=kv_caches,
        attention_metadata=attention_metadata,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
    )

    try:
        yield
    finally:
        _vllm_model_wrapper_context = prev_context
