"""
Defines the abstract contracts for cache managers.
"""
from typing import Protocol


class IKVCacheManager(Protocol):
    """
    Abstract contract for a KVCacheManager.
    """
    # Add methods and properties from vllm.v1.core.kv_cache_manager.KVCacheManager
    # that tpu_inference actually uses.
    ...


class IEncoderCacheManager(Protocol):
    """
    Abstract contract for an EncoderCacheManager.
    """
    # Add methods and properties from vllm.v1.core.encoder_cache_manager.EncoderCacheManager
    # that tpu_inference actually uses.
    ...


class IMirroredProcessingCache(Protocol):
    """
    Abstract contract for a MirroredProcessingCache.
    """
    # Add methods and properties from vllm.v1.engine.mm_input_cache.MirroredProcessingCache
    # that tpu_inference actually uses.
    ...
