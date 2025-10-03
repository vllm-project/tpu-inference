"""
This module defines the scheduler interface contract required by tpu_inference.
"""

from typing import Dict, Protocol

# tpu_inference now depends on its own, locally defined interfaces.
from .cache import IEncoderCacheManager, IKVCacheManager
from .request import IRequest


class IScheduler(Protocol):
    """
    An extended interface for a scheduler, tailored to the needs
    of advanced orchestration engines.

    This contract is defined by tpu_inference and must be implemented by
    any client library (like vLLM) that wishes to use this orchestrator.
    """

    @property
    def requests(self) -> Dict[str, IRequest]:
        ...

    @property
    def kv_cache_manager(self) -> IKVCacheManager:
        ...

    @property
    def encoder_cache_manager(self) -> IEncoderCacheManager:
        ...
