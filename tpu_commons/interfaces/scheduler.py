"""
This module defines the scheduler interface contract required by tpu_commons.

Dependency Strategy:
- It is ACCEPTABLE to depend on abstract interfaces (like SchedulerInterface) from vllm.
- It is NOT ideal to depend on concrete data structures or managers from vllm.
  These create tight coupling. The dependencies marked with TODO will be
  abstracted away in a future refactoring phase.
"""

from typing import Dict

# TODO(yarongmu-google): Decouple this class.
# This is a concrete manager class from vllm. In the future, this should be
# replaced by an abstract interface (e.g., IEncoderCacheManager).
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
# TODO(yarongmu-google): Decouple this class.
# This is a concrete manager class from vllm. In the future, this should be
# replaced by an abstract interface (e.g., IKVCacheManager).
from vllm.v1.core.kv_cache_manager import KVCacheManager
# This is an abstract interface from vllm. Depending on it is the correct
# approach for loose coupling and is a long-term dependency.
from vllm.v1.core.sched.interface import SchedulerInterface
# TODO(yarongmu-google): Decouple this data structure.
# This is a concrete data structure from vllm. In the future, this should be
# replaced by an abstract interface (e.g., IRequest) that tpu_commons owns.
from vllm.v1.request import Request


class IScheduler(SchedulerInterface):
    """
    An extended interface for the vLLM Scheduler, tailored to the needs
    of advanced orchestration engines.

    It inherits the standard vLLM scheduler interface and adds direct
    access to managers and request dictionaries required for complex
    scheduling strategies like prefill/decode separation.
    """

    @property
    def requests(self) -> Dict[str, Request]:
        ...

    @property
    def kv_cache_manager(self) -> KVCacheManager:
        ...

    @property
    def encoder_cache_manager(self) -> EncoderCacheManager:
        ...
