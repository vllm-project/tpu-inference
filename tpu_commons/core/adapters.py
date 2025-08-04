"""
This module provides adapter classes that wrap concrete vLLM objects and make
them conform to the abstract interfaces defined by tpu_commons.
"""

# These imports are safe as they are inside the adapter layer.
from vllm.config import VllmConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.request import Request

# These are the contracts we are adapting to.
from tpu_commons.interfaces.config import IConfig
from tpu_commons.interfaces.engine import IEngineCore
from tpu_commons.interfaces.request import IRequest
from tpu_commons.interfaces.scheduler import IScheduler


class VllmConfigAdapter(IConfig):
    """ Adapts a vllm.config.VllmConfig to the IConfig interface. """

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config

    # ... delegate methods ...


class VllmSchedulerAdapter(IScheduler):
    """ Adapts a vllm Scheduler to the IScheduler interface. """

    def __init__(self, vllm_scheduler: SchedulerInterface):
        self._vllm_scheduler = vllm_scheduler

    # ... delegate methods ...


class VllmRequestAdapter(IRequest):
    """ Adapts a vllm Request to the IRequest interface. """

    def __init__(self, vllm_request: Request):
        self._vllm_request = vllm_request

    # ... delegate methods ...


class VllmEngineAdapter(IEngineCore):
    """ Adapts a vllm EngineCore to the IEngineCore interface. """

    def __init__(self, vllm_engine: vLLMEngineCore):
        self._vllm_engine = vllm_engine

    # ... delegate methods ...
