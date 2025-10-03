"""
Defines the abstract contract for a configuration object.
"""
from typing import Any, Optional, Protocol

from .config_parts import (ICacheConfig, ICompilationConfig, IModelConfig,
                           IParallelConfig, ISchedulerConfig,
                           ISpeculativeConfig)


class IConfig(Protocol):
    """
    A minimal, abstract interface for a configuration object.

    This protocol defines only the methods and properties that tpu_inference
    requires to operate. Client libraries (like vLLM) will provide concrete
    implementations that satisfy this contract.
    """

    @property
    def cache_config(self) -> ICacheConfig:
        ...

    @property
    def compilation_config(self) -> ICompilationConfig:
        ...

    @property
    def model_config(self) -> Optional[IModelConfig]:
        ...

    @property
    def parallel_config(self) -> IParallelConfig:
        ...

    @property
    def scheduler_config(self) -> ISchedulerConfig:
        ...

    @property
    def speculative_config(self) -> Optional[ISpeculativeConfig]:
        ...

    # Escape hatch for direct access when needed by the adapter.
    @property
    def vllm_config(self) -> Any:
        ...
