"""
Defines the abstract contract for a configuration object.
"""
from typing import Any, Protocol


class IConfig(Protocol):
    """
    A minimal, abstract interface for a configuration object.

    This protocol defines only the methods and properties that tpu_commons
    requires to operate. Client libraries (like vLLM) will provide concrete
    implementations that satisfy this contract.
    """

    # Add methods and properties from vllm.config.VllmConfig that are
    # actually used by the orchestration logic.
    # For example:
    @property
    def scheduler_config(self) -> object:
        ...

    @property
    def cache_config(self) -> object:
        ...

    @property
    def vllm_config(self) -> Any:
        ...
