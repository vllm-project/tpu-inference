"""
This module defines the engine interface contracts required by tpu_inference.
"""

from typing import TYPE_CHECKING, Any, Protocol

# tpu_inference now depends on its own, locally defined interfaces.
from .cache import IMirroredProcessingCache
from .outputs import IStructuredOutputManager
from .scheduler import IScheduler

# This block is only processed by type checkers, not at runtime.
if TYPE_CHECKING:
    from .outputs import IModelRunnerOutput


class IEngineProc(Protocol):
    """
    A high-level interface for any process that can be launched by a client.
    It defines the single entry point for starting the process's main loop.
    """

    def run_busy_loop(self) -> None:
        ...


class IDisaggEngineCoreProc(IEngineProc):
    """
    An interface for the disaggregated engine process. It inherits the common
    IEngineProc contract.
    """
    pass


class IEngineCore(Protocol):
    """
    An interface defining the contract for an Engine Core building block.
    This mirrors the public API of a vLLM Engine Core that is used by the
    DisaggEngineCoreProc.
    """
    scheduler: IScheduler
    mm_input_cache_server: IMirroredProcessingCache
    structured_output_manager: IStructuredOutputManager
    model_executor: Any

    def execute_model_with_error_logging(self, *args,
                                         **kwargs) -> "IModelRunnerOutput":
        ...

    def shutdown(self) -> None:
        ...
