"""
This module defines the engine interface contracts required by tpu_commons.

Dependency Strategy:
- It is ACCEPTABLE to depend on abstract interfaces (like IScheduler) from within tpu_commons.
- It is NOT ideal to depend on concrete data structures or managers from vllm.
  These create tight coupling. The dependencies marked with TODO will be
  abstracted away in a future refactoring phase.
"""

from typing import TYPE_CHECKING, Protocol

# TODO(yarongmu-google): Decouple this class.
# This is a concrete cache class from vllm. In the future, this should be
# replaced by an abstract interface (e.g., IMirroredProcessingCache).
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
# TODO(yarongmu-google): Decouple this class.
# This is a concrete manager class from vllm. In the future, this should be
# replaced by an abstract interface (e.g., IStructuredOutputManager).
from vllm.v1.structured_output import StructuredOutputManager

# This is an internal interface. Depending on it is correct.
from tpu_commons.interfaces.scheduler import IScheduler

# This block is only processed by type checkers, not at runtime.
# This prevents a circular import error.
if TYPE_CHECKING:
    # TODO(yarongmu-google): This import points to a concrete vllm class.
    # It should be replaced with an abstract interface in the future.
    from vllm.v1.outputs import ModelRunnerOutput


class IEngineProc(Protocol):
    """
    A high-level interface for any process that can be launched by vLLM.
    It defines the single entry point for starting the process's main loop.
    """

    def run_busy_loop(self) -> None:
        pass


class IDisaggEngineCoreProc(IEngineProc):
    """
    An interface for the disaggregated engine process. It inherits the common
    IEngineProc contract.
    """
    pass


class IEngineCore(Protocol):
    """
    An interface defining the contract for a vLLM Engine Core building block.
    This is a direct mirror of the public API of vllm.v1.engine.core.EngineCore
    that is used by the DisaggEngineCoreProc.
    """
    scheduler: IScheduler
    mm_input_cache_server: MirroredProcessingCache
    structured_output_manager: StructuredOutputManager

    def execute_model_with_error_logging(self, *args,
                                         **kwargs) -> "ModelRunnerOutput":
        pass

    def shutdown(self) -> None:
        pass
