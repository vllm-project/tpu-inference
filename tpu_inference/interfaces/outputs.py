"""
Defines the abstract contracts for model and structured outputs.
"""
from typing import Protocol


class IModelRunnerOutput(Protocol):
    """
    Abstract contract for the output of a model runner.
    """
    # Add methods and properties from vllm.v1.outputs.ModelRunnerOutput
    # that tpu_inference actually uses.
    ...


class IStructuredOutputManager(Protocol):
    """
    Abstract contract for a StructuredOutputManager.
    """
    # Add methods and properties from vllm.v1.structured_output.StructuredOutputManager
    # that tpu_inference actually uses.
    ...
