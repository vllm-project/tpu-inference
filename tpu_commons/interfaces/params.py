"""
Defines the abstract contracts for sampling and pooling parameters.
"""
from typing import Any, Protocol


class IPoolingParams(Protocol):
    """
    Abstract contract for PoolingParams.
    """
    ...


class ISamplingParams(Protocol):
    """
    Abstract contract for SamplingParams.
    """

    @property
    def sampling_type(self) -> Any:
        ...
