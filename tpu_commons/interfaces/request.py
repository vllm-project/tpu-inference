"""
Defines the abstract contract for a Request.
"""
from typing import Protocol


class IRequest(Protocol):
    """
    A minimal, abstract interface for a request.

    This protocol defines only the methods and properties that tpu_commons
    requires to operate. Client libraries (like vLLM) will provide concrete
    implementations that satisfy this contract.
    """

    def is_finished(self) -> bool:
        ...

    def get_request_id(self) -> str:
        ...
