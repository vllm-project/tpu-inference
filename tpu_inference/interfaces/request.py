"""
Defines the abstract contract for a Request.
"""
from typing import Any, Protocol


class IRequest(Protocol):
    """
    A minimal, abstract interface for a request.

    This protocol defines only the methods and properties that tpu_inference
    requires to operate. Client libraries (like vLLM) will provide concrete
    implementations that satisfy this contract.
    """

    @property
    def vllm_request(self) -> Any:
        ...

    def is_finished(self) -> bool:
        ...

    def get_request_id(self) -> str:
        ...

    # Add mm_hashes. it's used by `if request.mm_hashes is not None:`.

    # Add other methods and properties from vllm.v1.request.Request that are
    # actually used by the orchestration logic.
    # For example:
    # @property
    # def prompt(self) -> str: ...
    #
    # @property
    # def prompt_token_ids(self) -> list[int]: ...
    #
    # def is_finished(self) -> bool: ...
    #
    # ... etc.
