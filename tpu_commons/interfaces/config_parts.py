"""
Defines the abstract contracts for the component parts of an IConfig.
"""
from typing import Any, Optional, Protocol

import torch


class IModelConfig(Protocol):

    @property
    def dtype(self) -> torch.dtype:
        ...

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        ...

    @property
    def use_mla(self) -> bool:
        ...


class ICacheConfig(Protocol):

    @property
    def block_size(self) -> Optional[int]:
        ...

    @block_size.setter
    def block_size(self, value: Optional[int]) -> None:
        ...


class IParallelConfig(Protocol):

    @property
    def worker_cls(self) -> str:
        ...

    @worker_cls.setter
    def worker_cls(self, value: str) -> None:
        ...


class ISchedulerConfig(Protocol):

    @property
    def max_num_seqs(self) -> int:
        ...

    @property
    def is_multi_step(self) -> bool:
        ...

    @property
    def is_multimodal_model(self) -> bool:
        ...

    @property
    def disable_chunked_mm_input(self) -> bool:
        ...

    @disable_chunked_mm_input.setter
    def disable_chunked_mm_input(self, value: bool) -> None:
        ...

    @property
    def enable_chunked_prefill(self) -> bool:
        ...

    @enable_chunked_prefill.setter
    def enable_chunked_prefill(self, value: bool) -> None:
        ...

    @property
    def chunked_prefill_enabled(self) -> bool:
        ...

    @chunked_prefill_enabled.setter
    def chunked_prefill_enabled(self, value: bool) -> None:
        ...

    @property
    def max_model_len(self) -> int:
        ...

    @property
    def max_num_batched_tokens(self) -> int:
        ...

    @max_num_batched_tokens.setter
    def max_num_batched_tokens(self, value: int) -> None:
        ...


class ICompilationConfig(Protocol):

    @property
    def level(self) -> Any:
        ...

    @level.setter
    def level(self, value: Any) -> None:
        ...

    @property
    def backend(self) -> str:
        ...

    @backend.setter
    def backend(self, value: str) -> None:
        ...


class ISpeculativeConfig(Protocol):
    ...
