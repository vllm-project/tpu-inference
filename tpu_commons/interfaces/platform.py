"""
Defines the abstract contract for a hardware platform.
"""
from typing import Any, Optional, Protocol, Union

import torch

from .config import IConfig
from .params import IPoolingParams, ISamplingParams


class IPlatform(Protocol):
    """
    A minimal, abstract interface for a hardware platform.
    """

    def can_update_inplace(self) -> bool:
        ...

    def check_and_update_config(self, vllm_config: IConfig) -> None:
        ...

    def get_attn_backend_cls(self, selected_backend: Any, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool, use_mla: bool,
                             has_sink: bool, use_spare: bool) -> str:
        ...

    def get_device_communicator_cls(self) -> str:
        ...

    def get_device_name(self, device_id: int = 0) -> str:
        ...

    def get_device_total_memory(self, device_id: int = 0) -> int:
        ...

    def get_infinity_values(self, dtype: torch.dtype) -> tuple[float, float]:
        ...

    def get_lora_vocab_padding_size(self) -> int:
        ...

    def get_punica_wrapper(self) -> str:
        ...

    def inference_mode(self) -> Any:
        ...

    def is_async_output_supported(self, enforce_eager: Optional[bool]) -> bool:
        ...

    def is_kv_cache_dtype_supported(self, kv_cache_dtype: str) -> bool:
        ...

    def is_pin_memory_available(self) -> bool:
        ...

    def set_device(self, device: torch.device) -> None:
        ...

    def supports_v1(self, model_config: Any) -> bool:
        ...

    def use_all_gather(self) -> bool:
        ...

    def validate_request(
        self,
        prompt: Any,
        params: Union[ISamplingParams, IPoolingParams],
        processed_inputs: Any,
    ) -> None:
        ...
