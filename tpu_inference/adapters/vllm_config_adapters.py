"""
Adapters for wrapping concrete vLLM config objects in tpu_inference interfaces.
"""
from typing import Any, Optional

import torch

from tpu_inference.interfaces.config_parts import (ICacheConfig,
                                                   ICompilationConfig,
                                                   IModelConfig,
                                                   IParallelConfig,
                                                   ISchedulerConfig)


class VllmCacheConfigAdapter(ICacheConfig):

    def __init__(self, vllm_cache_config: Any):
        self._vllm_cache_config = vllm_cache_config

    @property
    def block_size(self) -> Optional[int]:
        return self._vllm_cache_config.block_size

    @block_size.setter
    def block_size(self, value: Optional[int]) -> None:
        self._vllm_cache_config.block_size = value


class VllmCompilationConfigAdapter(ICompilationConfig):

    def __init__(self, vllm_compilation_config: Any):
        self._vllm_compilation_config = vllm_compilation_config

    @property
    def level(self) -> Any:
        return self._vllm_compilation_config.level

    @level.setter
    def level(self, value: Any) -> None:
        self._vllm_compilation_config.level = value

    @property
    def backend(self) -> str:
        return self._vllm_compilation_config.backend

    @backend.setter
    def backend(self, value: str) -> None:
        self._vllm_compilation_config.backend = value


class VllmModelConfigAdapter(IModelConfig):

    def __init__(self, vllm_model_config: Any):
        self._vllm_model_config = vllm_model_config

    @property
    def dtype(self) -> torch.dtype:
        return self._vllm_model_config.dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        self._vllm_model_config.dtype = value

    @property
    def use_mla(self) -> bool:
        return self._vllm_model_config.use_mla


class VllmParallelConfigAdapter(IParallelConfig):

    def __init__(self, vllm_parallel_config: Any):
        self._vllm_parallel_config = vllm_parallel_config

    @property
    def worker_cls(self) -> str:
        return self._vllm_parallel_config.worker_cls

    @worker_cls.setter
    def worker_cls(self, value: str) -> None:
        self._vllm_parallel_config.worker_cls = value


class VllmSchedulerConfigAdapter(ISchedulerConfig):

    def __init__(self, vllm_scheduler_config: Any):
        self._vllm_scheduler_config = vllm_scheduler_config

    @property
    def max_num_seqs(self) -> int:
        return self._vllm_scheduler_config.max_num_seqs

    @property
    def is_multi_step(self) -> bool:
        return self._vllm_scheduler_config.is_multi_step

    @property
    def is_multimodal_model(self) -> bool:
        return self._vllm_scheduler_config.is_multimodal_model

    @property
    def disable_chunked_mm_input(self) -> bool:
        return self._vllm_scheduler_config.disable_chunked_mm_input

    @disable_chunked_mm_input.setter
    def disable_chunked_mm_input(self, value: bool) -> None:
        self._vllm_scheduler_config.disable_chunked_mm_input = value

    @property
    def enable_chunked_prefill(self) -> bool:
        return self._vllm_scheduler_config.enable_chunked_prefill

    @enable_chunked_prefill.setter
    def enable_chunked_prefill(self, value: bool) -> None:
        self._vllm_scheduler_config.enable_chunked_prefill = value

    @property
    def chunked_prefill_enabled(self) -> bool:
        return self._vllm_scheduler_config.chunked_prefill_enabled

    @chunked_prefill_enabled.setter
    def chunked_prefill_enabled(self, value: bool) -> None:
        self._vllm_scheduler_config.chunked_prefill_enabled = value

    @property
    def max_model_len(self) -> int:
        return self._vllm_scheduler_config.max_model_len

    @property
    def max_num_batched_tokens(self) -> int:
        return self._vllm_scheduler_config.max_num_batched_tokens

    @max_num_batched_tokens.setter
    def max_num_batched_tokens(self, value: int) -> None:
        self._vllm_scheduler_config.max_num_batched_tokens = value
