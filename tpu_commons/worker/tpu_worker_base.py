# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from vllm.lora.request import LoRARequest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput


class TPUWorkerBase(ABC):
    """Base class for TPU workers."""

    def __init__(self):
        pass

    @abstractmethod
    def init_device(self):
        """Initialize the TPU device and distributed environment."""
        pass

    @abstractmethod
    def determine_available_memory(self) -> int:
        """Determine available memory for the TPU worker."""
        pass

    @abstractmethod
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        pass

    @abstractmethod
    def profile(self, is_start: bool = True):
        pass

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def compile_or_warm_up_model(self) -> None:
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        pass

    @abstractmethod
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate  KV cache with the specified kv_cache_config."""
        pass

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return
