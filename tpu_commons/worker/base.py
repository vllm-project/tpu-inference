# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn

from tpu_commons.di.abstracts import (AbstractKVCacheConfig,
                                      AbstractKVCacheSpec, AbstractLoRARequest,
                                      AbstractModelRunnerOutput,
                                      AbstractSchedulerOutput)
from tpu_commons.di.interfaces import HostInterface


class AbstractTpuWorker(ABC):
    """Base class for TPU workers.

    This class defines a pure, host-agnostic contract for what a TPU worker
    must be able to do. It is intentionally decoupled from any specific host
    system like vLLM or SGLang.

    Architectural Note on Dependencies:
    This abstract class only depends on other abstractions (e.g., HostInterface).
    It does NOT hold configuration objects from any specific host (e.g.,
    VllmConfig). Doing so would create a "leaky abstraction," forcing all
    future implementations to depend on a concrete detail from a single host.

    The responsibility for managing concrete configuration is pushed down to the
    concrete subclasses (e.g., TPUWorkerJax), which keeps this base class
    pure and truly reusable across different host systems.
    """

    def __init__(self, host_interface: Optional[HostInterface] = None):
        self.host_interface = host_interface

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache with the given number of blocks."""
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
        scheduler_output: "AbstractSchedulerOutput",
    ) -> Optional[AbstractModelRunnerOutput]:
        pass

    @abstractmethod
    def profile(self, is_start: bool = True):
        pass

    @abstractmethod
    def add_lora(self, lora_request: "AbstractLoRARequest") -> bool:
        """Adds a LoRA adapter to the worker."""
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
    def get_kv_cache_spec(self) -> dict[str, AbstractKVCacheSpec]:
        pass

    @abstractmethod
    def initialize_from_config(self,
                               kv_cache_config: AbstractKVCacheConfig) -> None:
        """Allocate  KV cache with the specified kv_cache_config."""
        pass

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return
