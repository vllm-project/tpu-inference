"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Optional

from tpu_commons.adapters.vllm_adapters import (VllmLoRARequestAdapter,
                                                VllmSchedulerOutputAdapter)
from tpu_commons.di.interfaces import BackendInterface, HostInterface
from tpu_commons.worker.base import AbstractTpuWorker
from tpu_commons.worker.tpu_worker_jax import TPUWorker


class TPUBackend(BackendInterface):
    """
  The main entry point for the host system to interact with the TPU backend.

  This class implements the BackendInterface. It is responsible for creating
  and managing the concrete TPU worker instance and delegating calls to it.
  """

    def __init__(self,
                 host_interface: Optional[HostInterface] = None,
                 **worker_kwargs):
        """
        Initializes the TPUBackend.

        Args:
            host_interface: An optional object that implements the HostInterface,
                            providing a way for the backend to communicate with the host.
            **worker_kwargs: Additional keyword arguments to be passed to the
                            worker's constructor.
        """
        self.worker: AbstractTpuWorker = TPUWorker(
            host_interface=host_interface, **worker_kwargs)

    def launch_tpu_batch(self, batch_to_launch):
        """
        Launches a batch of requests on the TPU worker and returns the result.

        Args:
            batch_to_launch: The batch of requests to be processed.

        Returns:
            The result of the model execution.
        """
        adapted_batch = VllmSchedulerOutputAdapter(batch_to_launch)
        return self.worker.execute_model(adapted_batch)

    def add_lora(self, lora_request):
        """
        Adds a LoRA adapter to the worker.

        Args:
            lora_request: The LoRA request to be processed.
        """
        adapted_lora_request = VllmLoRARequestAdapter(lora_request)
        return self.worker.add_lora(adapted_lora_request)
