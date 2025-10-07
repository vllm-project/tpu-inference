# SPDX-License-Identifier: Apache-2.0
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.request import Request as VllmRequest
from vllm.v1.request import RequestStatus

from tpu_commons.interfaces.config import IConfig
from tpu_commons.interfaces.engine import IEngineCore, IScheduler
from tpu_commons.interfaces.request import IRequest


class VllmConfigAdapter(IConfig):
    """Wraps a vLLM VllmConfig object to expose it as an IConfig."""

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config

    @property
    def vllm_config(self) -> VllmConfig:
        """Returns the underlying VllmConfig."""
        return self._vllm_config

    @property
    def scheduler_config(self) -> Any:
        return self._vllm_config.scheduler_config

    @property
    def cache_config(self) -> Any:
        return self._vllm_config.cache_config


class VllmSchedulerAdapter(IScheduler):
    """Wraps a vLLM Scheduler to expose it as an IScheduler."""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    @property
    def requests(self) -> dict[str, IRequest]:
        return self._scheduler.requests

    @property
    def kv_cache_manager(self) -> Any:
        return self._scheduler.kv_cache_manager

    @property
    def encoder_cache_manager(self) -> Any:
        return self._scheduler.encoder_cache_manager

    def add_request(self, request: IRequest) -> None:
        # Unwrap the IRequest to pass the concrete vllm.Request
        self._scheduler.add_request(request.vllm_request)

    def __getattr__(self, name: str) -> Any:
        # Pass through other methods like 'schedule', 'has_requests', etc.
        return getattr(self._scheduler, name)


class VllmEngineAdapter(IEngineCore):
    """Wraps a vLLM EngineCore to expose it as an IEngineCore."""

    def __init__(self, engine_core: vLLMEngineCore):
        self._engine_core = engine_core
        # Wrap the concrete scheduler in our scheduler adapter
        self._scheduler = VllmSchedulerAdapter(engine_core.scheduler)

    @property
    def scheduler(self) -> IScheduler:
        # Return the adapted scheduler
        return self._scheduler

    @property
    def model_executor(self) -> Any:
        return self._engine_core.model_executor

    def execute_model_with_error_logging(self, *args, **kwargs) -> Any:
        return self._engine_core.execute_model_with_error_logging(
            *args, **kwargs)

    def shutdown(self) -> None:
        self._engine_core.shutdown()


class VllmRequestAdapter(IRequest):
    """Wraps a vLLM Request object to expose it as an IRequest."""

    def __init__(self, vllm_request: VllmRequest):
        self._vllm_request = vllm_request

    @property
    def vllm_request(self) -> VllmRequest:
        """Provides access to the underlying concrete request for unwrapping."""
        return self._vllm_request

    @property
    def request_id(self) -> str:
        return self._vllm_request.request_id

    @property
    def num_computed_tokens(self) -> int:
        return self._vllm_request.num_computed_tokens

    @property
    def num_output_placeholders(self) -> int:
        return self._vllm_request.num_output_placeholders

    @property
    def num_tokens(self) -> int:
        return self._vllm_request.num_tokens

    @property
    def num_tokens_with_spec(self) -> int:
        return self._vllm_request.num_tokens_with_spec

    @num_computed_tokens.setter
    def num_computed_tokens(self, value: int) -> None:
        self._vllm_request.num_computed_tokens = value

    @property
    def status(self) -> RequestStatus:
        return self._vllm_request.status

    @property
    def prompt_token_ids(self):
        return self._vllm_request.prompt_token_ids

    @property
    def all_token_ids(self):
        return self._vllm_request.all_token_ids

    @property
    def sampling_params(self):
        return self._vllm_request.sampling_params

    @property
    def lora_request(self):
        return self._vllm_request.lora_request

    @property
    def block_hashes(self):
        return self._vllm_request.block_hashes

    @status.setter
    def status(self, value: RequestStatus) -> None:
        self._vllm_request.status = value

    def is_finished(self) -> bool:
        return self._vllm_request.is_finished()

    def get_request_id(self) -> str:
        return self._vllm_request.request_id
