# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jaxtyping
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.lora.request import LoRARequest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from tpu_commons import utils_jax as utils
from tpu_commons.di.abstracts import (AbstractKVCacheConfig,
                                      AbstractLoRARequest,
                                      AbstractSchedulerOutput)
from tpu_commons.di.interfaces import HostInterface
from tpu_commons.logger import init_logger
from tpu_commons.runner.jax.tpu_jax_runner import TPUModelRunner
from tpu_commons.worker._temporary_vllm_compat import (
    adapt_kv_cache_config_if_needed, adapt_lora_request_if_needed,
    adapt_scheduler_output_if_needed)
from tpu_commons.worker.base import AbstractTpuWorker

logger = init_logger(__name__)


class TPUWorker(AbstractTpuWorker):

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False,
                 devices=None,
                 host_interface: Optional[HostInterface] = None):
        super().__init__(host_interface)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.devices = devices if devices is not None else []

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Delay profiler initialization to the start of the profiling.
        # This is because in vLLM V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profile_dir = None
        if envs.VLLM_TORCH_PROFILER_DIR and self.rank < 1:
            # For TPU, we can only have 1 active profiler session for 1 profiler
            # server. So we only profile on rank0.
            self.profile_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        self.profile_dir)

        logger.info(f"Using devices: {self.devices}")

        use_jax_profiler_server = os.getenv("USE_JAX_PROFILER_SERVER", False)
        if use_jax_profiler_server:
            jax_profiler_server_port = int(
                os.getenv("JAX_PROFILER_SERVER_PORT", 9999))
            logger.info(
                f"Starting JAX profiler server on port {jax_profiler_server_port}"
            )
            jax.profiler.start_server(jax_profiler_server_port)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        ensure_kv_transfer_initialized(self.vllm_config)
        if not self.devices:
            tp = self.parallel_config.tensor_parallel_size
            self.devices = jax.devices()[:tp]
        logger.warning(f"Init devices | "
                       f"devices={self.devices} | "
                       f"hbm={utils.hbm_usage_gb(self.devices)}Gb")

        self.model_runner = TPUModelRunner(self.vllm_config, self.devices)

    def determine_available_memory(self) -> int:
        hbm_usage = utils.hbm_usage_bytes(self.devices)
        hbm_free = [limit - used for used, limit in hbm_usage]
        total_hbm_free = sum(hbm_free)
        taxed_hbm = total_hbm_free * self.cache_config.gpu_memory_utilization
        return taxed_hbm

    def execute_model(
        self,
        scheduler_output: Union[AbstractSchedulerOutput, SchedulerOutput],
    ) -> Optional[ModelRunnerOutput]:
        # NOTE: This method intentionally returns a concrete vLLM type, which
        # violates the pure abstract contract of the base class. This is a
        # deliberate, temporary compromise for the same reasons outlined in
        # the `get_kv_cache_spec` method.

        # Adapt the input if necessary (temporary compatibility layer)
        adapted_scheduler_output = adapt_scheduler_output_if_needed(
            scheduler_output)

        # Unwrap the adapter to get the concrete vLLM object
        vllm_scheduler_output = adapted_scheduler_output.vllm_scheduler_output
        output = self.model_runner.execute_model(vllm_scheduler_output)

        if has_kv_transfer_group():
            finished_sending, finished_recving = (
                get_kv_transfer_group().get_finished(
                    scheduler_output.finished_req_ids))
            if finished_sending or finished_recving:
                if output is EMPTY_MODEL_RUNNER_OUTPUT:
                    output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
                output.finished_sending = finished_sending
                output.finished_recving = finished_recving

            # Clear KVConnector state for this step.
            get_kv_transfer_group().clear_connector_metadata()

            # with a connector, the scheduler expects output from all workers
            return output

        return output if self.is_driver_worker else None

    def add_lora(
        self,
        lora_request: Union[AbstractLoRARequest, LoRARequest],
    ) -> bool:
        # Adapt the input if necessary (temporary compatibility layer)
        adapted_lora_request = adapt_lora_request_if_needed(lora_request)

        # Unwrap the adapter to get the concrete vLLM object
        vllm_lora_request = adapted_lora_request.vllm_lora_request  # noqa: F841

        raise NotImplementedError(
            "LoRA is not supported by the JAX worker yet.")

    def profile(self, is_start: bool = True):
        if is_start:
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = os.getenv("PYTHON_TRACER_LEVEL", 0)
            jax.profiler.start_trace(self.profile_dir,
                                     profiler_options=options)
        else:
            jax.profiler.stop_trace()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        self.model_runner._init_random()

    def get_model(self):
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        # NOTE: This method intentionally returns a concrete vLLM type, which
        # violates the pure abstract contract of the base class. This is a
        # deliberate, temporary compromise.
        #
        # The vLLM executor that calls this method expects the concrete
        # `vllm.KVCacheSpec` object to perform its own internal logic. If we
        # returned an abstract adapter, the vLLM code would break.
        #
        # The ideal long-term solution is for the vLLM DI container to be
        # responsible for this translation. When vLLM can be modified, this
        # method should be changed to return `dict[str, AbstractKVCacheSpec]`,
        # and the vLLM side should be updated to handle the translation.
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(
        self,
        kv_cache_config: Union[AbstractKVCacheConfig, KVCacheConfig],
    ) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        adapted_kv_cache_config = adapt_kv_cache_config_if_needed(
            kv_cache_config)
        vllm_kv_cache_config = adapted_kv_cache_config.vllm_kv_cache_config
        self.model_runner.initialize_kv_cache(vllm_kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def sync_weights(
        self,
        updated_weights: jaxtyping.PyTree,
        mappings: Dict[str, Tuple[str, Tuple[str]]],
        transpose_keys: Dict[str, Tuple[int]],
        reshard_fn: Callable[[jaxtyping.PyTree, jaxtyping.PyTree],
                             jaxtyping.PyTree] = None
    ) -> None:
        """Sync the updated weights to the model runner."""
        return self.model_runner._sync_weights(updated_weights=updated_weights,
                                               mappings=mappings,
                                               transpose_keys=transpose_keys,
                                               reshard_fn=reshard_fn)
