# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from tpu_commons import utils_jax as utils
from tpu_commons.logger import init_logger
from tpu_commons.runner.tpu_jax_runner_v2 import TPUModelRunner

logger = init_logger(__name__)


class TPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if rank != local_rank:
            raise NotImplementedError(
                "Multi host serving is not supported yet.")

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

        self._set_visible_devices()

    def _set_visible_devices(self):
        num_request_devices = self.parallel_config.tensor_parallel_size
        num_available_devices = utils.get_local_available_devices()
        if num_request_devices > num_available_devices:
            raise ValueError(
                f"Request {num_request_devices} TPU devices but only {num_available_devices} available"
            )
        device_ids = list(
            range(
                self.local_rank * num_request_devices,
                (self.local_rank + 1) * num_request_devices,
            ))
        utils.set_visible_device_ids(device_ids)

    def init_device(self):
        if self.model_config.seed is None:
            self.model_config.seed = 0
        random_key = utils.init_random(self.model_config.seed)

        self.devices = jax.local_devices()
        self.global_devices = jax.devices()

        logger.info(f"Init devices | "
                    f"local_devices={len(self.devices)} | "
                    f"hbm={utils.hbm_usage_gb(self.devices)}Gb | "
                    f"global_devices={len(self.global_devices)}")

        self.model_runner = TPUModelRunner(self.vllm_config, self.devices,
                                           random_key)

    def determine_available_memory(self) -> int:
        # We don't trigger a dummy batch run to calculate the usage,
        # we get the available size after loading the model directly.
        hbm_usage = utils.hbm_usage_bytes(self.devices)
        hbm_free = [limit - used for used, limit in hbm_usage]
        min_hbm_free = min(hbm_free)
        taxed_hbm = min_hbm_free * self.cache_config.gpu_memory_utilization
        return taxed_hbm

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if is_start:
            jax.profiler.start_trace(self.profile_dir)
        else:
            jax.profiler.stop_trace()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        return

    def get_model(self):
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return
