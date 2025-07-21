# ruff: noqa
# isort: skip_file
# yapf: disable
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TPU worker class."""
import copy
import os
from typing import Optional, Union

import jax

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from vllm.v1.attention.backends.pallas import TPU_HEAD_SIZE_ALIGNMENT
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats

from tpu_commons.di.abstracts import (AbstractKVCacheConfig,
                                      AbstractSchedulerOutput)
from tpu_commons.di.interfaces import HostInterface
from tpu_commons.logger import init_logger
from tpu_commons.models.torchax.torchax_wrapper import with_torchax_global
from tpu_commons.runner.tpu_torchax_runner import TPUModelRunner
from tpu_commons.worker.base import AbstractTpuWorker
from tpu_commons.worker._temporary_vllm_compat import (
    adapt_kv_cache_config_if_needed)

logger = init_logger(__name__)


class TPUWorker(AbstractTpuWorker):

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False,
                 host_interface: Optional[HostInterface] = None):
        super().__init__(host_interface)
        self.is_driver_worker = is_driver_worker
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        self.original_parallel_config = None
        if self.use_spmd:
            # Under SPMD mode, distributed env is initialized as if there is
            # only one worker/device.
            self.original_parallel_config = copy.deepcopy(self.parallel_config)
            self.parallel_config.tensor_parallel_size = 1
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.world_size = 1
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if rank != local_rank:
            raise NotImplementedError(
                "Multi host serving is not supported yet.")

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Delay profiler initialization to the start of the profiling.
        # This is because in vLLM V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profiler = None
        self.profile_dir = None
        if envs.VLLM_TORCH_PROFILER_DIR and self.rank < 1:
            # For TPU, we can only have 1 active profiler session for 1 profiler
            # server. So we only profile on rank0.
            self.profile_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        self.profile_dir)

        if self.model_config.seed is None:
            self.model_config.seed = 0

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @with_torchax_global
    def init_device(self):
        # Note: Currently the XLA compiler wrongly uses 2D ring strategy on 1D
        # ring, the xla tpu compiler flag
        # `xla_tpu_force_1d_allreduce_at_chunk_count` is a temporary solution to
        # fix this. It will be removed after the bug in XLA compiler is fixed.
        os.environ["LIBTPU_INIT_ARGS"] = (
            os.environ.get("LIBTPU_INIT_ARGS", "") +
            " --xla_tpu_force_1d_allreduce_at_chunk_count=1"
            " --xla_jf_conv_input_fusion=False")
        # --xla_jf_conv_input_fusion=False is used to improve the perf of
        # quantized matmul.
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        self._init_tpu_worker_distributed_environment(
            self.rank, self.distributed_init_method, self.local_rank)

        # Device initialization should happen after initializing
        # the distributed runtime.
        self.device = torch.device("jax:0")
        self.device_config.device = self.device

        rank = jax.process_index()

        # Use persistent cache to avoid XLA recompilation.
        # NOTE(woosuk): Set per-rank cache path since different ranks
        # can have slightly different XLA graphs.
        world_size = self.parallel_config.world_size

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = \
            TPUModelRunner(self.vllm_config, self.device,
                           self.original_parallel_config)

        if rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @with_torchax_global
    def determine_available_memory(self) -> int:
        # `max_num_tokens >= max_num_batched_tokens` due to padding.
        self.model_runner.profile_run(self.model_runner.max_num_tokens)

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        m = jax.local_devices()[0].memory_stats()
        total_memory_size = m["bytes_limit"]
        current_mem = m["bytes_in_use"]
        # TODO: Torchax OOMs if we use the same heuristic as torchxla.
        profiled = m["peak_bytes_in_use"]

        # Calculate the TPU KV cache size based on profiling.
        usable_memory_size = int(total_memory_size *
                                 self.cache_config.gpu_memory_utilization)
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)
        head_size = self.model_config.get_head_size()
        if head_size > 0:
            padded_head_size = cdiv(
                head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
            if padded_head_size != head_size:
                logger.warning_once("head size is padded to %d",
                                    padded_head_size)
            # We adjust the usable memory size for the KV cache to prevent OOM
            # errors, even after padding the head_size.
            tpu_kv_cache_bytes = (tpu_kv_cache_bytes * head_size //
                                  padded_head_size)
        if self.original_parallel_config is not None:
            tpu_kv_cache_bytes *= \
                self.original_parallel_config.tensor_parallel_size

        return int(tpu_kv_cache_bytes)

    def execute_model(
        self,
        scheduler_output: "AbstractSchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        # NOTE: This method intentionally returns a concrete vLLM type, which
        # violates the pure abstract contract of the base class. This is a
        # deliberate, temporary compromise for the same reasons outlined in
        # the `get_kv_cache_spec` method.

        # Unwrap the adapter to get the concrete vLLM object
        vllm_scheduler_output = scheduler_output.vllm_scheduler_output
        output = self.model_runner.execute_model(vllm_scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if self.rank < 1:
            if self.profile_dir is None:
                raise RuntimeError("Profiler is not enabled.")
            profiler = jax.profiler
            if is_start:
                if self.profiler is None:
                    self.profiler = profiler.start_server(9012)
                profiler.start_trace(self.profile_dir)
            else:
                profiler.stop_trace()

    def load_model(self) -> None:
        self.model_runner.load_model()

    @with_torchax_global
    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

    def get_model(self) -> nn.Module:
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

    def _init_tpu_worker_distributed_environment(
        self,
        rank: int,
        distributed_init_method: Optional[str] = None,
        local_rank: int = -1,
    ) -> None:
        """Initialize the distributed environment."""
        # NOTE(woosuk): This is just to initialize the TP group and broadcast
        # the input objects on CPU. The all-reduce and all-gather ops on TPU
        # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
        # own context.
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size)
