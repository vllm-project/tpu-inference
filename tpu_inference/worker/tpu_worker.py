# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxlib
import jaxtyping
import vllm.envs as vllm_envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.v1 import utils as vllm_utils
from vllm.v1.core.kv_cache_utils import get_num_blocks, get_uniform_page_size
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from tpu_inference import envs, utils
from tpu_inference.distributed.utils import (get_host_ip, get_kv_transfer_port,
                                             get_node_id)
from tpu_inference.layers.jax.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger
from tpu_inference.runner.kv_cache import get_rpa_page_size_bytes
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

_DTYPE: dict[str, jnp.dtype] = {
    "bfloat16": jnp.bfloat16,
    "float": jnp.float32,
    "float32": jnp.float32,
}


class TPUWorker:

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False,
                 devices=None):
        # If we use vLLM's model implementation in PyTorch, we should set it
        # with torch version of the dtype.
        impl = envs.MODEL_IMPL_TYPE
        if impl != "vllm":  # vllm-pytorch implementation does not need this conversion

            # NOTE(wenlong): because sometimes mm needs to use torch for preprocessing
            if not isinstance(vllm_config.model_config.dtype, str):
                logger.warning(
                    "The model dtype is not properly set for JAX backend. "
                    "Overwriting it to jnp.bfloat16")
                vllm_config.model_config.dtype = jnp.bfloat16
            else:
                vllm_config.model_config.dtype = _DTYPE.get(
                    vllm_config.model_config.dtype, jnp.bfloat16)

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.devices = devices if devices is not None else []
        self.device_ranks = set(device.id for device in self.devices
                                if isinstance(device, jaxlib._jax.Device))

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Delay profiler initialization to the start of the profiling.
        # This is because in vLLM V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profile_dir = None
        if vllm_envs.VLLM_TORCH_PROFILER_DIR and self.rank < 1:
            if not self.devices or 0 in self.device_ranks:
                # For TPU, we can only have 1 active profiler session for 1 profiler
                # server. So we only profile on rank0.
                self.profile_dir = vllm_envs.VLLM_TORCH_PROFILER_DIR
                logger.info("Profiling enabled. Traces will be saved to: %s",
                            self.profile_dir)

        use_jax_profiler_server = os.getenv("USE_JAX_PROFILER_SERVER", False)
        # Only one instance of profiler is allowed
        if use_jax_profiler_server and self.rank < 1:
            if not self.devices or 0 in self.device_ranks:
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
        if not self.devices:
            sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
            device_indexes = sharding_config.device_indexes
            if device_indexes is not None and len(device_indexes) > 0:
                # Enforcing the devices sequence to be consistent with the specified device indexes
                all_devices = jax.devices()
                device_dict = {device.id: device for device in all_devices}
                self.devices = []
                for device_index in device_indexes:
                    device = device_dict[device_index]
                    if device is None:
                        raise KeyError(
                            f"Device index {device_index} not found in "
                            f"jax.devices() with IDs {list(device_dict.keys())}!"
                        )
                    self.devices.append(device)
                self.devices = self.devices[:sharding_config.total_devices]
            else:
                self.devices = jax.devices()[:sharding_config.total_devices]

        # Initialize the vLLM distribution layer as a single chip environment,
        # we'll swap the model's parallel modules with TPU SPMD equivalents.
        with set_current_vllm_config(self.vllm_config):
            temp_file = tempfile.mkstemp()[1]
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{temp_file}",
                backend="gloo",
            )
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
        ensure_kv_transfer_initialized(self.vllm_config)
        self.model_runner = TPUModelRunner(self.vllm_config, self.devices)
        logger.info(f"Init worker | "
                    f"rank={self.rank} | "
                    f"node_id={get_node_id()} | "
                    f"is_driver_worker={self.is_driver_worker} | "
                    f"hbm={utils.hbm_usage_gb(self.devices)}GiB")
        vllm_utils.report_usage_stats(self.vllm_config)

    def determine_available_memory(self) -> int:
        gpu_memory_utilization = self.cache_config.gpu_memory_utilization
        hbm_usage = utils.hbm_usage_bytes(self.devices)
        total_hbm_limit = total_hbm_used = 0
        for used, limit in hbm_usage:
            total_hbm_used += used
            total_hbm_limit += limit

        total_hbm_limit_cap = total_hbm_limit * gpu_memory_utilization
        total_hbm_avail = int(total_hbm_limit_cap - total_hbm_used)

        total_hbm_limit_gb = round(total_hbm_limit / utils.GBYTES, 2)
        total_hbm_limit_cap_gb = round(total_hbm_limit_cap / utils.GBYTES, 2)
        total_hbm_used_gb = round(total_hbm_used / utils.GBYTES, 2)
        total_hbm_avail_gb = round(total_hbm_avail / utils.GBYTES, 2)

        logger.info(f"Memory statistics | "
                    f"{total_hbm_limit_gb=}GiB | "
                    f"{total_hbm_limit_cap_gb=}GiB | "
                    f"{total_hbm_used_gb=}GiB | "
                    f"{total_hbm_avail_gb=}GiB")

        if total_hbm_avail <= 0:
            raise ValueError(f"{total_hbm_used_gb=}GiB exceeds "
                             f"{total_hbm_limit_cap_gb=}GiB by "
                             f"{-total_hbm_avail_gb}GiB. Please consider "
                             f"increasing --gpu-memory-utilization from "
                             f"{gpu_memory_utilization} to a larger value.")
        return total_hbm_avail

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Optional[ModelRunnerOutput]:
        # NOTE: This method intentionally returns a concrete vLLM type, which
        # violates the pure abstract contract of the base class. This is a
        # deliberate, temporary compromise for the same reasons outlined in
        # the `get_kv_cache_spec` method.

        output = self.model_runner.execute_model(scheduler_output)

        # With a connector, the scheduler expects output from all workers
        # TODO(mrjunwan): Figure out if this is ok after https://github.com/vllm-project/vllm/pull/26866
        if has_kv_transfer_group():
            return output

        return output if self.is_driver_worker else None

    def sample_tokens(self,
                      grammar_output: GrammarOutput) -> ModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def add_lora(
        self,
        lora_request: LoRARequest,
    ) -> bool:
        raise NotImplementedError(
            "LoRA is not supported by the JAX worker yet.")

    def profile(self, is_start: bool = True):
        if is_start:
            options = jax.profiler.ProfileOptions()
            # default: https://docs.jax.dev/en/latest/profiling.html#general-options
            options.python_tracer_level = os.getenv("PYTHON_TRACER_LEVEL", 0)
            options.host_tracer_level = os.getenv("HOST_TRACER_LEVEL", 1)
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

    def reset_mm_cache(self) -> None:
        pass

    def get_model(self):
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

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
        kv_cache_specs = self.model_runner.get_kv_cache_spec()

        if len(kv_cache_specs) == 0:
            return kv_cache_specs

        # TODO(kyuyeunk): Instead of checking page_size_bytes here, introduce
        # feature that allows overriding page_size_bytes of KVCacheSpec.
        vllm_page_size_bytes = get_uniform_page_size(kv_cache_specs)
        rpa_page_size_bytes = get_rpa_page_size_bytes(self.model_runner.mesh,
                                                      kv_cache_specs)

        if vllm_page_size_bytes != rpa_page_size_bytes:
            logger.info(
                f"KV cache page size calculated by vLLM "
                f"({vllm_page_size_bytes} Bytes) does not match with actual "
                f"page size used by RPA kernel ({rpa_page_size_bytes} Bytes). "
                f"Recalculating number of KV blocks using actual page size.")

            available_memory = self.determine_available_memory()
            num_blocks = get_num_blocks(self.vllm_config, len(kv_cache_specs),
                                        available_memory, rpa_page_size_bytes)

            cache_config = self.vllm_config.cache_config
            cache_config.num_gpu_blocks_override = num_blocks

        return kv_cache_specs

    def initialize_from_config(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def get_node_kv_ip_port(self) -> tuple[int, str, int]:
        node_id = get_node_id()
        ip = get_host_ip()
        port = get_kv_transfer_port()
        return (int(node_id), ip, int(port))

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

    def shutdown(self) -> None:
        return
