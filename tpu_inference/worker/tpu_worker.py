# SPDX-License-Identifier: Apache-2.0

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import jax
import jaxlib
import jaxtyping
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.tasks import SupportedTask
from vllm.v1 import utils as vllm_utils
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from tpu_inference import envs, utils
from tpu_inference.distributed import jax_parallel_state
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.distributed.utils import (get_device_topology_order_id,
                                             get_host_ip, get_kv_transfer_port)
from tpu_inference.layers.common.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


@dataclass
class PPConfig:
    vllm_config: VllmConfig
    rank: int
    ip: str
    prev_worker_ip: str
    pp_world_size: int

    # default env vars for
    # TPU_PROCESS_BOUNDS, TPU_CHIPS_PER_PROCESS_BOUNDS, TPU_VISIBLE_CHIPS
    # if PP is used in single host.
    default_tpu_process_bounds: str = field(init=False)
    default_tpu_chips_per_process_bounds: str = field(init=False)
    default_tpu_visible_chips: str = field(init=False)

    def __post_init__(self):
        # In the isolated stage architecture, each pipeline stage operates
        # as an independent JAX cluster.
        # For single-host PP or multi-host with one host per stage,
        # the cluster size for each stage is 1.

        # Defines the 3D topology of the JAX processes (nodes) in the cluster. Syntax: "a,b,c"
        self.default_tpu_process_bounds = "1,1,1"
        # Defines how many physical TPU chips are allocated to each JAX process. Syntax: "a,b,c"
        self.default_tpu_chips_per_process_bounds = "1,1,1"
        # Specifies the physical IDs of the TPU chips that this specific process is allowed to see and use. Syntax: comma separated ints like "a", "a,b"
        self.default_tpu_visible_chips = f"{self.rank}"

        if self.pp_world_size > 1:
            # We need to know if we are on a 2-core-per-chip (v7) or 1-core-per-chip (v6) system
            from tpu_inference import tpu_info
            cores_per_chip = tpu_info.get_num_cores_per_chip()

            # This tells us how many logical JAX devices (cores) this specific pipeline stage needs. If you are doing Tensor Parallelism (TP) within a stage, total_devices would be the TP size.
            sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
            total_cores_per_stage = sharding_config.total_devices

            # Number of physical chips needed for this stage.
            chips_per_stage = math.ceil(total_cores_per_stage / cores_per_chip)

            if chips_per_stage > 0:
                start_chip = self.rank * chips_per_stage
                self.default_tpu_visible_chips = ",".join(
                    str(i)
                    for i in range(start_chip, start_chip + chips_per_stage))

                # Set bounds to match the visible chips exactly.
                self.default_tpu_chips_per_process_bounds = f"1,{chips_per_stage},1"


class TPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        devices=None,
        ip: str = "localhost",
        prev_worker_ip: str = "localhost",
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.devices = devices if devices is not None else []
        self.device_ranks = set(device.id for device in self.devices
                                if isinstance(device, jaxlib._jax.Device))
        self.pp_config = PPConfig(vllm_config, rank, ip, prev_worker_ip,
                                  self.parallel_config.pipeline_parallel_size)

        # Explicitly trigger RunAI download on the worker if needed.
        # This handles downloading config.json and other non-weight files to the
        # worker's local cache before VllmModelWrapper initialization.
        if hasattr(self.model_config, "maybe_pull_model_tokenizer_for_runai"):
            self.model_config.maybe_pull_model_tokenizer_for_runai(
                self.model_config.model, self.model_config.tokenizer)

        # Delay profiler initialization to the start of the profiling.
        # This is because in vLLM V1, MP runtime is initialized before the
        # TPU Worker is initialized. The profiler server needs to start after
        # MP runtime is initialized.
        self.profile_dir = None
        self.vllm_config.profiler_config
        profiler_config = self.vllm_config.profiler_config
        if profiler_config.profiler == "torch" and self.rank < 1 and self.pp_config.pp_world_size == 1:
            if not self.devices or 0 in self.device_ranks:
                # For TPU, we can only have 1 active profiler session for 1 profiler
                # server. So we only profile on rank0.
                self.profile_dir = profiler_config.torch_profiler_dir
                logger.info("Profiling enabled. Traces will be saved to: %s",
                            self.profile_dir)

        # For PP, we use MPMD so we want to profile every worker.
        if self.pp_config.pp_world_size > 1 and profiler_config.profiler == "torch":
            self.profile_dir = os.path.join(
                profiler_config.torch_profiler_dir,
                f"pprank_{self.rank}_ppworldsize_{self.pp_config.pp_world_size}"
            )
            os.makedirs(self.profile_dir, exist_ok=True)

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

        # step_counter is used to calculate uuid to transfer intermediate tensors.
        self.step_counter = 0

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self,
                    tpu_process_bounds="",
                    tpu_chips_per_process_bounds="",
                    tpu_visible_chips=""):
        # set tpu visible devices for Jax runtime in single host PP.
        multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "").lower()
        if multihost_backend != "ray" and self.parallel_config.pipeline_parallel_size > 1:
            tpu_ports = [
                jax_parallel_state.BASE_JAX_PORT + i
                for i in range(self.pp_config.pp_world_size)
            ]
            os.environ["TPU_PROCESS_ADDRESSES"] = ",".join(
                [f"localhost:{port}" for port in tpu_ports])
            os.environ["TPU_PROCESS_PORT"] = f"{tpu_ports[self.rank]}"
            os.environ["CLOUD_TPU_TASK_ID"] = f"{self.rank}"

            # Note: Below is the setting for v6e8 host (8 chips of v6e)
            # Replace with your own topology.
            # There are 2 ways of subslicing a v6e
            # 1) 2 slices with 4 TPU chips each, we can do PP=2, TP=1/2/3/4
            #   TPU_PROCESS_BOUNDS = "1,1,1"
            #   TPU_CHIPS_PER_PROCESS_BOUNDS = "1,4,1"
            #   TPU_VISIBLE_CHIPS = "0,1,2,3" or "4,5,6,7"
            # 2) 1 chip for each subslice, with at most 8 subslices,
            #    we can do TP=1, PP=1/2/3/4/5/6/7/8
            os.environ[
                "TPU_PROCESS_BOUNDS"] = tpu_process_bounds \
                    if tpu_process_bounds \
                        else self.pp_config.default_tpu_process_bounds
            os.environ[
                "TPU_CHIPS_PER_PROCESS_BOUNDS"] = tpu_chips_per_process_bounds \
                    if tpu_chips_per_process_bounds \
                        else self.pp_config.default_tpu_chips_per_process_bounds
            os.environ[
                "TPU_VISIBLE_CHIPS"] = tpu_visible_chips \
                    if tpu_visible_chips \
                        else self.pp_config.default_tpu_visible_chips

        if not self.devices:
            sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
            device_indexes = sharding_config.device_indexes
            if device_indexes is not None and len(device_indexes) > 0:
                # Enforcing the devices sequence to be consistent with the specified device indexes
                all_local_devices = jax.local_devices()
                device_dict = {
                    device.id: device
                    for device in all_local_devices
                }
                self.devices = []
                for device_index in device_indexes:
                    device = device_dict[device_index]
                    if device is None:
                        raise KeyError(
                            f"Device index {device_index} not found in "
                            f"jax.local_devices() with IDs {list(device_dict.keys())}!"
                        )
                    self.devices.append(device)
                assert len(self.devices) >= sharding_config.total_devices
                self.devices = self.devices[:sharding_config.total_devices]
            else:
                if self.pp_config.pp_world_size > 1:
                    # We only support a mixed tp + pp scenario that tp size is
                    #  smaller or equals the total TPUs in one node
                    # say: we have 4 nodes with 4 TPUs each, we can only do pp:4, tp:4, but not pp:2, tp:8
                    assert jax.local_device_count(
                    ) >= sharding_config.total_devices
                    self.devices = jax.local_devices()[:sharding_config.
                                                       total_devices]
                else:
                    # In a multi-host distributed env, say: Ray, local_device count may smaller
                    # than the total devices, we just choose the smaller set here.
                    self.devices = jax.devices()[:sharding_config.
                                                 total_devices]

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

        jax_parallel_state.init_pp_distributed_environment(
            self.pp_config.ip,
            self.rank,
            self.parallel_config.pipeline_parallel_size,
            self.devices[0],
            need_pp=self.parallel_config.pipeline_parallel_size > 1)

        ensure_kv_transfer_initialized(self.vllm_config)

        is_first_rank = True
        is_last_rank = True
        self.topology_order_id = self.rank
        if self.parallel_config.pipeline_parallel_size > 1:
            is_first_rank = self.rank == 0
            is_last_rank = self.rank == self.pp_config.pp_world_size - 1
        else:
            # topology_order_id is used to determine the KV cache
            # mapping between P/D workers
            if multihost_backend == "ray":
                self.topology_order_id = get_device_topology_order_id(
                    jax.local_devices(), jax.devices())

        self.model_runner = TPUModelRunner(self.vllm_config, self.devices,
                                           self.rank, is_first_rank,
                                           is_last_rank)
        logger.info(f"Init worker | "
                    f"rank={self.rank} | "
                    f"is_first_rank={is_first_rank} | "
                    f"is_last_rank={is_last_rank} | "
                    f"topology_order_id={self.topology_order_id} | "
                    f"is_driver_worker={self.is_driver_worker} | "
                    f"hbm={utils.hbm_usage_gb(self.devices)}GiB |"
                    f"self.devices={self.devices} | "
                    f"total devices={jax.devices()} | "
                    f"local_devices={jax.local_devices()}")
        vllm_utils.report_usage_stats(self.vllm_config)

    def initialize_pp_transfer_connect(self):
        if self.rank == 0:
            return
        jax_parallel_state.connect(self.pp_config.prev_worker_ip,
                                   self.rank - 1)

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

        if self.parallel_config.pipeline_parallel_size == 1 or self.rank == 0:
            intermediate_tensors = None
        else:
            # receive intermediate tensors
            uuid = self.model_runner.get_uuid_for_jax_transfer(
                scheduler_output, self.rank - 1, self.step_counter)
            # TODO: this method might only works for vllm model, not sure about jax models.
            tensor_spec = self.model_runner.get_intermediate_tensor_spec(
                scheduler_output.total_num_scheduled_tokens)
            intermediate_tensors_dict = get_pp_group().recv_tensor_dict(
                uuid, tensor_spec)
            intermediate_tensors = JaxIntermediateTensors(
                intermediate_tensors_dict)

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        if isinstance(output, JaxIntermediateTensors):
            assert self.parallel_config.pipeline_parallel_size > 1
            assert not get_pp_group().is_last_rank
            # send intermediate tensors
            uuid = self.model_runner.get_uuid_for_jax_transfer(
                scheduler_output, self.rank, self.step_counter)
            get_pp_group().send_tensor_dict(uuid, output.tensors)
            self.step_counter += 1
            return None
        else:
            self.step_counter += 1
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

    def profile(self,
                is_start: bool = True,
                profile_prefix: str | None = None):
        if is_start:
            options = jax.profiler.ProfileOptions()
            # default: https://docs.jax.dev/en/latest/profiling.html#general-options
            options.python_tracer_level = envs.PYTHON_TRACER_LEVEL
            options.host_tracer_level = os.getenv("HOST_TRACER_LEVEL", 1)
            jax.profiler.start_trace(self.profile_dir,
                                     profiler_options=options)
        else:
            jax.profiler.stop_trace()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> float:
        self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        self.model_runner._init_random()
        return self.compilation_config.compilation_time

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
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        # Precompile functions with large vocab_size tensors before allocating KV cache to avoid OOM
        if not (envs.SKIP_JAX_PRECOMPILE or
                (hasattr(self.model_runner.model_config, "enforce_eager")
                 and self.model_runner.model_config.enforce_eager)):
            self.model_runner.compilation_manager._precompile_sampling()
            self.model_runner.compilation_manager._precompile_gather_logprobs()
        self.model_runner.initialize_kv_cache(kv_cache_config,
                                              self.topology_order_id)

    def get_node_kv_ip_port(self) -> tuple[int, str, int]:
        ip = get_host_ip()
        port = get_kv_transfer_port()
        return (int(self.topology_order_id), ip, int(port))

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

    def delete_kv_cache(self) -> None:
        self.model_runner.delete_kv_cache()

    def reinitialize_kv_cache(self) -> None:
        self.model_runner.reinitialize_kv_cache()

    # Ray executor do not need handshake metadata
    # as we pass the kv_parameters through proxy server
    def get_kv_connector_handshake_metadata(self) -> None:
        pass
