# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from array import array
from typing import Any, Dict, List, Optional

import ray
import vllm.envs as envs
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE
from vllm.utils.network_utils import (get_distributed_init_method, get_ip,
                                      get_open_port)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.ray_distributed_executor import \
    RayDistributedExecutor as RayDistributedExecutorV1
from vllm.v1.executor.ray_executor import RayWorkerMetaData
from vllm.v1.executor.ray_utils import RayWorkerWrapper, _wait_until_pg_ready

from tpu_inference.logger import init_logger

try:
    from ray._private.state import available_resources_per_node
except ImportError:
    # Ray 2.9.x doesn't expose `available_resources_per_node`
    from ray._private.state import state as _state
    available_resources_per_node = _state._available_resources_per_node

import asyncio
from collections import defaultdict

import msgspec
from vllm.v1.outputs import SamplerOutput

from tpu_inference.distributed.utils import set_node_kv_ip_port

logger = init_logger(__name__)


def _encode_hook(obj: Any) -> Any:
    """Custom msgspec enc hook that supports array types and MultiModalKwargsItem.

    See https://jcristharif.com/msgspec/api.html#msgspec.msgpack.Encoder
    """
    if isinstance(obj, array):
        assert obj.typecode == VLLM_TOKEN_ID_ARRAY_TYPE, (
            f"vLLM array type should use '{VLLM_TOKEN_ID_ARRAY_TYPE}' type. "
            f"Given array has a type code of {obj.typecode}.")
        return obj.tobytes()
    if isinstance(obj, MultiModalKwargsItem):
        return dict(obj)


class RayDistributedExecutor(RayDistributedExecutorV1):
    """Ray-based distributed executor for TPU.

    The implementation is similar to vllm/executor/ray_distributed_executor.py
    with these major differences:

    1. self._init_executor():
       VLLM_USE_RAY_SPMD_WORKER=1, in which the driver worker is the same as other workers.
    2. self._initialize_ray_cluster():
       This sets placement_group_specs for TPU.
       In vLLM one GPU maps to one placement group.
       While here one TPU node with all chips maps to one placement group.
    3. self._init_workers_ray():
       This set TPU resources when create each worker.
       And we omit the driver worker related logic.
    """

    def _init_executor(self) -> None:
        self.forward_dag: Optional[ray.dag.CompiledDAG] = None

        os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"

        # Currently, this requires USE_RAY_SPMD_WORKER=True.
        self.use_ray_compiled_dag = True
        # If it is true, then we do not distinguish between the
        # "driver worker" vs other workers. Also, the rank 0 worker will
        # be executed in a remote Ray worker. Currently this requires
        # USE_RAY_COMPILED_DAG=True.
        self.use_ray_spmd_worker = True

        assert self.uses_ray
        self._initialize_ray_cluster()
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.input_encoder = msgspec.msgpack.Encoder(enc_hook=_encode_hook)
        self.output_decoder = msgspec.msgpack.Decoder(
            Optional[List[SamplerOutput]])

        self.pp_locks: Optional[List[asyncio.Lock]] = None

        self.scheduler_output: SchedulerOutput | None = None

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        if self.has_connector:
            ip_port = self.collective_rpc("get_node_kv_ip_port")
            for item in ip_port:
                set_node_kv_ip_port(item)
        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or not self.vllm_config.ec_transfer_config.is_ec_producer)

    def _initialize_ray_cluster(self) -> None:
        """Initialize the distributed cluster with Ray.

        it will connect to the Ray cluster and create a placement group
        for the workers, which includes the specification of the resources
        for each distributed worker.
        """
        from vllm.platforms import current_platform

        if ray.is_initialized():
            logger.info(
                "Ray is already initialized. Skipping Ray initialization.")
        else:
            logger.warning("Ray is not initialized, this is mainly for test.")
            ray.init()

        device_str = current_platform.ray_device_key
        if not device_str:
            raise ValueError(
                f"current platform {current_platform.device_name} does not "
                "support ray.")

        pp_size = self.parallel_config.pipeline_parallel_size
        placement_group_specs: List[Dict[str, float]] = []

        ray_nodes = ray.nodes()
        logger.info(f"RayDistributedExecutor | ray_nodes={ray_nodes}")

        if pp_size == 1:
            placement_group_specs = [{
                device_str: node['Resources'][device_str]
            } for node in ray_nodes]
        else:
            assert pp_size == len(
                ray_nodes
            ), f"Cannot use PP across hosts, please set --pipeline-parallel-size to 1 or {len(ray_nodes)}"
            num_devices_per_pp_rank = self.vllm_config.sharding_config.total_devices
            placement_group_specs = [{
                device_str: num_devices_per_pp_rank
            } for _ in range(pp_size)]

        # vLLM engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")
        # This way, at least bundle is required to be created in a current
        # node.
        placement_group_specs[0][f"node:{current_ip}"] = 0.001
        logger.info(
            f"RayDistributedExecutor | placement_group_specs={placement_group_specs}"
        )

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

        assert current_placement_group is not None
        # Set the placement group in the parallel config
        self.parallel_config.placement_group = current_placement_group

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # The workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        # Create the workers.
        bundle_indices: List[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            bundle_indices = list(
                map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, \
            ("VLLM_RAY_BUNDLE_INDICES must have the same size"
            f" as the world size, but got {bundle_indices=} "
            f"and {self.parallel_config.world_size=}")
            assert len(set(bundle_indices)) == len(bundle_indices), \
            ("VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
            f" but got {bundle_indices=}")
        else:
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)

        worker_metadata: List[RayWorkerMetaData] = []
        driver_ip = get_ip()
        num_tpu_per_worker = placement_group.bundle_specs[0].get(
            current_platform.ray_device_key, 0)
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=0,
                resources={
                    current_platform.ray_device_key: num_tpu_per_worker
                },
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                       rpc_rank=rank)
            worker_metadata.append(
                RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get([
            each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
            for each in worker_metadata
        ])

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        logger.debug(f"Initialized worker_metadata: {worker_metadata}")

        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(worker_metadata,
                                        key=sort_by_driver_then_worker_ip)
        start_rank = 0
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i + start_rank
        logger.info(
            f"Initialized sorted worker_metadata: {sorted_worker_metadata}")

        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in sorted_worker_metadata
        }
        self.collective_rpc("adjust_rank", args=(rerank_mapping, ))

        # Get the set of TPU IDs used on each node.
        worker_node_and_tpu_ids = []
        for worker in self.workers:
            worker_node_and_tpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote()) \
            ) # type: ignore

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_tpus = defaultdict(list)  # node id -> list of tpu ids

        for i, (node_id, tpu_ids) in enumerate(worker_node_and_tpu_ids):
            node_workers[node_id].append(i)
            # `tpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            tpu_ids = [int(x) for x in tpu_ids]
            node_tpus[node_id].extend(tpu_ids)
        for node_id, tpu_ids in node_tpus.items():
            node_tpus[node_id] = sorted(tpu_ids)
        logger.info(
            f"RayDistributedExecutor | node_workers={node_workers} | node_tpus={node_tpus}"
        )

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            logger.warning(
                f"Got {n_nodes} nodes but with {n_ips} IP addresses. "
                "This is not a typical production setup whose "
                "number of nodes and IPs is euqal. This setup may "
                "lead to unexpected behaviors.")

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [{
            current_platform.device_control_env_var:
            ",".join(map(str, node_tpus[node_id])),
        } for (node_id, _) in worker_node_and_tpu_ids]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="workers")

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self.collective_rpc("update_environment_variables",
                            args=(self._get_env_vars_to_be_updated(), ))

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_tpu_ids):
            local_rank = node_workers[node_id].index(rank)
            ip = sorted_worker_metadata[rank].ip
            prev_ip = sorted_worker_metadata[rank - 1].ip if rank > 0 else ""
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
                ip=ip,
                prev_worker_ip=prev_ip,
            )
            all_kwargs.append(kwargs)
        self.collective_rpc("init_worker", args=(all_kwargs, ))
        self.collective_rpc("init_device")
        if self.parallel_config.pipeline_parallel_size > 1:
            self.collective_rpc("initialize_pp_transfer_connect")
        self.collective_rpc("load_model")

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                num_tp_workers = int(
                    self.parallel_config.tensor_parallel_size //
                    num_tpu_per_worker)
                for tp_rank in range(num_tp_workers):
                    # PP=2, TP=4, num_tpu_per_worker=2
                    # pp_tp_workers = [[0, 1], [2, 3]]
                    rank = (pp_rank * num_tp_workers) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

    # Ray executor do not need handshake metadata
    # as we pass the kv_parameters through proxy server
    def get_kv_connector_handshake_metadata(self) -> None:
        pass
