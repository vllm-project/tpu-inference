# Copyright 2026 Google LLC
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
from typing import Any, Dict, List

import ray
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2
from vllm.v1.executor.ray_utils import _wait_until_pg_ready

from tpu_inference.distributed.utils import set_node_kv_ip_port
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class RayDistributedExecutorV2(RayExecutorV2):
    """Ray-based distributed executor V2 for TPU.

    Inherits from vLLM V1's RayExecutorV2, leveraging the high-performance
    MessageQueue-based control plane, while keeping TPU-specific cluster
    initialization and JAX SPMD parallelisms.
    """

    def _init_executor(self) -> None:
        # Step 1: Pre-initialize the Ray cluster with custom TPU placement group.
        # This ensures the parent class's RayExecutorV2._init_executor()
        # reuses our TPU-optimized placement group.
        self._initialize_ray_cluster()

        # Step 2: Call the parent RayExecutorV2 initialization.
        # We temporarily patch vLLM's initialize_ray_cluster to a no-op during
        # this call to bypass the hardcoded check that rejects bundles with > 1 TPU.
        import vllm.v1.executor.ray_executor_v2 as ray_executor_v2
        orig_init_ray_cluster = ray_executor_v2.initialize_ray_cluster
        ray_executor_v2.initialize_ray_cluster = lambda *args, **kwargs: None

        try:
            super()._init_executor()
        finally:
            ray_executor_v2.initialize_ray_cluster = orig_init_ray_cluster

        # Step 3: Set up KV connector if enabled.
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        if self.has_connector:
            ip_port = self.collective_rpc("get_node_kv_ip_port")
            for item in ip_port:
                set_node_kv_ip_port(item)

    def _initialize_ray_cluster(self) -> None:
        """Initialize the distributed cluster with Ray.

        Creates a TPU-optimized placement group where all chips on a TPU node
        are packed into a single bundle, instead of 1 GPU per bundle.
        """
        # Reuse existing placement group if already provided
        if self.parallel_config.placement_group is not None:
            logger.info(
                f"Using existing placement group: {self.parallel_config.placement_group}"
            )
            return

        # Disable Ray usage stats collection
        if os.environ.get("RAY_USAGE_STATS_ENABLED", "0") != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

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
        logger.info(f"RayDistributedExecutorV2 | ray_nodes={ray_nodes}")

        # Filter nodes that have the required TPU resource
        nodes_with_device = [
            n for n in ray_nodes if device_str in n.get("Resources", {})
        ]
        logger.info(
            f"RayDistributedExecutorV2 | nodes_with_device={len(nodes_with_device)} "
            f"(filtered from {len(ray_nodes)} total nodes)")

        if pp_size == 1:
            placement_group_specs = [{
                device_str: node['Resources'][device_str]
            } for node in nodes_with_device]
        else:
            assert pp_size == len(nodes_with_device), (
                f"Cannot use PP across hosts, please set --pipeline-parallel-size "
                f"to 1 or {len(nodes_with_device)}")
            num_devices_per_pp_rank = self.vllm_config.sharding_config.total_devices
            placement_group_specs = [{
                device_str: num_devices_per_pp_rank
            } for _ in range(pp_size)]

        # Bind the first bundle to the current node (vLLM engine node)
        current_ip = get_ip()
        placement_group_specs[0][f"node:{current_ip}"] = 0.001
        logger.info(
            f"RayDistributedExecutorV2 | placement_group_specs={placement_group_specs}"
        )

        # Pack resources on nodes as much as possible
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

        assert current_placement_group is not None
        self.parallel_config.placement_group = current_placement_group

    def _get_actor_resource_kwargs(self) -> dict[str, Any]:
        """Return Ray actor resource kwargs for the TPU platform.

        For TPU, one Ray actor runs on one node and consumes all TPU resources
        of that node, defined by the placement group bundle.
        """
        placement_group = self.parallel_config.placement_group
        device_key = current_platform.ray_device_key
        # Get the number of TPU chips in the first bundle
        num_tpu_per_worker = placement_group.bundle_specs[0].get(device_key, 0)
        return {"num_gpus": 0, "resources": {device_key: num_tpu_per_worker}}

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        """Override to support JAX SPMD.

        Each TPU host runs a single JAX process.
        - pp_size is the actual pipeline parallel size.
        - tp_size is the number of TPU hosts per pipeline stage.
        - world_size is the total number of TPU hosts (tp_size * pp_size).
        """
        placement_group = self.parallel_config.placement_group
        assert placement_group is not None, "Placement group must be initialized first"

        total_hosts = len(placement_group.bundle_specs)
        actual_pp_size = self.parallel_config.pipeline_parallel_size

        # Calculate number of hosts per pipeline stage
        hosts_per_stage = total_hosts // actual_pp_size
        assert total_hosts % actual_pp_size == 0, (
            f"Total TPU hosts ({total_hosts}) must be divisible by "
            f"pipeline_parallel_size ({actual_pp_size})")

        self.world_size = total_hosts
        self.local_world_size = total_hosts

        tp_size = hosts_per_stage
        pp_size = actual_pp_size
        pcp_size = 1
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        # Set up JAX pipeline parallel transfer connection across PP stages.
        # Only needed if actual pipeline parallel size is > 1.
        if self.parallel_config.pipeline_parallel_size > 1:
            for rank in range(1, self.world_size):
                self.collective_rpc("initialize_pp_transfer_connect",
                                    unique_reply_rank=rank)

    def _get_output_rank(self) -> int:
        # The last PP stage produces the final token outputs.
        return self.world_size - 1

    def _is_driver_worker(self, rank: int) -> bool:
        # All spawned workers are driver workers for their respective stages.
        return True
