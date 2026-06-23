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

import unittest
from unittest.mock import MagicMock, patch


class MockVllmConfig:

    def __init__(self):
        self.parallel_config = MagicMock()
        self.parallel_config.world_size = 4
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.pipeline_parallel_size = 1
        self.parallel_config.ray_workers_use_nsight = False
        self.parallel_config.placement_group = None
        self.parallel_config.max_parallel_loading_workers = None

        self.sharding_config = MagicMock()
        self.sharding_config.total_devices = 2

        self.model_config = MagicMock()
        self.cache_config = MagicMock()
        self.lora_config = MagicMock()
        self.load_config = MagicMock()
        self.scheduler_config = MagicMock()
        self.speculative_config = MagicMock()
        self.prompt_adapter_config = MagicMock()
        self.observability_config = MagicMock()
        self.device_config = MagicMock()
        self.ec_transfer_config = MagicMock()


@patch("vllm.v1.executor.ray_executor_v2.RayExecutorV2.__init__",
       lambda x, y: None)
@patch("tpu_inference.executors.ray_distributed_executor_v2.ray")
@patch("tpu_inference.executors.ray_distributed_executor_v2.current_platform")
@patch("tpu_inference.executors.ray_distributed_executor_v2.get_ip",
       return_value="127.0.0.1")
@patch(
    "tpu_inference.executors.ray_distributed_executor_v2._wait_until_pg_ready")
class TestTpuRayDistributedExecutorV2(unittest.TestCase):

    def setUp(self):
        from tpu_inference.executors.ray_distributed_executor_v2 import \
            RayDistributedExecutorV2
        self.RayDistributedExecutorV2 = RayDistributedExecutorV2

        self.vllm_config = MockVllmConfig()
        self.vllm_config.parallel_config.placement_group = None
        self.vllm_config.kv_transfer_config = None

    def test_initialize_ray_cluster_basic(self, mock_wait_until_pg_ready,
                                          mock_get_ip, mock_platform,
                                          mock_ray):
        # --- Setup mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = False
        mock_ray.nodes.return_value = [{
            "NodeID": "node_1",
            "Resources": {
                "TPU": 4
            }
        }]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        mock_wait_until_pg_ready.return_value = None

        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 4}]
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        # --- Test ---
        executor._initialize_ray_cluster()

        # --- Assertions ---
        mock_ray.init.assert_called_once()
        self.assertEqual(executor.parallel_config.placement_group,
                         mock_placement_group)
        mock_ray.util.placement_group.assert_called_once()

    def test_get_parallel_sizes_jax_spmd(self, mock_wait_until_pg_ready,
                                         mock_get_ip, mock_platform, mock_ray):
        self.vllm_config.parallel_config.pipeline_parallel_size = 2
        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.vllm_config.parallel_config

        mock_placement_group = MagicMock()
        # Mock 2 hosts for PP=2 (1 host per stage)
        mock_placement_group.bundle_specs = [{"TPU": 4}, {"TPU": 4}]
        executor.parallel_config.placement_group = mock_placement_group

        tp, pp, pcp = executor._get_parallel_sizes()

        self.assertEqual(tp, 1)
        self.assertEqual(pp, 2)
        self.assertEqual(pcp, 1)
        self.assertEqual(executor.world_size, 2)
        self.assertEqual(executor.local_world_size, 2)

    def test_get_actor_resource_kwargs_reads_pg_bundle(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        mock_platform.ray_device_key = "TPU"
        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 8}]

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.vllm_config.parallel_config
        executor.parallel_config.placement_group = mock_placement_group

        resource_kwargs = executor._get_actor_resource_kwargs()

        self.assertEqual(resource_kwargs, {
            "num_gpus": 0,
            "resources": {
                "TPU": 8
            }
        })

    def test_initialize_ray_cluster_reuses_existing_pg(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        mock_platform.ray_device_key = "TPU"
        existing_pg = MagicMock()

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config
        executor.parallel_config.placement_group = existing_pg

        executor._initialize_ray_cluster()

        mock_ray.util.placement_group.assert_not_called()
        self.assertEqual(executor.parallel_config.placement_group, existing_pg)
