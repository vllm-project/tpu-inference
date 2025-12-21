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

import unittest
from unittest.mock import MagicMock, patch


# Mock VllmConfig and its nested configs to avoid dependencies on the actual
# classes, which can be complex to instantiate for testing.
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


@patch(
    "vllm.v1.executor.ray_distributed_executor.RayDistributedExecutor.__init__",
    lambda x, y: None)
@patch("tpu_inference.executors.ray_distributed_executor.envs")
@patch("tpu_inference.executors.ray_distributed_executor.ray")
@patch("tpu_inference.executors.ray_distributed_executor.current_platform")
@patch("tpu_inference.executors.ray_distributed_executor.get_ip",
       return_value="127.0.0.1")
@patch("tpu_inference.executors.ray_distributed_executor.get_open_port",
       return_value=12345)
@patch(
    "tpu_inference.executors.ray_distributed_executor.available_resources_per_node"
)
@patch("tpu_inference.executors.ray_distributed_executor._wait_until_pg_ready")
class TestTpuRayDistributedExecutor(unittest.TestCase):

    def setUp(self):
        # Import the class under test inside the test method to ensure
        # patches are applied.
        from tpu_inference.executors.ray_distributed_executor import \
            RayDistributedExecutor
        self.RayDistributedExecutor = RayDistributedExecutor

        self.vllm_config = MockVllmConfig()
        # Reset placement group for each test as it might be modified.
        self.vllm_config.parallel_config.placement_group = None
        self.vllm_config.kv_transfer_config = None

    def test_init_executor_basic_flow(self, mock_wait_until_pg_ready,
                                      mock_avail_resources, mock_get_port,
                                      mock_get_ip, mock_platform, mock_ray,
                                      mock_envs):
        # --- Setup mocks ---
        mock_envs.VLLM_USE_RAY_COMPILED_DAG = True
        mock_envs.VLLM_USE_RAY_SPMD_WORKER = True
        mock_envs.VLLM_RAY_BUNDLE_INDICES = ""

        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_platform.device_control_env_var = "TPU_VISIBLE_CHIPS"
        mock_platform.additional_env_vars = []

        mock_ray.is_initialized.return_value = False
        mock_ray.nodes.return_value = [{"Resources": {"TPU": 4}}]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        mock_avail_resources.return_value = {"node_1": {"TPU": 4}}

        mock_wait_until_pg_ready.return_value = None

        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 1}] * 4
        mock_ray.util.placement_group.return_value = mock_placement_group

        mock_worker = MagicMock()
        mock_worker.get_node_and_gpu_ids.remote.return_value = [("node_1",
                                                                 [0, 1, 2, 3])]
        mock_ray.remote.return_value.remote.return_value = mock_worker

        # Simulate remote calls on the worker
        mock_ray.get.side_effect = [
            ["127.0.0.1"] * 4,  # worker_ips
            *[("node_1", [i]) for i in range(4)]  # worker_node_and_tpu_ids
        ]

        executor = self.RayDistributedExecutor(self.vllm_config)
        # Members of the parent class
        executor.uses_ray = True
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config
        executor.collective_rpc = MagicMock()
        executor.collective_rpc.return_value = None

        # --- Initialization ---
        executor._init_executor()

        # --- Assertions ---
        mock_ray.init.assert_called_once()
        self.assertIsNotNone(executor.parallel_config.placement_group)
        self.assertEqual(len(executor.workers), 4)

    def test_initialize_ray_cluster_no_tpu_on_driver_raises_error(
            self, mock_wait_until_pg_ready, mock_avail_resources,
            mock_get_port, mock_get_ip, mock_platform, mock_ray, mock_envs):
        # --- Setup Mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = False
        mock_ray.nodes.return_value = [{"Resources": {"TPU": 4}}]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "driver_node"
        # Simulate no TPUs on the driver node
        mock_avail_resources.return_value = {
            "driver_node": {
                "CPU": 8
            },
            "worker_node": {
                "TPU": 4
            }
        }

        executor = self.RayDistributedExecutor(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        # --- Test and Assert ---
        with self.assertRaisesRegex(ValueError,
                                    "Current node has no TPU available"):
            executor._initialize_ray_cluster()

    def test_init_workers_ray_sorts_correctly(self, mock_wait_until_pg_ready,
                                              mock_avail_resources,
                                              mock_get_port, mock_get_ip,
                                              mock_platform, mock_ray,
                                              mock_envs):
        # --- Setup Mocks ---
        mock_envs.VLLM_RAY_BUNDLE_INDICES = ""
        mock_platform.ray_device_key = "TPU"
        mock_get_ip.return_value = "10.0.0.1"  # Driver IP

        mock_pg = MagicMock()
        mock_pg.bundle_specs = [{"TPU": 1}] * 4

        mock_workers = [MagicMock() for _ in range(4)]
        mock_ray.remote.return_value.return_value.remote.side_effect = mock_workers

        # Simulate IPs for workers created with ranks 0, 1, 2, 3
        worker_ips = ["10.0.0.2", "10.0.0.3", "10.0.0.1", "10.0.0.4"]
        mock_ray.get.side_effect = [
            worker_ips,  # worker_ips
            *[('node_1', ['0', '1', '2', '3']),
              ('node_2', ['4', '5', '6', '7']),
              ('node_3', ['8', '9', '10', '11']),
              ('node_4', ['12', '13', '14', '15'])]  # worker_node_and_tpu_ids
        ]

        executor = self.RayDistributedExecutor(self.vllm_config)
        executor.use_ray_spmd_worker = True
        executor.parallel_config = self.vllm_config.parallel_config
        executor.vllm_config = self.vllm_config
        executor.parallel_config.ray_workers_use_nsight = False
        executor.collective_rpc = MagicMock()
        executor.collective_rpc.return_value = None

        # --- Call method under test ---
        executor._init_workers_ray(mock_pg)

        # --- Assertions ---
        # Expected sorted order of workers: driver, then by IP
        # Original workers: 0 (10.0.0.2), 1 (10.0.0.3), 2 (10.0.0.1), 3 (10.0.0.2)
        # Sorted workers: 2 (driver), 0, 3 (same IP), 1
        self.assertEqual(executor.workers, [
            mock_workers[2], mock_workers[0], mock_workers[1], mock_workers[3]
        ])
