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
@patch("vllm.platforms.current_platform")
@patch("tpu_inference.executors.ray_distributed_executor.get_ip",
       return_value="127.0.0.1")
@patch("tpu_inference.executors.ray_distributed_executor.get_open_port",
       return_value=12345)
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
                                      mock_get_port, mock_get_ip,
                                      mock_vllm_platform, mock_platform,
                                      mock_ray, mock_envs):
        # --- Setup mocks ---
        mock_envs.VLLM_USE_RAY_COMPILED_DAG = True
        mock_envs.VLLM_USE_RAY_SPMD_WORKER = True
        mock_envs.VLLM_RAY_BUNDLE_INDICES = ""

        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_platform.device_control_env_var = "TPU_VISIBLE_CHIPS"
        mock_platform.additional_env_vars = []
        mock_vllm_platform.ray_device_key = "TPU"
        mock_vllm_platform.device_name = "tpu"
        mock_vllm_platform.device_control_env_var = "TPU_VISIBLE_CHIPS"
        mock_vllm_platform.additional_env_vars = []

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
            self, mock_wait_until_pg_ready, mock_get_port, mock_get_ip,
            mock_vllm_platform, mock_platform, mock_ray, mock_envs):
        # --- Setup Mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_vllm_platform.ray_device_key = "TPU"
        mock_vllm_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = False
        # ray.nodes() is now used to get resources
        # Simulate driver node without TPU and worker node with TPU
        mock_ray.nodes.return_value = [{
            "NodeID": "driver_node",
            "Resources": {
                "CPU": 8
            }
        }, {
            "NodeID": "worker_node",
            "Resources": {
                "TPU": 4
            }
        }]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "driver_node"

        executor = self.RayDistributedExecutor(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        # --- Test and Assert ---
        with self.assertRaisesRegex(ValueError,
                                    "Current node has no TPU available"):
            executor._initialize_ray_cluster()

    def test_init_workers_ray_sorts_correctly(self, mock_wait_until_pg_ready,
                                              mock_get_port, mock_get_ip,
                                              mock_vllm_platform,
                                              mock_platform, mock_ray,
                                              mock_envs):
        # --- Setup Mocks ---
        mock_envs.VLLM_RAY_BUNDLE_INDICES = ""
        mock_platform.ray_device_key = "TPU"
        mock_get_ip.return_value = "10.0.0.1"  # Driver IP
        mock_vllm_platform.ray_device_key = "TPU"

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

    def test_initialize_ray_cluster_reuses_existing_placement_group(
            self, mock_wait_until_pg_ready, mock_get_port, mock_get_ip,
            mock_vllm_platform, mock_platform, mock_ray, mock_envs):
        """Test that existing placement group is reused."""
        # --- Setup Mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_vllm_platform.ray_device_key = "TPU"
        mock_vllm_platform.device_name = "tpu"

        # Create a mock placement group (e.g., from ray.serve.llm)
        existing_pg = MagicMock()
        existing_pg.bundle_specs = [{"TPU": 4}] * 4

        executor = self.RayDistributedExecutor(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config
        # Set existing placement group (simulating ray.serve.llm)
        executor.parallel_config.placement_group = existing_pg

        # --- Call method under test ---
        executor._initialize_ray_cluster()

        # --- Assertions ---
        # Should NOT create a new placement group
        mock_ray.util.placement_group.assert_not_called()
        mock_wait_until_pg_ready.assert_not_called()
        # Should keep the existing placement group
        self.assertEqual(executor.parallel_config.placement_group, existing_pg)

    def test_initialize_ray_cluster_filters_non_tpu_nodes(
            self, mock_wait_until_pg_ready, mock_get_port, mock_get_ip,
            mock_vllm_platform, mock_platform, mock_ray, mock_envs):
        """Test that non-TPU nodes are filtered from placement group."""
        # --- Setup Mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_vllm_platform.ray_device_key = "TPU"
        mock_vllm_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = True
        # Simulate cluster with head node (no TPU) and 4 TPU worker nodes
        mock_ray.nodes.return_value = [
            {
                "NodeID": "head_node",
                "Resources": {
                    "CPU": 8
                }
            },  # No TPU
            {
                "NodeID": "tpu_node_1",
                "Resources": {
                    "TPU": 4,
                    "CPU": 160
                }
            },
            {
                "NodeID": "tpu_node_2",
                "Resources": {
                    "TPU": 4,
                    "CPU": 160
                }
            },
            {
                "NodeID": "tpu_node_3",
                "Resources": {
                    "TPU": 4,
                    "CPU": 160
                }
            },
            {
                "NodeID": "tpu_node_4",
                "Resources": {
                    "TPU": 4,
                    "CPU": 160
                }
            },
        ]
        # Current node is a TPU node
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "tpu_node_1"
        mock_get_ip.return_value = "10.0.0.1"

        mock_placement_group = MagicMock()
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = self.RayDistributedExecutor(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config
        executor.parallel_config.placement_group = None

        # --- Call method under test ---
        executor._initialize_ray_cluster()

        # --- Assertions ---
        # Placement group should be created with 4 bundles (only TPU nodes)
        call_args = mock_ray.util.placement_group.call_args
        placement_group_specs = call_args[0][0]
        # Should have 4 bundles (one per TPU node), not 5
        self.assertEqual(len(placement_group_specs), 4)
        # Each bundle should have TPU resources
        for spec in placement_group_specs:
            self.assertIn("TPU", spec)

    def test_initialize_ray_cluster_uses_ray_nodes_for_resource_check(
            self, mock_wait_until_pg_ready, mock_get_port, mock_get_ip,
            mock_vllm_platform, mock_platform, mock_ray, mock_envs):
        """Test that ray.nodes() is used for TPU availability check."""
        # --- Setup Mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_vllm_platform.ray_device_key = "TPU"
        mock_vllm_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = True
        # Simulate node with TPU resources
        mock_ray.nodes.return_value = [
            {
                "NodeID": "tpu_node_1",
                "Resources": {
                    "TPU": 4,
                    "CPU": 160
                }
            },
        ]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "tpu_node_1"
        mock_get_ip.return_value = "10.0.0.1"

        mock_placement_group = MagicMock()
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = self.RayDistributedExecutor(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config
        executor.parallel_config.placement_group = None

        # --- Call method under test ---
        # Should not raise error since ray.nodes() returns TPU resources
        executor._initialize_ray_cluster()

        # --- Assertions ---
        mock_ray.util.placement_group.assert_called_once()
        self.assertIsNotNone(executor.parallel_config.placement_group)


class TestAsyncResultFuture(unittest.TestCase):
    """Tests for AsyncResultFuture."""

    def setUp(self):
        self.ray_patcher = patch(
            "tpu_inference.executors.ray_distributed_executor.ray")
        self.mock_ray = self.ray_patcher.start()
        self.addCleanup(self.ray_patcher.stop)

        from tpu_inference.executors.ray_distributed_executor import \
            AsyncResultFuture
        self.AsyncResultFuture = AsyncResultFuture

    def test_inherits_from_future(self):
        from concurrent.futures import Future
        future = self.AsyncResultFuture(MagicMock(), [MagicMock()])
        self.assertIsInstance(future, Future)

    def test_result_fetches_output_from_first_worker(self):
        result_ids_ref = MagicMock()
        worker0, worker1 = MagicMock(), MagicMock()
        workers = [worker0, worker1]
        result_ids = [10, 20]
        self.mock_ray.get.side_effect = [result_ids, "expected_output"]

        future = self.AsyncResultFuture(result_ids_ref, workers)
        output = future.result(timeout=5)

        # ray.get called first to resolve result_ids
        first_call = self.mock_ray.get.call_args_list[0]
        self.assertEqual(first_call[0][0], result_ids_ref)
        self.assertEqual(first_call[1].get("timeout"), 5)

        # execute_method.remote called on each worker with its result_id
        worker0.execute_method.remote.assert_called_once_with(
            "get_execute_model_output", 10)
        worker1.execute_method.remote.assert_called_once_with(
            "get_execute_model_output", 20)

        # Only the first worker's ref is passed to the final ray.get
        second_call = self.mock_ray.get.call_args_list[1]
        self.assertEqual(second_call[0][0],
                         worker0.execute_method.remote.return_value)
        self.assertEqual(output, "expected_output")


class TestRayWorkerWrapperAsyncScheduling(unittest.TestCase):
    """Tests for async scheduling logic in RayWorkerWrapper."""

    def setUp(self):
        self.parent_init_patcher = patch(
            "vllm.v1.executor.ray_utils.RayWorkerWrapper.__init__",
            return_value=None)
        self.parent_init_patcher.start()
        self.addCleanup(self.parent_init_patcher.stop)

        self.pp_group_patcher = patch(
            "tpu_inference.executors.ray_distributed_executor.get_pp_group")
        self.mock_get_pp_group = self.pp_group_patcher.start()
        self.addCleanup(self.pp_group_patcher.stop)

        from tpu_inference.executors.ray_distributed_executor import \
            RayWorkerWrapper
        self.wrapper = RayWorkerWrapper.__new__(RayWorkerWrapper)
        RayWorkerWrapper.__init__(self.wrapper)

    def _setup_async_worker(self, async_scheduling=True, is_last_rank=True):
        self.wrapper.vllm_config = MagicMock()
        self.wrapper.vllm_config.scheduler_config.async_scheduling = (
            async_scheduling)
        self.mock_get_pp_group.return_value.is_last_rank = is_last_rank
        self.wrapper.worker = MagicMock()
        self.wrapper.worker.model_runner = MagicMock()

    def test_execute_model_ray_without_async_scheduling_calls_super(self):
        self._setup_async_worker(async_scheduling=False)
        execute_input = (MagicMock(), MagicMock())

        with patch(
                "vllm.v1.executor.ray_utils.RayWorkerWrapper.execute_model_ray",
                return_value="super_output") as mock_super:
            result = self.wrapper.execute_model_ray(execute_input)

        mock_super.assert_called_once_with(execute_input)
        self.assertEqual(result, "super_output")

    def test_execute_model_ray_async_returns_result_id(self):
        self._setup_async_worker(async_scheduling=True, is_last_rank=True)
        self.wrapper.worker.model_runner.execute_model.return_value = (
            MagicMock())

        result = self.wrapper.execute_model_ray((MagicMock(), MagicMock()))

        self.assertEqual(result, 1)
        self.assertIn(1, self.wrapper._execute_model_outputs)

    def test_execute_model_ray_async_increments_result_id_per_call(self):
        self._setup_async_worker(async_scheduling=True, is_last_rank=True)
        self.wrapper.worker.model_runner.execute_model.return_value = (
            MagicMock())

        first = self.wrapper.execute_model_ray((MagicMock(), MagicMock()))
        second = self.wrapper.execute_model_ray((MagicMock(), MagicMock()))

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        self.assertIn(1, self.wrapper._execute_model_outputs)
        self.assertIn(2, self.wrapper._execute_model_outputs)

    def test_execute_model_ray_async_samples_when_execute_model_returns_none(
            self):
        self._setup_async_worker(async_scheduling=True, is_last_rank=True)
        grammar_output = MagicMock()
        sample_output = MagicMock()
        self.wrapper.worker.model_runner.execute_model.return_value = None
        self.wrapper.worker.model_runner.sample_tokens.return_value = (
            sample_output)

        self.wrapper.execute_model_ray((MagicMock(), grammar_output))

        self.wrapper.worker.model_runner.sample_tokens.assert_called_once_with(
            grammar_output)
        self.assertEqual(self.wrapper._execute_model_outputs[1], sample_output)

    def test_get_execute_model_output_calls_get_output_for_async_runner_output(
            self):
        from tpu_inference.runner.tpu_runner import AsyncTPUModelRunnerOutput
        self.wrapper.vllm_config = MagicMock()
        self.wrapper.vllm_config.scheduler_config.async_scheduling = True
        mock_output = MagicMock(spec=AsyncTPUModelRunnerOutput)
        mock_final = MagicMock()
        mock_output.get_output.return_value = mock_final
        self.wrapper._execute_model_outputs = {5: mock_output}

        result = self.wrapper.get_execute_model_output(5)

        mock_output.get_output.assert_called_once()
        self.assertEqual(result, mock_final)
        self.assertNotIn(5, self.wrapper._execute_model_outputs)


class TestRayDistributedExecutorExecuteDag(unittest.TestCase):
    """Tests for _execute_dag async scheduling in RayDistributedExecutor."""

    def setUp(self):
        self.init_patcher = patch(
            "tpu_inference.executors.ray_distributed_executor"
            ".RayDistributedExecutor._init_executor",
            return_value=None)
        self.init_patcher.start()
        self.addCleanup(self.init_patcher.stop)

        from tpu_inference.executors.ray_distributed_executor import \
            RayDistributedExecutor
        self.executor = RayDistributedExecutor.__new__(RayDistributedExecutor)
        self.executor.scheduler_config = MagicMock()
        self.executor.forward_dag = None
        self.executor.has_connector = False
        self.executor.workers = [MagicMock(), MagicMock()]

    def tearDown(self):
        # Reset forward_dag to None so that __del__ -> shutdown() does not
        # call ray.kill() on MagicMock workers (which would auto-init Ray and
        # raise ValueError: "ray.kill() only supported for actors").
        self.executor.forward_dag = None

    def test_without_async_scheduling_delegates_to_super(self):
        self.executor.scheduler_config.async_scheduling = False

        with patch(
                "vllm.v1.executor.ray_distributed_executor"
                ".RayDistributedExecutor._execute_dag",
                return_value="super_result") as mock_super:
            result = self.executor._execute_dag(MagicMock(),
                                                MagicMock(),
                                                non_block=False)

        self.assertEqual(mock_super.call_count, 1)
        self.assertEqual(result, "super_result")

    def test_async_scheduling_returns_async_result_future(self):
        from tpu_inference.executors.ray_distributed_executor import \
            AsyncResultFuture
        self.executor.scheduler_config.async_scheduling = True
        mock_dag = MagicMock()
        mock_dag.execute.return_value = MagicMock()

        with patch.object(self.executor,
                          "_compiled_ray_dag",
                          return_value=mock_dag):
            result = self.executor._execute_dag(MagicMock(),
                                                MagicMock(),
                                                non_block=True)

        self.assertIsInstance(result, AsyncResultFuture)
        self.assertIs(result.workers, self.executor.workers)

    def test_async_scheduling_passes_correct_inputs_to_dag(self):
        self.executor.scheduler_config.async_scheduling = True
        scheduler_output = MagicMock()
        grammar_output = MagicMock()
        mock_dag = MagicMock()
        mock_dag.execute.return_value = MagicMock()

        with patch.object(self.executor,
                          "_compiled_ray_dag",
                          return_value=mock_dag):
            self.executor._execute_dag(scheduler_output,
                                       grammar_output,
                                       non_block=True)

        mock_dag.execute.assert_called_once_with(
            (scheduler_output, grammar_output))
