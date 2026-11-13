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

    def test_shutdown_kills_actors_before_joining_monitor(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        import threading

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.shutdown_lock = threading.Lock()
        executor.shutting_down = False

        call_order = []
        handle_0 = MagicMock()
        handle_0.rank = 0
        handle_1 = MagicMock()
        handle_1.rank = 1
        executor.ray_worker_handles = [handle_0, handle_1]

        mock_broadcast_mq = MagicMock()
        mock_broadcast_mq.shutdown.side_effect = (
            lambda: call_order.append("broadcast_mq_shutdown"))
        executor.rpc_broadcast_mq = mock_broadcast_mq

        mock_response_mq = MagicMock()
        mock_response_mq.shutdown.side_effect = (
            lambda: call_order.append("response_mq_shutdown"))
        executor.response_mqs = [mock_response_mq]

        mock_ray.kill.side_effect = lambda actor: call_order.append(
            f"ray_kill_{actor}")
        executor._join_monitor_thread = MagicMock(
            side_effect=lambda: call_order.append("join_monitor"))

        executor.shutdown()

        self.assertTrue(executor.shutting_down)
        self.assertIsNone(executor.rpc_broadcast_mq)
        self.assertEqual(executor.response_mqs, [])
        self.assertEqual(
            call_order,
            [
                f"ray_kill_{handle_0.actor}",
                f"ray_kill_{handle_1.actor}",
                "broadcast_mq_shutdown",
                "response_mq_shutdown",
                "join_monitor",
            ],
        )

        # Second call should be a no-op
        call_order.clear()
        executor.shutdown()
        self.assertEqual(call_order, [])

    def test_initialize_ray_cluster_filters_non_tpu_nodes(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        """Non-TPU nodes (e.g. a KubeRay head) must not get a bundle."""
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {
                "NodeID": "head_node",
                "Resources": {
                    "CPU": 8
                }
            },
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
                "NodeID": "no_resources_node"
            },
        ]
        mock_placement_group = MagicMock()
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        executor._initialize_ray_cluster()

        mock_ray.init.assert_not_called()
        specs = mock_ray.util.placement_group.call_args[0][0]
        self.assertEqual(specs, [
            {
                "TPU": 4,
                "node:127.0.0.1": 0.001
            },
            {
                "TPU": 4
            },
        ])
        self.assertEqual(
            mock_ray.util.placement_group.call_args[1]["strategy"], "PACK")
        mock_wait_until_pg_ready.assert_called_once_with(mock_placement_group)
        self.assertEqual(executor.parallel_config.placement_group,
                         mock_placement_group)

    def test_initialize_ray_cluster_pipeline_parallel(self,
                                                      mock_wait_until_pg_ready,
                                                      mock_get_ip,
                                                      mock_platform, mock_ray):
        """With PP>1, one bundle per PP stage sized by sharding_config."""
        mock_platform.ray_device_key = "TPU"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{
            "NodeID": f"tpu_node_{i}",
            "Resources": {
                "TPU": 4
            }
        } for i in range(2)]
        self.vllm_config.parallel_config.pipeline_parallel_size = 2
        self.vllm_config.sharding_config.total_devices = 4

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        executor._initialize_ray_cluster()

        specs = mock_ray.util.placement_group.call_args[0][0]
        self.assertEqual(specs, [
            {
                "TPU": 4,
                "node:127.0.0.1": 0.001
            },
            {
                "TPU": 4
            },
        ])

    def test_initialize_ray_cluster_pipeline_parallel_node_count_mismatch(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        """PP size must equal the number of TPU hosts."""
        mock_platform.ray_device_key = "TPU"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{
            "NodeID": f"tpu_node_{i}",
            "Resources": {
                "TPU": 4
            }
        } for i in range(3)]
        self.vllm_config.parallel_config.pipeline_parallel_size = 2

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        with self.assertRaisesRegex(AssertionError,
                                    "Cannot use PP across hosts"):
            executor._initialize_ray_cluster()
        mock_ray.util.placement_group.assert_not_called()

    def test_initialize_ray_cluster_platform_without_ray_raises(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        mock_platform.ray_device_key = ""
        mock_platform.device_name = "fake_device"
        mock_ray.is_initialized.return_value = True

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config

        with self.assertRaisesRegex(ValueError, "does not support ray"):
            executor._initialize_ray_cluster()
        mock_ray.util.placement_group.assert_not_called()

    def _run_init_executor(self, super_init_side_effect=None):
        """Runs _init_executor with the parent init and cluster init mocked.

        Returns (executor, mocks) where mocks records what the parent
        RayExecutorV2._init_executor observed.
        """
        import vllm.v1.executor.ray_executor_v2 as ray_executor_v2

        orig_init_ray_cluster = ray_executor_v2.initialize_ray_cluster
        observed = {}

        def fake_super_init(_self):
            observed["init_ray_cluster"] = (
                ray_executor_v2.initialize_ray_cluster)
            observed["cluster_initialized_first"] = (
                _self._initialize_ray_cluster.called)
            if super_init_side_effect is not None:
                raise super_init_side_effect

        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.vllm_config.parallel_config
        executor._initialize_ray_cluster = MagicMock()
        executor.collective_rpc = MagicMock(
            return_value=["10.0.0.1:1234", "10.0.0.2:1234"])

        with patch(
                "vllm.v1.executor.ray_executor_v2.RayExecutorV2._init_executor",
                fake_super_init), patch(
                    "tpu_inference.executors.ray_distributed_executor_v2."
                    "maybe_install_hybrid_coordinator_hooks"
                ) as mock_hooks, patch(
                    "tpu_inference.executors.ray_distributed_executor_v2."
                    "set_node_kv_ip_port") as mock_set_kv:
            try:
                executor._init_executor()
            finally:
                # The vLLM hook must always be restored.
                self.assertIs(ray_executor_v2.initialize_ray_cluster,
                              orig_init_ray_cluster)

        observed["orig_init_ray_cluster"] = orig_init_ray_cluster
        observed["hooks"] = mock_hooks
        observed["set_kv"] = mock_set_kv
        return executor, observed

    def test_init_executor_without_kv_connector(self, mock_wait_until_pg_ready,
                                                mock_get_ip, mock_platform,
                                                mock_ray):
        self.vllm_config.kv_transfer_config = None

        executor, observed = self._run_init_executor()

        observed["hooks"].assert_called_once_with(self.vllm_config)
        executor._initialize_ray_cluster.assert_called_once()
        self.assertTrue(observed["cluster_initialized_first"])
        # vLLM's initialize_ray_cluster is a no-op during the parent init so
        # that it doesn't reject multi-chip TPU bundles.
        self.assertIsNot(observed["init_ray_cluster"],
                         observed["orig_init_ray_cluster"])
        self.assertIsNone(observed["init_ray_cluster"](MagicMock()))
        self.assertFalse(executor.has_connector)
        executor.collective_rpc.assert_not_called()
        observed["set_kv"].assert_not_called()

    def test_init_executor_with_kv_connector(self, mock_wait_until_pg_ready,
                                             mock_get_ip, mock_platform,
                                             mock_ray):
        self.vllm_config.kv_transfer_config = MagicMock()

        executor, observed = self._run_init_executor()

        self.assertTrue(executor.has_connector)
        executor.collective_rpc.assert_called_once_with("get_node_kv_ip_port")
        self.assertEqual(
            [c.args for c in observed["set_kv"].call_args_list],
            [("10.0.0.1:1234", ), ("10.0.0.2:1234", )],
        )

    def test_init_executor_restores_ray_cluster_hook_on_failure(
            self, mock_wait_until_pg_ready, mock_get_ip, mock_platform,
            mock_ray):
        with self.assertRaisesRegex(RuntimeError, "parent init failed"):
            # _run_init_executor asserts the hook is restored in `finally`.
            self._run_init_executor(
                super_init_side_effect=RuntimeError("parent init failed"))

    def test_post_init_executor_connects_pp_stages(self,
                                                   mock_wait_until_pg_ready,
                                                   mock_get_ip, mock_platform,
                                                   mock_ray):
        self.vllm_config.parallel_config.pipeline_parallel_size = 4
        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.vllm_config.parallel_config
        executor.world_size = 4
        executor.collective_rpc = MagicMock()

        executor._post_init_executor()

        self.assertEqual(
            [c.kwargs for c in executor.collective_rpc.call_args_list],
            [{
                "unique_reply_rank": 1
            }, {
                "unique_reply_rank": 2
            }, {
                "unique_reply_rank": 3
            }],
        )
        for c in executor.collective_rpc.call_args_list:
            self.assertEqual(c.args, ("initialize_pp_transfer_connect", ))

    def test_post_init_executor_noop_without_pp(self, mock_wait_until_pg_ready,
                                                mock_get_ip, mock_platform,
                                                mock_ray):
        self.vllm_config.parallel_config.pipeline_parallel_size = 1
        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.vllm_config.parallel_config
        executor.world_size = 2
        executor.collective_rpc = MagicMock()

        executor._post_init_executor()

        executor.collective_rpc.assert_not_called()

    def test_output_rank_and_driver_worker(self, mock_wait_until_pg_ready,
                                           mock_get_ip, mock_platform,
                                           mock_ray):
        executor = self.RayDistributedExecutorV2(self.vllm_config)
        executor.world_size = 4

        # The last PP stage produces the final outputs.
        self.assertEqual(executor._get_output_rank(), 3)
        for rank in range(executor.world_size):
            self.assertTrue(executor._is_driver_worker(rank))
