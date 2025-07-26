# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

from vllm.config import (CacheConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import ModelRunnerOutput
from vllm.v1.request import Request

# The class we are testing
from tpu_commons.core.core_tpu import DisaggEngineCoreProc


class TestDisaggEngineCoreProc(unittest.TestCase):

    def setUp(self):
        # Patch disagg_utils to control slice configuration.
        self.mock_disagg_utils_patcher = patch(
            'tpu_commons.core.core_tpu.disagg_utils')
        self.mock_disagg_utils = self.mock_disagg_utils_patcher.start()
        self.mock_disagg_utils.get_prefill_slices.return_value = (
            4, )  # One prefill engine
        self.mock_disagg_utils.get_decode_slices.return_value = (
            2, )  # One decode engine
        self.addCleanup(self.mock_disagg_utils_patcher.stop)

        # Patch vLLMEngineCore to avoid its complex initialization.
        self.mock_engine_core_patcher = patch(
            'tpu_commons.core.core_tpu.vLLMEngineCore')
        self.mock_vLLMEngineCore = self.mock_engine_core_patcher.start()
        # Make the mock constructor return another mock.
        self.mock_prefill_engine_instance = MagicMock(
            name="PrefillEngineInstance")
        self.mock_decode_engine_instance = MagicMock(
            name="DecodeEngineInstance")
        self.mock_vLLMEngineCore.side_effect = [
            self.mock_prefill_engine_instance, self.mock_decode_engine_instance
        ]
        self.addCleanup(self.mock_engine_core_patcher.stop)

        # Patch the ZMQ handshake to isolate the test.
        self.mock_handshake_patcher = patch(
            'tpu_commons.core.core_tpu.DisaggEngineCoreProc._perform_handshake'
        )
        self.mock_handshake = self.mock_handshake_patcher.start()
        # Mock the context manager to avoid entering it.
        self.mock_handshake.return_value.__enter__.return_value = MagicMock(
            outputs=["output_addr"], coordinator_output=None)
        self.addCleanup(self.mock_handshake_patcher.stop)

        # Patch threads to avoid them running in the background.
        self.jet_thread_patcher = patch("tpu_commons.core.core_tpu.JetThread",
                                        MagicMock)
        self.mock_jet_thread = self.jet_thread_patcher.start()
        self.addCleanup(self.jet_thread_patcher.stop)

        self.thread_patcher = patch("threading.Thread", MagicMock)
        self.mock_thread = self.thread_patcher.start()
        self.addCleanup(self.thread_patcher.stop)

        # Mock jax.devices
        self.mock_jax_devices_patcher = patch('jax.devices',
                                              return_value=[MagicMock()] * 8)
        self.mock_jax_devices = self.mock_jax_devices_patcher.start()
        self.addCleanup(self.mock_jax_devices_patcher.stop)

        # VLLM Config
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.scheduler_config = MagicMock(
            spec=SchedulerConfig)
        self.mock_vllm_config.parallel_config = MagicMock(spec=ParallelConfig)
        self.mock_vllm_config.cache_config = MagicMock(spec=CacheConfig)

        self.mock_vllm_config.scheduler_config.max_num_seqs = 16
        self.mock_vllm_config.cache_config.block_size = 5
        self.mock_vllm_config.device_config = MagicMock()

    def test_initialization(self):
        """Tests that DisaggEngineCoreProc initializes correctly."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        # Assert correct number of engines were created
        self.mock_disagg_utils.get_prefill_slices.assert_called_once()
        self.mock_disagg_utils.get_decode_slices.assert_called_once()

        # 1 prefill + 1 decode
        self.assertEqual(self.mock_vLLMEngineCore.call_count, 2)
        self.assertEqual(len(proc._prefill_engines), 1)
        self.assertEqual(len(proc._decode_engines), 1)

        # Assert that the correct executor is passed.
        from tpu_commons.core.disagg_executor import DisaggExecutor
        executor_class_arg = self.mock_vLLMEngineCore.call_args_list[0][0][1]
        self.assertIs(executor_class_arg, DisaggExecutor)

    def test_add_request(self):
        """Tests adding a request to the engine."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        mock_engine_request = MagicMock(spec=EngineCoreRequest)
        mock_engine_request.request_id = "test-req-1"
        mock_engine_request.mm_hashes = None  # Simplify by not using multimodal
        mock_engine_request.mm_inputs = []
        mock_engine_request.use_structured_output = False
        mock_engine_request.kv_transfer_params = None

        # Mock the prefill engine's scheduler
        mock_prefill_scheduler = self.mock_prefill_engine_instance.scheduler

        # Call the method under test
        proc.add_request(mock_engine_request)

        # Assert that the request was added to the first prefill engine's scheduler
        mock_prefill_scheduler.add_request.assert_called_once()
        added_vllm_request = mock_prefill_scheduler.add_request.call_args[0][0]
        self.assertIsInstance(added_vllm_request, Request)
        self.assertEqual(added_vllm_request.request_id, "test-req-1")

        # Assert that the request is tracked internally
        self.assertIn("test-req-1", proc._requests)
        self.assertEqual(proc._requests["test-req-1"], added_vllm_request)

    def test_shutdown(self):
        """Tests the shutdown procedure."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        proc.shutdown()

        self.mock_prefill_engine_instance.shutdown.assert_called_once()
        self.mock_decode_engine_instance.shutdown.assert_called_once()

    def test_utility_method_dispatch(self):
        """Tests that utility methods are dispatched to the prefill engine."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        # Mock a method on the prefill engine instance
        self.mock_prefill_engine_instance.list_loras.return_value = {1, 2, 3}

        # Simulate a utility request from a client
        utility_request = (0, "call-id-1", "list_loras", ())
        proc._handle_client_request(EngineCoreRequestType.UTILITY,
                                    utility_request)

        # Check that the method was called on the correct engine
        self.mock_prefill_engine_instance.list_loras.assert_called_once()

        # Check that the result was put on the output queue
        client_idx, output = proc.output_queue.get_nowait()
        self.assertEqual(client_idx, 0)
        self.assertIsInstance(output, EngineCoreOutputs)
        self.assertIsInstance(output.utility_output, UtilityOutput)
        self.assertEqual(output.utility_output.call_id, "call-id-1")
        self.assertEqual(output.utility_output.result, {1, 2, 3})

    def test_prefill_logic(self):
        """Tests the core logic of the _prefill method for one iteration."""
        # 1. Arrange
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        prefill_engine = self.mock_prefill_engine_instance
        transfer_backlog = proc._transfer_backlogs[0]
        _ = proc.output_queue

        # Mock the request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "prefill-req-1"
        mock_request._all_token_ids = [1, 2, 3]
        proc._requests[mock_request.request_id] = mock_request

        # Mock scheduler and its sub-components
        prefill_engine.scheduler = MagicMock()
        prefill_engine.scheduler.has_requests.return_value = True
        mock_scheduler_output = MagicMock(spec=SchedulerOutput)
        mock_scheduler_output.total_num_scheduled_tokens = 10
        prefill_engine.scheduler.schedule.return_value = mock_scheduler_output

        # Mock model execution result
        mock_runner_output = MagicMock(spec=ModelRunnerOutput)
        mock_runner_output.req_id_to_index = {mock_request.request_id: 0}
        mock_runner_output.sampled_token_ids = [[123]
                                                ]  # >0 tokens indicates done
        prefill_engine.execute_model.return_value = mock_runner_output

        # Mock KV cache operations
        prefill_engine.scheduler.kv_cache_manager.get_block_ids.return_value = (
            [10, 11], )
        mock_kv_cache = [MagicMock(name="layer0_cache")]
        prefill_engine.model_executor.driver_worker.model_runner.get_kv_cache_for_block_ids.return_value = mock_kv_cache

        # Mock scheduler update and use it to stop the loop for the test
        mock_engine_outputs = {0: MagicMock(spec=EngineCoreOutputs)}

        def stop_loop_and_return_outputs(*args, **kwargs):
            proc.live = False  # This will stop the while loop in _prefill
            return mock_engine_outputs

        prefill_engine.scheduler.update_from_output.side_effect = stop_loop_and_return_outputs

        # Set up scheduler state required for request removal logic
        prefill_engine.scheduler.requests = {
            mock_request.request_id: mock_request
        }
        prefill_engine.scheduler.running = [mock_request]
        prefill_engine.scheduler._cached_reqs_data = {
            mock_request.request_id: MagicMock()
        }

        # 2. Act
        proc.live = True
        proc._prefill(
            0)  # This will run one iteration and exit via side_effect

        # 3. Assert
        # Check that KV cache was put on transfer backlog
        kv_cache_map = transfer_backlog.get_nowait()
        self.assertIn(mock_request.request_id, kv_cache_map)
        self.assertEqual(kv_cache_map[mock_request.request_id], mock_kv_cache)

        # Check that request was freed from the scheduler's state
        self.assertEqual(len(prefill_engine.scheduler.running), 0)
        prefill_engine.scheduler.kv_cache_manager.free.assert_called_with(
            mock_request)
        self.assertNotIn(mock_request.request_id,
                         prefill_engine.scheduler.requests)

    def test_transfer_logic(self):
        """Tests the _transfer method logic for routing to the least busy engine."""
        # 1. Arrange
        # Configure mocks for multiple decode engines before creating the proc.
        self.mock_disagg_utils.get_decode_slices.return_value = (2, 2)
        self.mock_decode_engine_instance_1 = MagicMock(name="DecodeEngine1")
        self.mock_decode_engine_instance_2 = MagicMock(name="DecodeEngine2")
        self.mock_vLLMEngineCore.side_effect = [
            self.mock_prefill_engine_instance,
            self.mock_decode_engine_instance_1,
            self.mock_decode_engine_instance_2,
        ]

        # Mock scheduler counts to control routing. Engine 2 is less busy.
        self.mock_decode_engine_instance_1.scheduler.get_request_counts.return_value = (
            5, 0)
        self.mock_decode_engine_instance_2.scheduler.get_request_counts.return_value = (
            2, 0)

        # Mock the transfer_kv_cache method on the target engine.
        mock_transferred_cache = MagicMock(name="TransferredKVCache")
        self.mock_decode_engine_instance_2.model_executor.driver_worker.model_runner.transfer_kv_cache.return_value = mock_transferred_cache

        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        # Place item on the transfer backlog.
        transfer_backlog = proc._transfer_backlogs[0]
        mock_kv_cache_map = {"req-1": MagicMock(name="OriginalKVCache")}
        transfer_backlog.put(mock_kv_cache_map)
        transfer_backlog.put(None)  # Sentinel to stop the loop.

        # 2. Act
        proc._transfer(0)

        # 3. Assert
        self.mock_decode_engine_instance_1.model_executor.driver_worker.model_runner.transfer_kv_cache.assert_not_called(
        )
        self.mock_decode_engine_instance_2.model_executor.driver_worker.model_runner.transfer_kv_cache.assert_called_once(
        )

        output_item = proc._decode_backlogs[1].get_nowait()
        self.assertEqual(output_item["req_id"], "req-1")
        self.assertEqual(output_item["cache"], mock_transferred_cache)

    def test_decode_logic(self):
        """Tests the core logic of the _decode method for one iteration."""
        # 1. Arrange
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(),
            log_stats=False,
        )

        decode_engine = self.mock_decode_engine_instance
        decode_backlog = proc._decode_backlogs[0]
        output_queue = proc.output_queue

        # Mock the request and its state as it would be after prefill
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "decode-req-1"
        mock_request.num_computed_tokens = 10  # Simulates it's post-prefill
        proc._requests[mock_request.request_id] = mock_request

        # Place a work item on the decode backlog for the thread to pick up
        mock_kv_cache = [MagicMock(name="layer0_cache_decode")]
        prefill_output = {"req_id": "decode-req-1", "cache": mock_kv_cache}
        decode_backlog.put(prefill_output)
        decode_backlog.put(
            None)  # Sentinel to stop the inner loop after one item

        # Mock the decode engine's scheduler and its sub-components
        decode_engine.scheduler = MagicMock()
        decode_engine.scheduler.has_requests.return_value = False
        decode_engine.scheduler.get_request_counts.return_value = (0, 0)
        decode_engine.scheduler.kv_cache_manager.get_block_ids.return_value = (
            [20, 21], )
        # Set up lists to track state changes
        decode_engine.scheduler.running = []
        decode_engine.scheduler.requests = {}

        # Mock the outputs of the scheduler and model runner
        mock_scheduler_output = MagicMock(spec=SchedulerOutput)
        mock_scheduler_output.scheduled_cached_reqs = MagicMock(
            spec=CachedRequestData)
        mock_scheduler_output.scheduled_cached_reqs.num_computed_tokens = 1
        mock_scheduler_output.scheduled_cached_reqs.new_block_ids = 1
        mock_scheduler_output.total_num_scheduled_tokens = 1
        decode_engine.scheduler.schedule.return_value = mock_scheduler_output

        mock_runner_output = MagicMock(spec=ModelRunnerOutput)
        decode_engine.execute_model.return_value = mock_runner_output

        mock_engine_outputs = {0: MagicMock(spec=EngineCoreOutputs)}

        def stop_loop(*args, **kwargs):
            proc.live = False  # Stop the outer while loop after one full cycle
            return mock_engine_outputs

        decode_engine.scheduler.update_from_output.side_effect = stop_loop
        decode_engine.model_executor.driver_worker.model_runner.input_batch.num_reqs = 3

        # 2. Act
        proc.live = True
        proc._decode(0)

        # 3. Assert
        # Check that the request was inserted and its state updated
        decode_engine.model_executor.driver_worker.model_runner.insert_request_with_kv_cache.assert_called_once(
        )
        self.assertIn(mock_request, decode_engine.scheduler.running)
        self.assertNotIn(mock_request.request_id, proc._requests)

        # Check that the main decode steps were called
        decode_engine.scheduler.schedule.assert_called_once()
        decode_engine.execute_model.assert_called_once_with(
            mock_scheduler_output)

        # Check that the final output was put on the main output queue
        client_idx, output = output_queue.get_nowait()
        self.assertEqual(client_idx, 0)
        self.assertEqual(output, mock_engine_outputs[0])


if __name__ == '__main__':
    unittest.main()
