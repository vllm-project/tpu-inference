# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

from vllm.config import ParallelConfig, VllmConfig
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request

from tpu_inference.core.core_tpu import (DisaggEngineCore,
                                         DisaggEngineCoreProc,
                                         _DisaggOrchestrator)


class TestDisaggEngineCore(unittest.TestCase):

    def setUp(self):
        # Patch disagg_utils to control slice configuration.
        self.mock_disagg_utils_patcher = patch(
            'tpu_inference.core.core_tpu.disagg_utils')
        self.mock_disagg_utils = self.mock_disagg_utils_patcher.start()
        self.mock_disagg_utils.get_prefill_slices.return_value = (
            4, )  # One prefill engine
        self.mock_disagg_utils.get_decode_slices.return_value = (
            2, )  # One decode engine
        self.addCleanup(self.mock_disagg_utils_patcher.stop)

        # Patch the orchestrator to test the adapter in isolation
        self.mock_orchestrator_patcher = patch(
            'tpu_inference.core.core_tpu._DisaggOrchestrator')
        self.mock_orchestrator = self.mock_orchestrator_patcher.start()
        self.addCleanup(self.mock_orchestrator_patcher.stop)

        # Patch vLLMEngineCore to avoid its complex initialization.
        self.mock_engine_core_patcher = patch(
            'tpu_inference.core.core_tpu.vLLMEngineCore')
        self.mock_vLLMEngineCore = self.mock_engine_core_patcher.start()
        self.addCleanup(self.mock_engine_core_patcher.stop)

        # Mock jax.devices
        self.mock_jax_devices_patcher = patch('jax.devices',
                                              return_value=[MagicMock()] * 8)
        self.mock_jax_devices = self.mock_jax_devices_patcher.start()
        self.addCleanup(self.mock_jax_devices_patcher.stop)

        # VLLM Config
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.parallel_config = MagicMock(spec=ParallelConfig)
        self.mock_vllm_config.device_config = MagicMock()
        self.mock_vllm_config.cache_config = MagicMock()
        self.mock_vllm_config.cache_config.prefix_caching_hash_algo = "builtin"
        self.mock_vllm_config.cache_config.block_size = 5
        self.mock_vllm_config.__post_init__ = MagicMock()

    def test_initialization(self):
        """Tests that the adapter initializes the orchestrator correctly."""
        engine = DisaggEngineCore(
            vllm_config=self.mock_vllm_config,
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        self.mock_orchestrator.assert_called_once()
        args, kwargs = self.mock_orchestrator.call_args
        self.assertIsInstance(kwargs['config'], VllmConfig)
        self.assertEqual(kwargs['config'], self.mock_vllm_config)
        self.assertEqual(kwargs['output_queue'], engine.output_queue)
        self.assertEqual(len(kwargs['prefill_engines']), 1)
        self.assertEqual(len(kwargs['decode_engines']), 1)
        self.assertEqual(kwargs['prefill_slice_sizes'], (4, ))
        self.assertEqual(kwargs['decode_slice_sizes'], (2, ))

    def test_add_request(self):
        """Tests that the adapter correctly delegates add_request to the orchestrator."""
        engine = DisaggEngineCore(
            vllm_config=self.mock_vllm_config,
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"
        mock_request.pooling_params = None
        mock_request.kv_transfer_params = None

        engine.add_request(mock_request)

        self.mock_orchestrator.return_value.add_request.assert_called_once()
        # Get the argument passed to add_request
        passed_request = self.mock_orchestrator.return_value.add_request.call_args[
            0][0]

        # Assert it's the correct type (the Request directly)
        self.assertIsInstance(passed_request, Request)
        self.assertEqual(passed_request.request_id, "test_req")

    def test_shutdown(self):
        """Tests that the adapter correctly delegates shutdown to the orchestrator."""
        engine = DisaggEngineCore(
            vllm_config=self.mock_vllm_config,
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        engine.shutdown()

        self.mock_orchestrator.return_value.shutdown.assert_called_once()


class TestDisaggEngineCoreProc(unittest.TestCase):

    def setUp(self):
        # Patch disagg_utils to control slice configuration.
        self.mock_disagg_utils_patcher = patch(
            'tpu_inference.core.core_tpu.disagg_utils')
        self.mock_disagg_utils = self.mock_disagg_utils_patcher.start()
        self.mock_disagg_utils.get_prefill_slices.return_value = (
            4, )  # One prefill engine
        self.mock_disagg_utils.get_decode_slices.return_value = (
            2, )  # One decode engine
        self.addCleanup(self.mock_disagg_utils_patcher.stop)

        # Patch the orchestrator to test the adapter in isolation
        self.mock_orchestrator_patcher = patch(
            'tpu_inference.core.core_tpu._DisaggOrchestrator')
        self.mock_orchestrator = self.mock_orchestrator_patcher.start()
        self.addCleanup(self.mock_orchestrator_patcher.stop)

        # Patch vLLMEngineCore to avoid its complex initialization.
        self.mock_engine_core_patcher = patch(
            'tpu_inference.core.core_tpu.vLLMEngineCore')
        self.mock_vLLMEngineCore = self.mock_engine_core_patcher.start()
        self.addCleanup(self.mock_engine_core_patcher.stop)

        # Patch the ZMQ handshake to isolate the test.
        self.mock_handshake_patcher = patch(
            'tpu_inference.core.core_tpu.DisaggEngineCoreProc._perform_handshake'
        )
        self.mock_handshake = self.mock_handshake_patcher.start()
        self.mock_handshake.return_value.__enter__.return_value = MagicMock(
            outputs=["output_addr"], coordinator_output=None)
        self.addCleanup(self.mock_handshake_patcher.stop)

        # Patch threads to avoid them running in the background.
        def mock_thread_constructor(*args, **kwargs):
            mock_thread = MagicMock()

            def mock_start():
                # Check if this is the input thread by looking at target and args
                target = kwargs.get('target')
                thread_args = kwargs.get('args', ())

                # If this is the input thread (process_input_sockets), set the ready_event
                if (target and hasattr(target, '__name__')
                        and target.__name__ == 'process_input_sockets'):
                    assert len(
                        thread_args
                    ) == 4, "Expected 4 arguments for vllm process_input_sockets function"
                    ready_event = thread_args[
                        3]  # ready_event is the 4th argument
                    ready_event.set()

            mock_thread.start = mock_start
            mock_thread.is_alive.return_value = True
            return mock_thread

        self.thread_patcher = patch("threading.Thread",
                                    side_effect=mock_thread_constructor)
        self.mock_thread = self.thread_patcher.start()
        self.addCleanup(self.thread_patcher.stop)

        # Mock jax.devices
        self.mock_jax_devices_patcher = patch('jax.devices',
                                              return_value=[MagicMock()] * 8)
        self.mock_jax_devices = self.mock_jax_devices_patcher.start()
        self.addCleanup(self.mock_jax_devices_patcher.stop)

        # VLLM Config
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.parallel_config = MagicMock(spec=ParallelConfig)
        self.mock_vllm_config.device_config = MagicMock()
        self.mock_vllm_config.cache_config = MagicMock()
        self.mock_vllm_config.cache_config.prefix_caching_hash_algo = "builtin"
        self.mock_vllm_config.cache_config.block_size = 5
        self.mock_vllm_config.__post_init__ = MagicMock()

    def test_initialization(self):
        """Tests that the adapter initializes the orchestrator correctly."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        self.mock_orchestrator.assert_called_once()
        args, kwargs = self.mock_orchestrator.call_args
        self.assertIsInstance(kwargs['config'], VllmConfig)
        self.assertEqual(kwargs['config'], self.mock_vllm_config)
        self.assertEqual(kwargs['output_queue'], proc.output_queue)
        self.assertEqual(len(kwargs['prefill_engines']), 1)
        self.assertEqual(len(kwargs['decode_engines']), 1)
        self.assertEqual(kwargs['prefill_slice_sizes'], (4, ))
        self.assertEqual(kwargs['decode_slice_sizes'], (2, ))

    def test_add_request(self):
        """Tests that the adapter correctly delegates add_request to the orchestrator."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        mock_request = MagicMock(spec=EngineCoreRequest)
        mock_request.request_id = "test_req"
        mock_request.mm_hashes = None
        mock_request.mm_kwargs = []
        mock_request.use_structured_output = False
        mock_request.pooling_params = None
        mock_request.sampling_params.structured_outputs = None
        mock_request.block_hashes = []

        mock_engine_request, _ = proc.preprocess_add_request(mock_request)

        proc.add_request(mock_engine_request)

        self.mock_orchestrator.return_value.add_request.assert_called_once()
        # Get the argument passed to add_request
        passed_request = self.mock_orchestrator.return_value.add_request.call_args[
            0][0]

        # Assert it's the correct type (the Request directly)
        self.assertIsInstance(passed_request, Request)
        self.assertEqual(passed_request.request_id, "test_req")

    def test_shutdown(self):
        """Tests that the adapter correctly delegates shutdown to the orchestrator."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        proc.shutdown()

        self.mock_orchestrator.return_value.shutdown.assert_called_once()

    def test_handle_client_request_add(self):
        """Tests that the adapter correctly handles an ADD request."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )
        mock_request = MagicMock(spec=EngineCoreRequest)
        mock_request.request_id = "test_req"
        mock_request.mm_hashes = None
        mock_request.mm_kwargs = []
        mock_request.use_structured_output = False
        mock_request.pooling_params = None
        mock_request.sampling_params.structured_outputs = None
        mock_request.block_hashes = []
        mock_request = proc.preprocess_add_request(mock_request)

        proc._handle_client_request(EngineCoreRequestType.ADD, mock_request)

        self.mock_orchestrator.return_value.add_request.assert_called_once()

    def test_handle_client_request_abort(self):
        """Tests that the adapter correctly handles an ABORT request."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )

        # This is currently a no-op, so we just check that it doesn't crash
        proc._handle_client_request(EngineCoreRequestType.ABORT, "test_req")

    def test_handle_client_request_utility(self):
        """Tests that the adapter correctly handles a UTILITY request."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            local_client=True,
            handshake_address="dummy_addr",
            executor_class=MagicMock(spec=Executor),
            log_stats=False,
        )
        # Mock a method on the prefill engine instance
        proc._prefill_engines = [MagicMock()]
        proc._prefill_engines[0].list_loras.return_value = {1, 2, 3}

        utility_request = (0, "call-id-1", "list_loras", ())
        proc._handle_client_request(EngineCoreRequestType.UTILITY,
                                    utility_request)

        proc._prefill_engines[0].list_loras.assert_called_once()
        self.assertTrue(proc.output_queue.qsize() > 0)


class TestDisaggOrchestrator(unittest.TestCase):

    def setUp(self):
        self.mock_config = MagicMock(spec=VllmConfig)
        self.mock_config.scheduler_config = MagicMock()
        self.mock_config.scheduler_config.max_num_seqs = 16
        self.mock_config.cache_config = MagicMock()
        self.mock_config.cache_config.block_size = 5

        self.mock_output_queue = MagicMock()
        self.mock_prefill_engine = MagicMock()
        self.mock_decode_engine = MagicMock()

        # The orchestrator accesses the scheduler on the engine.
        self.mock_prefill_engine.scheduler = MagicMock()
        self.mock_decode_engine.scheduler = MagicMock()

        # The orchestrator accesses the model_executor on the engine.
        self.mock_prefill_engine.model_executor = MagicMock()
        self.mock_decode_engine.model_executor = MagicMock()

        # Patch threads to avoid them running in the background.
        self.jet_thread_patcher = patch(
            "tpu_inference.core.core_tpu.JetThread", MagicMock)
        self.mock_jet_thread = self.jet_thread_patcher.start()
        self.addCleanup(self.jet_thread_patcher.stop)

    def test_initialization(self):
        """Tests that the orchestrator initializes correctly."""
        orchestrator = _DisaggOrchestrator(
            config=self.mock_config,
            output_queue=self.mock_output_queue,
            prefill_engines=[self.mock_prefill_engine],
            decode_engines=[self.mock_decode_engine],
            prefill_slice_sizes=(4, ),
            decode_slice_sizes=(2, ),
        )

        self.assertEqual(orchestrator._config, self.mock_config)
        self.assertEqual(orchestrator._output_queue, self.mock_output_queue)
        self.assertEqual(len(orchestrator._prefill_engines), 1)
        self.assertEqual(len(orchestrator._decode_engines), 1)
        self.assertEqual(len(orchestrator._all_threads),
                         3)  # 1 prefill, 1 transfer, 1 decode

    def test_add_request(self):
        """Tests that a new request is added to the prefill engine."""
        orchestrator = _DisaggOrchestrator(
            config=self.mock_config,
            output_queue=self.mock_output_queue,
            prefill_engines=[self.mock_prefill_engine],
            decode_engines=[self.mock_decode_engine],
            prefill_slice_sizes=(4, ),
            decode_slice_sizes=(2, ),
        )
        mock_request = MagicMock()
        mock_request.request_id = "test_req"

        orchestrator.add_request(mock_request)

        self.assertIn("test_req", orchestrator._requests)
        self.mock_prefill_engine.scheduler.add_request.assert_called_once_with(
            mock_request)

    def test_prefill_logic(self):
        """Tests the prefill logic of the orchestrator."""
        orchestrator = _DisaggOrchestrator(
            config=self.mock_config,
            output_queue=self.mock_output_queue,
            prefill_engines=[self.mock_prefill_engine],
            decode_engines=[self.mock_decode_engine],
            prefill_slice_sizes=(4, ),
            decode_slice_sizes=(2, ),
        )
        orchestrator.live = True

        # Mock scheduler output
        mock_scheduler_output = MagicMock()
        mock_scheduler_output.total_num_scheduled_tokens = 1
        self.mock_prefill_engine.scheduler.schedule.return_value = mock_scheduler_output

        # Mock model output
        mock_model_output = MagicMock()
        mock_model_output.req_id_to_index = {"test_req": 0}
        mock_model_output.sampled_token_ids = [[1]]
        self.mock_prefill_engine.model_executor.execute_model.return_value = mock_model_output

        # Mock request
        mock_request = MagicMock()
        orchestrator._requests["test_req"] = mock_request

        # Mock the side effect of update_from_output to stop the loop
        def stop_loop(*args, **kwargs):
            orchestrator.live = False
            return {}

        self.mock_prefill_engine.scheduler.update_from_output.side_effect = stop_loop

        orchestrator._prefill(0)

        self.mock_prefill_engine.model_executor.execute_model.assert_called_once(
        )
        self.assertTrue(orchestrator._transfer_backlogs[0].qsize() > 0)

    def test_transfer_logic(self):
        """Tests the transfer logic of the orchestrator."""
        orchestrator = _DisaggOrchestrator(
            config=self.mock_config,
            output_queue=self.mock_output_queue,
            prefill_engines=[self.mock_prefill_engine],
            decode_engines=[self.mock_decode_engine],
            prefill_slice_sizes=(4, ),
            decode_slice_sizes=(2, ),
        )
        orchestrator.live = True

        # Mock kv cache map
        mock_kv_cache_map = {"test_req": ([MagicMock()], [])}
        orchestrator._transfer_backlogs[0].put(mock_kv_cache_map)
        orchestrator._transfer_backlogs[0].put(
            None)  # Sentinel to stop the loop

        orchestrator._transfer(0)

        self.mock_decode_engine.model_executor.driver_worker.model_runner.transfer_kv_cache.assert_called_once(
        )
        self.assertTrue(orchestrator._decode_backlogs[0].qsize() > 0)

    def test_decode_logic(self):
        """Tests the decode logic of the orchestrator."""
        orchestrator = _DisaggOrchestrator(
            config=self.mock_config,
            output_queue=self.mock_output_queue,
            prefill_engines=[self.mock_prefill_engine],
            decode_engines=[self.mock_decode_engine],
            prefill_slice_sizes=(4, ),
            decode_slice_sizes=(2, ),
        )
        orchestrator.live = True

        # Mock prefill output
        mock_prefill_output = {
            "req_id": "test_req",
            "cache": [MagicMock()],
            "block_hashes": []
        }
        orchestrator._decode_backlogs[0].put(mock_prefill_output)
        orchestrator._decode_backlogs[0].put(None)  # Sentinel to stop the loop

        # Mock request
        mock_request = MagicMock()
        mock_request.num_computed_tokens = 10
        orchestrator._requests["test_req"] = mock_request

        # Mock scheduler and model runner states for the loop condition
        self.mock_decode_engine.scheduler.has_requests.return_value = False
        self.mock_decode_engine.scheduler.get_request_counts.return_value = (0,
                                                                             0)
        self.mock_decode_engine.model_executor.driver_worker.model_runner.input_batch.num_reqs = 0
        self.mock_decode_engine.scheduler.kv_cache_manager.get_block_ids.return_value = (
            [20, 21], )

        # Mock scheduler output
        mock_scheduler_output = MagicMock()
        mock_scheduler_output.total_num_scheduled_tokens = 1
        self.mock_decode_engine.scheduler.schedule.return_value = mock_scheduler_output

        # Mock model output
        mock_model_output = MagicMock()
        self.mock_decode_engine.model_executor.execute_model.return_value = mock_model_output

        # Mock the side effect of update_from_output to stop the loop
        def stop_loop(*args, **kwargs):
            orchestrator.live = False
            return {"test_req": MagicMock()}

        self.mock_decode_engine.scheduler.update_from_output.side_effect = stop_loop

        orchestrator._decode(0)

        self.mock_decode_engine.model_executor.execute_model.assert_called_once(
        )
        self.mock_output_queue.put_nowait.assert_called_once()

    def test_shutdown(self):
        """Tests that the orchestrator correctly shuts down its engines."""
        orchestrator = _DisaggOrchestrator(
            config=self.mock_config,
            output_queue=self.mock_output_queue,
            prefill_engines=[self.mock_prefill_engine],
            decode_engines=[self.mock_decode_engine],
            prefill_slice_sizes=(4, ),
            decode_slice_sizes=(2, ),
        )

        orchestrator.shutdown()

        self.mock_prefill_engine.shutdown.assert_called_once()
        self.mock_decode_engine.shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
