# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, call, patch

import msgspec
import torch
import zmq
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreRequest, EngineCoreRequestType,
                            UtilityOutput)
from vllm.v1.engine.core import ModelRunnerOutput
from vllm.v1.engine.utils import EngineHandshakeMetadata, EngineZmqAddresses
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# The class we are testing
from tpu_commons.core.core_tpu import (DisaggEngineCoreProc, EngineCore,
                                       EngineCoreProc)


class EngineCoreTest(unittest.TestCase):

    def setUp(self):
        """Set up the test environment by mocking dependencies."""
        # Mock configurations
        self.mock_model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        )
        self.mock_cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
        )
        self.mock_scheduler_config = SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            max_model_len=1024,
        )
        self.mock_vllm_config = VllmConfig(
            scheduler_config=self.mock_scheduler_config,
            model_config=self.mock_model_config,
            cache_config=self.mock_cache_config,
        )

        self._devices = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        # Mock external dependencies
        self.patch_jax_devices = patch(
            "jax.devices",
            return_value=self._devices,
        )
        self.patch_get_prefill_slices = patch(
            "tpu_commons.core.disagg_utils.get_prefill_slices",
            return_value=(4, ))
        self.patch_get_decode_slices = patch(
            "tpu_commons.core.disagg_utils.get_decode_slices",
            return_value=(4, ))
        self.patch_unify_kv_cache_configs = patch(
            "tpu_commons.core.core_tpu.unify_kv_cache_configs")
        self.patch_mm_cache = patch(
            "tpu_commons.core.core_tpu.MirroredProcessingCache")
        self.patch_struct_output = patch(
            "tpu_commons.core.core_tpu.StructuredOutputManager")

        # Start general patches
        self.mock_jax_devices = self.patch_jax_devices.start()
        self.patch_get_prefill_slices.start()
        self.patch_get_decode_slices.start()
        self.patch_unify_kv_cache_configs.start()
        self.mock_mm_cache_instance = self.patch_mm_cache.start()
        self.patch_struct_output.start()

        # Mock class instantiations
        self.mock_executor_class = MagicMock()
        self.mock_executor_class.return_value.get_kv_cache_specs.return_value = [
            {
                "layers.0":
                FullAttentionSpec(
                    block_size=64,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    use_mla=False,
                    sliding_window=None,
                ),
                "layers.1":
                FullAttentionSpec(
                    block_size=64,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    use_mla=False,
                    sliding_window=None,
                ),
                "layers.2":
                FullAttentionSpec(
                    block_size=64,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    use_mla=False,
                    sliding_window=None,
                ),
            }
        ]
        self.mock_executor_class.return_value.max_concurrent_batches = 8
        self.mock_executor_class.return_value.determine_available_memory.return_value = [
            8 * 1024 * 1024 * 1024,
        ]

        self.mock_engine_class = MagicMock()
        self.mock_orchestrator_class = MagicMock()

        self.patch_disagg_executor = patch(
            "tpu_commons.core.disagg_executor.DisaggExecutor",
            self.mock_executor_class)
        self.patch_jax_engine = patch("tpu_commons.core.core_tpu.JaxEngine",
                                      self.mock_engine_class)
        self.patch_orchestrator = patch("tpu_commons.core.orchestrator.Driver",
                                        self.mock_orchestrator_class)

        self.patch_disagg_executor.start()
        self.patch_jax_engine.start()
        self.patch_orchestrator.start()

        self.engine_core = EngineCore(
            vllm_config=self.mock_vllm_config,
            executor_class=MagicMock(),  # The patch will intercept this
            log_stats=False,
        )
        self.mock_orchestrator = self.engine_core.orchestrator
        self.mock_prefill_executor = self.engine_core.prefill_executors[0]
        self.mock_decode_executor = self.engine_core.decode_executors[0]

    def tearDown(self):
        """Stop all patches."""
        patch.stopall()

    def test_init(self):
        """Test initialization of EngineCore."""
        # Check that executor class was instantiated twice (prefill and decode)
        self.assertEqual(self.mock_executor_class.call_count, 2)

        # Check that devices were sliced and passed correctly
        self.mock_prefill_executor.init_with_devices.assert_has_calls(
            [call(self._devices[:4]),
             call(self._devices[4:8])])

        # Check that JaxEngine was initialized for both
        self.assertEqual(self.mock_engine_class.call_count, 2)

        # Check that Orchestrator was initialized correctly
        self.mock_orchestrator_class.assert_called_once_with(
            vllm_config=self.mock_vllm_config,
            prefill_engines=self.engine_core.prefill_engines,
            generate_engines=self.engine_core.decode_engines,
        )

    def test_add_request(self):
        """Test adding a request."""
        mock_request = EngineCoreRequest(
            request_id="test_req",
            prompt_token_ids=[1, 2, 3],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(temperature=0.0),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None)

        self.engine_core.add_request(mock_request)

        # Verify that orchestrator method is called
        self.mock_orchestrator.place_request_on_prefill_queue.assert_called_once(
        )

        # Verify that the argument is a vllm.v1.request.Request
        arg = self.mock_orchestrator.place_request_on_prefill_queue.call_args[
            0][0]
        self.assertIsInstance(arg, Request)
        self.assertEqual(arg.request_id, "test_req")

    def test_abort_requests(self):
        """Test aborting requests (currently a no-op)."""
        try:
            self.engine_core.abort_requests(["test_req_1"])
        except Exception as e:
            self.fail(f"abort_requests raised an unexpected exception: {e}")

    def test_shutdown(self):
        """Test shutting down the engine."""
        self.engine_core.shutdown()
        self.mock_orchestrator.stop.assert_called_once()
        self.assertEqual(self.mock_prefill_executor.shutdown.call_count, 2)

    def test_reset_mm_cache(self):
        """Test resetting the multi-modal cache."""
        self.engine_core.reset_mm_cache()
        self.mock_mm_cache_instance.return_value.reset.assert_called_once()

    def test_profile(self):
        """Test profiling passthrough."""
        self.engine_core.profile(is_start=True)
        self.mock_prefill_executor.profile.assert_called_once_with(True)

        self.engine_core.profile(is_start=False)
        self.mock_prefill_executor.profile.assert_called_with(False)

    def test_sleep_wake_is_sleeping(self):
        """Test sleep, wake_up, and is_sleeping passthrough."""
        self.engine_core.sleep(level=2)
        self.mock_prefill_executor.sleep.assert_called_once_with(2)

        self.engine_core.wake_up(tags=["tag1"])
        self.mock_prefill_executor.wake_up.assert_called_once_with(
            tags=["tag1"])

    def test_execute_dummy_batch(self):
        """Test executing a dummy batch passthrough."""
        self.engine_core.execute_dummy_batch()
        self.mock_prefill_executor.collective_rpc.assert_called_once_with(
            "execute_dummy_batch")

    def test_lora_operations(self):
        """Test LoRA related operations passthrough."""
        lora_request = LoRARequest("lora1", 1, "path/to/lora")

        self.engine_core.add_lora(lora_request)
        self.mock_prefill_executor.add_lora.assert_called_once_with(
            lora_request)

        self.engine_core.remove_lora(1)
        self.mock_prefill_executor.remove_lora.assert_called_once_with(1)

        self.engine_core.list_loras()
        self.mock_prefill_executor.list_loras.assert_called_once()

        self.engine_core.pin_lora(1)
        self.mock_prefill_executor.pin_lora.assert_called_once_with(1)

    def test_save_sharded_state(self):
        """Test saving sharded state passthrough."""
        path, pattern, max_size = "/tmp/sharded", "*.bin", 1024
        self.engine_core.save_sharded_state(path, pattern, max_size)
        self.mock_prefill_executor.save_sharded_state.assert_called_once_with(
            path=path, pattern=pattern, max_size=max_size)

    def test_collective_rpc(self):
        """Test collective RPC passthrough."""
        method, args, kwargs = "test_method", (1, "a"), {"key": "value"}
        self.engine_core.collective_rpc(method, args=args, kwargs=kwargs)
        self.mock_prefill_executor.collective_rpc.assert_called_once_with(
            method, None, args, kwargs)

    def test_save_tensorized_model(self):
        """Test saving a tensorized model passthrough."""
        config = {"config_key": "config_value"}
        self.engine_core.save_tensorized_model(config)
        self.mock_prefill_executor.save_tensorized_model.assert_called_once_with(
            tensorizer_config=config)


class EngineCoreProcTest(unittest.TestCase):

    @staticmethod
    def run_engine_proc_with_mocks(vllm_config, handshake_address, ppid):
        """This static method runs in a separate process to test EngineCoreProc."""

        # Ensure the child process exits if the parent process dies.
        def ppid_checker():
            while True:
                try:
                    os.kill(ppid, 0)
                except OSError:
                    os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(1)

        threading.Thread(target=ppid_checker, daemon=True).start()

        # We patch EngineCore.__init__ to inject our mocks.
        with patch("tpu_commons.core.core_tpu.EngineCore.__init__"
                   ) as mock_engine_core_init:

            def mock_init(self, *args, **kwargs):
                """A mocked __init__ for EngineCore."""

            mock_engine_core_init.side_effect = mock_init

            # executor_class is not used due to the patch.
            engine_proc = EngineCoreProc(
                vllm_config=vllm_config,
                on_head_node=True,
                handshake_address=handshake_address,
                executor_class=MagicMock(),
                log_stats=False,
            )
            engine_proc.vllm_config = vllm_config
            # Mock the orchestrator to produce predictable output
            engine_proc.orchestrator = MagicMock()
            engine_proc.orchestrator._vllm_output_backlogs = queue.Queue()

            def place_req_on_queue(req):
                """Simulate orchestrator processing and producing output."""
                output = EngineCoreOutputs(outputs=[
                    EngineCoreOutput(request_id=req.request_id,
                                     new_token_ids=[101])
                ])
                # The orchestrator is expected to produce a dict mapping
                # client_index to outputs. We assume one client (index 0).
                engine_proc.orchestrator._vllm_output_backlogs.put({0: output})

            engine_proc.orchestrator.place_request_on_prefill_queue.side_effect = (
                place_req_on_queue)

            # Mock executors for utility methods
            engine_proc.prefill_executors = [MagicMock()]
            engine_proc.decode_executors = [MagicMock()]
            engine_proc.prefill_executors[0].is_sleeping = False
            engine_proc.run_busy_loop()

    def setUp(self):
        """Set up the test by spawning EngineCoreProc and performing a handshake."""
        # Use "spawn" to avoid issues with CUDA in forked processes.
        self.ctx = mp.get_context("spawn")

        self.vllm_config = self._create_vllm_config()

        # ZMQ addresses for communication
        self.handshake_address = "ipc://handshake_test"
        self.input_address = "ipc://input_test"
        self.output_address = "ipc://output_test"

        # Start the engine process
        self.engine_proc = self.ctx.Process(
            target=self.run_engine_proc_with_mocks,
            args=(self.vllm_config, self.handshake_address, os.getpid()),
        )
        self.engine_proc.start()

        # Set up ZMQ sockets for the test (client side)
        self.zmq_ctx = zmq.Context()

        # Perform handshake to ensure the engine is ready
        self.handshake_socket = self.zmq_ctx.socket(zmq.ROUTER)
        self.handshake_socket.bind(self.handshake_address)
        self.perform_handshake()

        # Set up client sockets for sending/receiving data
        self.input_socket = self.zmq_ctx.socket(zmq.ROUTER)
        self.input_socket.bind(self.input_address)
        # The engine connects with a DEALER, so we receive its identity first.
        self.engine_identity = self.input_socket.recv()

        self.output_socket = self.zmq_ctx.socket(zmq.PULL)
        self.output_socket.bind(self.output_address)

        # Setup serializers
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

    def tearDown(self):
        """Clean up the test environment."""
        if self.engine_proc.is_alive():
            os.kill(self.engine_proc.pid, signal.SIGTERM)
        self.engine_proc.join(timeout=5)
        if self.engine_proc.is_alive():
            self.engine_proc.kill()

        self.handshake_socket.close()
        self.input_socket.close()
        self.output_socket.close()
        self.zmq_ctx.term()

    def _create_vllm_config(self):
        """Creates a mock VllmConfig for testing."""
        mock_model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        )
        mock_cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
        )
        mock_scheduler_config = SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            max_model_len=1024,
        )
        return VllmConfig(
            scheduler_config=mock_scheduler_config,
            model_config=mock_model_config,
            cache_config=mock_cache_config,
        )

    def perform_handshake(self):
        """Performs the ZMQ handshake between the test and the engine process."""
        # Wait for HELLO from engine
        engine_identity, hello_msg_bytes = self.handshake_socket.recv_multipart(
        )
        # hello_msg_bytes = self.handshake_socket.recv()
        hello_msg = msgspec.msgpack.decode(hello_msg_bytes)
        self.assertEqual(hello_msg["status"], "HELLO")

        # Send INIT to engine
        addresses = EngineZmqAddresses(
            inputs=[self.input_address],
            outputs=[self.output_address],
        )

        def custom_dict_factory(data):
            result = {}
            for k, v in data:
                if isinstance(v, float) or isinstance(v, bool):
                    result[k] = int(v)
                elif isinstance(v, (int, str)):
                    result[k] = v
                else:
                    result[k] = str(v)  # Convert other types to string
            return result

        handshake_metadata = EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config=asdict(self.vllm_config.parallel_config,
                                   dict_factory=custom_dict_factory),
        )
        init_msg = msgspec.msgpack.encode(handshake_metadata)

        self.handshake_socket.send_multipart([engine_identity, init_msg])
        # self.handshake_socket.send(init_msg)

        # Wait for READY from engine
        _, ready_msg_bytes = self.handshake_socket.recv_multipart()
        # ready_msg_bytes = self.handshake_socket.recv()
        ready_msg = msgspec.msgpack.decode(ready_msg_bytes)
        self.assertEqual(ready_msg["status"], "READY")

    def test_add_request_and_receive_output(self):
        """Tests sending a request and receiving the corresponding output."""
        request = EngineCoreRequest(
            request_id="test_req",
            prompt_token_ids=[1, 2, 3],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(temperature=0.0),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None)

        # Send the request to the engine process
        msg = (EngineCoreRequestType.ADD.value, *self.encoder.encode(request))
        self.input_socket.send_multipart([self.engine_identity, *msg])

        output_frames = self.output_socket.recv_multipart(copy=False)
        outputs = self.decoder.decode(output_frames)

        # Assertions
        self.assertIsInstance(outputs, EngineCoreOutputs)
        self.assertEqual(len(outputs.outputs), 1)
        output = outputs.outputs[0]
        self.assertEqual(output.request_id, "test_req")
        self.assertEqual(output.new_token_ids, [101])

    def test_utility_method(self):
        """Tests calling a utility method on the engine."""
        # Prepare the utility request
        call_id = 123
        method_name = "is_sleeping"
        args = ()
        utility_request = (0, call_id, method_name, args)

        # Send the request
        msg = (EngineCoreRequestType.UTILITY.value,
               *self.encoder.encode(utility_request))
        self.input_socket.send_multipart([self.engine_identity, *msg])

        # Receive the output
        output_frames = self.output_socket.recv_multipart(copy=False)
        outputs = self.decoder.decode(output_frames)

        # Assertions
        self.assertIsNotNone(outputs.utility_output)
        utility_output = outputs.utility_output
        self.assertEqual(utility_output.call_id, call_id)
        # The mock is_sleeping returns False
        self.assertFalse(utility_output.result)
        self.assertIsNone(utility_output.failure_message)


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
        self.jet_thread_patcher = patch(
            "tpu_commons.core.core_tpu.orchestrator.JetThread", MagicMock)
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

        self.mock_vllm_config.scheduler_config.max_num_seqs = 16
        self.mock_vllm_config.device_config = MagicMock()

    def test_initialization(self):
        """Tests that DisaggEngineCoreProc initializes correctly."""
        proc = DisaggEngineCoreProc(
            vllm_config=self.mock_vllm_config,
            on_head_node=True,
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
            on_head_node=True,
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
            on_head_node=True,
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
            on_head_node=True,
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
            on_head_node=True,
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
            on_head_node=True,
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
            on_head_node=True,
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
