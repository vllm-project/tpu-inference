# SPDX-License-Identifier: Apache-2.0
import unittest
import time
from unittest.mock import MagicMock, call, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request

# The class we are testing
from tpu_commons.core.core_tpu import EngineCore


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
                "layers.0": FullAttentionSpec(
                    block_size=64,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    use_mla=False,
                    sliding_window=None,
                ),
                "layers.1": FullAttentionSpec(
                    block_size=64,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    use_mla=False,
                    sliding_window=None,
                ),
                "layers.2": FullAttentionSpec(
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
        self.patch_jax_engine = patch(
            "tpu_commons.core.core_tpu.JaxEngine", self.mock_engine_class)
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
            [call(self._devices[:4]), call(self._devices[4:8])]
        )

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
            self.fail(
                f"abort_requests raised an unexpected exception: {e}")

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
        method, args, kwargs = "test_method", (
            1, "a"), {
                "key": "value"
            }
        self.engine_core.collective_rpc(method, args=args, kwargs=kwargs)
        self.mock_prefill_executor.collective_rpc.assert_called_once_with(
            method, None, args, kwargs)

    def test_save_tensorized_model(self):
        """Test saving a tensorized model passthrough."""
        config = {"config_key": "config_value"}
        self.engine_core.save_tensorized_model(config)
        self.mock_prefill_executor.save_tensorized_model.assert_called_once_with(
            tensorizer_config=config)


if __name__ == '__main__':
    unittest.main()
