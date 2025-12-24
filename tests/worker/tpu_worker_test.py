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

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import ModelConfig
from vllm.lora.request import LoRARequest
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import DraftTokenIds

# The class we are testing
from tpu_inference.worker.tpu_worker import TPUWorker


@pytest.fixture
def mock_vllm_config():
    """
    Provides a mock VllmConfig object for tests.
    This version builds the mock explicitly to avoid spec-related AttributeErrors.
    """
    # Create mocks for the nested config objects first
    mock_cache_conf = MagicMock()
    mock_cache_conf.gpu_memory_utilization = 0.9
    mock_cache_conf.num_gpu_blocks = 0
    mock_cache_conf.num_cpu_blocks = 0

    mock_parallel_conf = MagicMock()
    mock_parallel_conf.tensor_parallel_size = 2
    mock_parallel_conf.data_parallel_size = 1
    mock_parallel_conf.pipeline_parallel_size = 1
    mock_parallel_conf.nnodes = 1
    mock_parallel_conf.nnodes_within_dp = 1

    mock_additional_config = {}

    # Create the main config mock and attach the others without a top-level spec
    config = MagicMock()
    config.model_config = ModelConfig(model="Qwen/Qwen3-0.6B")
    config.cache_config = mock_cache_conf
    config.parallel_config = mock_parallel_conf
    config.additional_config = mock_additional_config

    config.sharding_config = MagicMock()
    config.sharding_config.total_devices = 2

    return config


class TestTPUWorker:
    """Test suite for the TPUWorker class."""

    #
    # --- Initialization Tests ---
    #

    def test_init_success(self, mock_vllm_config):
        """Tests successful initialization of TPUWorker."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True,
                           devices=['tpu:0'])
        assert worker.vllm_config == mock_vllm_config
        assert worker.rank == 0
        assert worker.local_rank == 0
        assert worker.is_driver_worker
        assert worker.profile_dir is None
        assert worker.devices == ['tpu:0']

    @patch('tpu_inference.worker.tpu_worker.vllm_envs')
    def test_init_with_profiler_on_rank_zero(self, mock_envs,
                                             mock_vllm_config):
        """Tests that the profiler directory is set correctly on rank 0."""
        mock_envs.VLLM_TORCH_PROFILER_DIR = "/tmp/profiles"
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method")
        assert worker.profile_dir == "/tmp/profiles"

    @patch('tpu_inference.worker.tpu_worker.vllm_envs')
    def test_init_with_profiler_on_other_ranks(self, mock_envs,
                                               mock_vllm_config):
        """Tests that the profiler directory is NOT set on non-rank 0 workers."""
        mock_envs.VLLM_TORCH_PROFILER_DIR = "/tmp/profiles"
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=1,
                           rank=1,
                           distributed_init_method="test_method")
        assert worker.profile_dir is None

    #
    # --- Device and Cache Initialization Tests ---
    #

    def test_initialize_cache(self, mock_vllm_config):
        """Tests setting the number of GPU and CPU cache blocks."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method")
        worker.initialize_cache(num_gpu_blocks=2048, num_cpu_blocks=1024)
        assert worker.cache_config.num_gpu_blocks == 2048
        assert worker.cache_config.num_cpu_blocks == 1024

    @patch('tpu_inference.worker.tpu_worker.TPUModelRunner')
    @patch('tpu_inference.worker.tpu_worker.utils')
    @patch('tpu_inference.worker.tpu_worker.jax')
    @patch('tpu_inference.worker.tpu_worker.ensure_kv_transfer_initialized')
    def test_init_device_with_provided_devices(
            self, mock_ensure_kv_transfer_initialized, mock_jax, mock_utils,
            mock_runner_cls, mock_vllm_config):
        """Tests init_device when devices are provided during construction."""
        mock_devices = ['tpu:0', 'tpu:1']
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           devices=mock_devices)

        worker.init_device()

        expected_rank = 0
        expected_is_first_rank = True
        expected_is_last_rank = True
        mock_runner_cls.assert_called_once_with(mock_vllm_config, mock_devices,
                                                expected_rank,
                                                expected_is_first_rank,
                                                expected_is_last_rank)
        assert isinstance(worker.model_runner, MagicMock)

    @patch('tpu_inference.worker.tpu_worker.TPUModelRunner')
    @patch('tpu_inference.worker.tpu_worker.utils')
    @patch('tpu_inference.worker.tpu_worker.jax')
    @patch('tpu_inference.worker.tpu_worker.ensure_kv_transfer_initialized')
    def test_init_device_autodetects_devices(
            self, mock_ensure_kv_transfer_initialized, mock_jax, mock_utils,
            mock_runner_cls, mock_vllm_config):
        """Tests init_device when devices are auto-detected via JAX."""
        worker = TPUWorker(
            vllm_config=mock_vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="test_method",
            devices=[]  # No devices provided, should trigger auto-detection
        )
        mock_jax.device_count.return_value = 4
        mock_jax.devices.return_value = ['tpu:0', 'tpu:1', 'tpu:2', 'tpu:3']

        worker.init_device()

        expected_devices = ['tpu:0', 'tpu:1']  # Sliced by tensor_parallel_size
        assert worker.devices == expected_devices
        expected_rank = 0
        expected_is_first_rank = True
        expected_is_last_rank = True
        mock_runner_cls.assert_called_once_with(mock_vllm_config,
                                                expected_devices,
                                                expected_rank,
                                                expected_is_first_rank,
                                                expected_is_last_rank)

    @patch('tpu_inference.worker.tpu_worker.utils')
    def test_determine_available_memory(self, mock_utils, mock_vllm_config):
        """Tests the available HBM memory calculation."""
        # Setup mock return for hbm_usage_bytes: [(used_bytes, limit_bytes), ...]
        mock_utils.hbm_usage_bytes.return_value = [
            (100 * 1024**3, 1000 * 1024**3), (200 * 1024**3, 1000 * 1024**3)
        ]
        mock_devices = ['tpu:0', 'tpu:1']
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           devices=mock_devices)

        available_mem = worker.determine_available_memory()

        mock_utils.hbm_usage_bytes.assert_called_once_with(mock_devices)
        # Total limit: 1000 + 1000 = 2000 GiB
        # Total cap: 2000 * 0.9 = 1800 GiB
        # Total used: 100 + 200 = 300 GiB
        # Total free = 1800 - 300 = 1500 GiB
        expected_mem = 1500 * 1024**3
        assert available_mem == expected_mem

    #
    # --- Core Logic Tests ---
    #

    @patch('tpu_inference.worker.tpu_worker.TPUModelRunner')
    def test_execute_model(self, mock_runner_cls, mock_vllm_config):
        """Tests that the driver worker executes the model and returns the concrete vLLM output."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test",
                           is_driver_worker=True)
        worker.model_runner = mock_runner_cls.return_value  # Assign mocked runner instance
        mock_scheduler_input = MagicMock()

        # The model runner returns a concrete vllm output
        mock_model_output = "concrete_model_output"
        worker.model_runner.execute_model.return_value = mock_model_output

        result = worker.execute_model(mock_scheduler_input)

        # Assert the runner was called with the scheduler output directly
        worker.model_runner.execute_model.assert_called_once_with(
            mock_scheduler_input, None)
        # Assert the final result is the concrete model output
        assert result == mock_model_output

    @patch('tpu_inference.worker.tpu_worker.TPUModelRunner')
    def test_execute_model_non_driver_returns_none(self, mock_runner_cls,
                                                   mock_vllm_config):
        """Tests that a non-driver worker executes the model but returns None."""
        worker = TPUWorker(
            vllm_config=mock_vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="test",
            is_driver_worker=False  # Not a driver
        )
        worker.model_runner = mock_runner_cls.return_value
        mock_scheduler_input = MagicMock()

        result = worker.execute_model(mock_scheduler_input)

        assert result is None

    def test_take_draft_token_ids(self, mock_vllm_config):
        """Tests that take_draft_token_ids correctly passes through from the runner."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()

        # Case 1: Runner returns a DraftTokenIds object
        mock_draft_tokens = DraftTokenIds(req_ids=["req1"],
                                          draft_token_ids=[[1, 2]])
        worker.model_runner.take_draft_token_ids.return_value = mock_draft_tokens

        result = worker.take_draft_token_ids()
        worker.model_runner.take_draft_token_ids.assert_called_once()
        assert result == mock_draft_tokens

    def test_add_lora_not_implemented(self, mock_vllm_config):
        """Tests that add_lora raises NotImplementedError."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        mock_lora_request = MagicMock()

        with pytest.raises(
                NotImplementedError,
                match="LoRA is not supported by the JAX worker yet."):
            worker.add_lora(mock_lora_request)

    def test_add_lora_not_implemented_lora_request(self, mock_vllm_config):
        """Tests that add_lora raises NotImplementedError."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        mock_lora_request = MagicMock(spec=LoRARequest)

        with pytest.raises(
                NotImplementedError,
                match="LoRA is not supported by the JAX worker yet."):
            worker.add_lora(mock_lora_request)

    #
    # --- Profiling and Health Check Tests ---
    #

    @patch('tpu_inference.worker.tpu_worker.jax')
    @patch.dict('os.environ', {"PYTHON_TRACER_LEVEL": "1"}, clear=True)
    def test_profile_start(self, mock_jax, mock_vllm_config):
        """Tests starting the JAX profiler."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.profile_dir = "/tmp/profile_dir"

        worker.profile(is_start=True)

        mock_jax.profiler.ProfileOptions.assert_called_once()
        mock_jax.profiler.start_trace.assert_called_once()
        args, kwargs = mock_jax.profiler.start_trace.call_args
        assert args[0] == "/tmp/profile_dir"
        # Verify options from env var were used
        assert kwargs['profiler_options'].python_tracer_level == 1

    @patch('tpu_inference.worker.tpu_worker.jax')
    def test_profile_stop(self, mock_jax, mock_vllm_config):
        """Tests stopping the JAX profiler."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.profile(is_start=False)
        mock_jax.profiler.stop_trace.assert_called_once()

    def test_check_health(self, mock_vllm_config):
        """Tests that check_health runs without error."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        try:
            worker.check_health()
        except Exception as e:
            pytest.fail(
                f"TPUWorker.check_health() raised an unexpected exception: {e}"
            )

    #
    # --- Pass-through Method Tests ---
    #

    @pytest.mark.parametrize(
        "worker_method_name, runner_method_name, method_args", [
            ("load_model", "load_model", []),
            ("get_model", "get_model", []),
            ("get_kv_cache_spec", "get_kv_cache_spec", []),
        ])
    def test_runner_passthrough_methods(self, worker_method_name,
                                        runner_method_name, method_args,
                                        mock_vllm_config):
        """Tests methods that are simple pass-throughs to the TPUModelRunner."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()

        # Call the worker method and assert the underlying runner method was called
        getattr(worker, worker_method_name)(*method_args)
        mock_runner_method = getattr(worker.model_runner, runner_method_name)
        mock_runner_method.assert_called_once_with(*method_args)

    def test_initialize_from_config(self, mock_vllm_config):
        """Tests the special case pass-through for initialize_from_config."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()
        worker.topology_order_id = 0
        mock_input_config = MagicMock()

        worker.initialize_from_config(mock_input_config)

        worker.model_runner.initialize_kv_cache.assert_called_once_with(
            mock_input_config, 0)

    def test_initialize_from_config_kv_cache_config(self, mock_vllm_config):
        """Tests the special case pass-through for initialize_from_config."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()
        worker.topology_order_id = 0
        mock_input_config = MagicMock(spec=KVCacheConfig)

        worker.initialize_from_config(mock_input_config)

        worker.model_runner.initialize_kv_cache.assert_called_once_with(
            mock_input_config, 0)

    def test_compile_or_warm_up_model(self, mock_vllm_config):
        """Tests the special case pass-through for model compilation/warmup."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()

        worker.compile_or_warm_up_model()

        # This method calls two different runner methods
        worker.model_runner.capture_model.assert_called_once()
        worker.model_runner._init_random.assert_called_once()

    def test_get_supported_tasks(self, mock_vllm_config):
        """Test get_supported_tasks passthrough to model runner."""
        worker = TPUWorker(vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()
        worker.model_runner.get_supported_tasks.return_value = ("generate", )

        _ = worker.get_supported_tasks()

        worker.model_runner.get_supported_tasks.assert_called_once()
