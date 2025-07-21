import os
from unittest.mock import MagicMock, patch

import pytest
import torch
# Dependencies that will be mocked
from vllm.v1.outputs import ModelRunnerOutput

# Import the abstract classes and interfaces for mocking
from tpu_commons.di.abstracts import (AbstractKVCacheConfig,
                                      AbstractSchedulerOutput)
from tpu_commons.di.interfaces import HostInterface
# The class we are testing
from tpu_commons.worker.tpu_worker_torchax import TPUWorker


@pytest.fixture(autouse=True)
def set_tpu_backend_env():
    """Set TPU_BACKEND_TYPE=torchax for all tests in this module."""
    with patch.dict(os.environ, {"TPU_BACKEND_TYPE": "torchax"}):
        yield


@pytest.fixture
def mock_host_interface():
    """Provides a mock HostInterface for tests."""
    return MagicMock(spec=HostInterface)


@pytest.fixture
def mock_vllm_config():
    """
    Provides a mock VllmConfig object for tests.
    This version builds the mock explicitly to avoid spec-related AttributeErrors.
    """
    # Create mocks for the nested config objects first
    mock_model_conf = MagicMock()
    mock_model_conf.trust_remote_code = False
    mock_model_conf.dtype = torch.bfloat16

    mock_cache_conf = MagicMock()
    mock_cache_conf.gpu_memory_utilization = 0.9
    mock_cache_conf.num_gpu_blocks = 0
    mock_cache_conf.num_cpu_blocks = 0
    mock_cache_conf.cache_dtype = "auto"

    mock_parallel_conf = MagicMock()
    mock_parallel_conf.tensor_parallel_size = 2

    # Create the main config mock and attach the others without a top-level spec
    config = MagicMock()
    config.model_config = mock_model_conf
    config.cache_config = mock_cache_conf
    config.parallel_config = mock_parallel_conf

    # The TPUWorker also accesses cache_config directly from the top-level config
    # in its `initialize_cache` method.
    config.cache_config = mock_cache_conf

    return config


class TestTPUWorker:
    """Test suite for the TPUWorker class."""

    #
    # --- Initialization Tests ---
    #

    def test_init_success(self, mock_host_interface, mock_vllm_config):
        """Tests successful initialization of TPUWorker."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True)
        assert worker.host_interface == mock_host_interface
        assert worker.vllm_config == mock_vllm_config
        assert worker.rank == 0
        assert worker.local_rank == 0
        assert worker.is_driver_worker
        assert worker.profile_dir is None

    def test_init_success_with_bfloat16_cache_dtype(self, mock_host_interface,
                                                    mock_vllm_config):
        """Tests successful initialization of TPUWorker with bfloat16 cache dtype."""
        # Override the cache_dtype to bfloat16
        mock_vllm_config.cache_config.cache_dtype = "bfloat16"

        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True)
        assert worker.vllm_config == mock_vllm_config
        assert worker.rank == 0
        assert worker.local_rank == 0
        assert worker.is_driver_worker
        assert worker.profile_dir is None
        # Verify the cache dtype is as expected
        assert worker.vllm_config.cache_config.cache_dtype == "bfloat16"

    def test_init_multi_host_not_implemented(self, mock_host_interface,
                                             mock_vllm_config):
        """Tests that multi-host (rank != local_rank) raises NotImplementedError."""
        with pytest.raises(NotImplementedError,
                           match="Multi host serving is not supported yet."):
            TPUWorker(
                host_interface=mock_host_interface,
                vllm_config=mock_vllm_config,
                local_rank=0,
                rank=1,  # Different rank from local_rank
                distributed_init_method="test_method")

    @patch('tpu_commons.worker.tpu_worker_torchax.envs')
    def test_init_with_spmd_enabled(self, mock_envs, mock_host_interface,
                                    mock_vllm_config):
        """Tests that use_spmd is set correctly when VLLM_XLA_USE_SPMD is True."""
        mock_envs.VLLM_XLA_USE_SPMD = True

        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True)

        assert worker.use_spmd is True
        # Verify that original_parallel_config was stored
        assert worker.original_parallel_config is not None
        # Verify that parallel_config was modified for SPMD mode
        assert worker.parallel_config.tensor_parallel_size == 1
        assert worker.parallel_config.pipeline_parallel_size == 1
        assert worker.parallel_config.world_size == 1

    @patch('tpu_commons.worker.tpu_worker_torchax.envs')
    def test_init_with_spmd_disabled(self, mock_envs, mock_host_interface,
                                     mock_vllm_config):
        """Tests that use_spmd is set correctly when VLLM_XLA_USE_SPMD is False."""
        mock_envs.VLLM_XLA_USE_SPMD = False

        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True)

        assert worker.use_spmd is False
        # Verify that original_parallel_config is None when SPMD is disabled
        assert worker.original_parallel_config is None
        # Verify that parallel_config was not modified
        assert worker.parallel_config.tensor_parallel_size == 2  # From mock fixture
        assert worker.parallel_config.pipeline_parallel_size != 1  # Should remain unchanged
        assert worker.parallel_config.world_size != 1  # Should remain unchanged

    @patch('tpu_commons.worker.tpu_worker_torchax.envs')
    def test_init_with_profiler_on_rank_zero(self, mock_envs,
                                             mock_host_interface,
                                             mock_vllm_config):
        """Tests that the profiler directory is set correctly on rank 0."""
        mock_envs.VLLM_TORCH_PROFILER_DIR = "/tmp/profiles"
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method")
        assert worker.profile_dir == "/tmp/profiles"

    @patch('tpu_commons.worker.tpu_worker_torchax.envs')
    def test_init_with_profiler_on_other_ranks(self, mock_envs,
                                               mock_host_interface,
                                               mock_vllm_config):
        """Tests that the profiler directory is NOT set on non-rank 0 workers."""
        mock_envs.VLLM_TORCH_PROFILER_DIR = "/tmp/profiles"
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=1,
                           rank=1,
                           distributed_init_method="test_method")
        assert worker.profile_dir is None

    #
    # --- Device and Cache Initialization Tests ---
    #

    def test_initialize_cache(self, mock_host_interface, mock_vllm_config):
        """Tests setting the number of GPU and CPU cache blocks."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method")
        worker.initialize_cache(num_gpu_blocks=2048, num_cpu_blocks=1024)
        assert worker.cache_config.num_gpu_blocks == 2048
        assert worker.cache_config.num_cpu_blocks == 1024

    @patch('tpu_commons.worker.tpu_worker_torchax.TPUModelRunner')
    @patch('tpu_commons.worker.tpu_worker_torchax.report_usage_stats')
    @patch('tpu_commons.worker.tpu_worker_torchax.jax')
    @patch('tpu_commons.worker.tpu_worker_torchax.torch')
    @patch('tpu_commons.worker.tpu_worker_torchax.os')
    @patch('tpu_commons.worker.tpu_worker_torchax.envs')
    @pytest.mark.parametrize(
        "rank,should_report_usage",
        [
            (0, True),  # rank 0 should report usage stats
            (1, False),  # rank 1 should not report usage stats
        ])
    def test_init_device(self, mock_envs, mock_os, mock_torch, mock_jax,
                         mock_report_usage_stats, mock_runner_cls,
                         mock_host_interface, mock_vllm_config, rank,
                         should_report_usage):
        """Tests init_device method functionality with different ranks."""
        # Setup mocks
        mock_jax.process_index.return_value = rank
        mock_device = torch.device("jax:0")
        mock_torch.device.return_value = mock_device
        mock_runner_instance = MagicMock()
        mock_runner_cls.return_value = mock_runner_instance

        # Create worker
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True)

        worker._init_tpu_worker_distributed_environment = MagicMock()

        # Call init_device
        worker.init_device()

        # Verify environment variables are set
        mock_os.environ.get.assert_called_with("LIBTPU_INIT_ARGS", "")

        # Verify torch configuration
        mock_torch.set_grad_enabled.assert_called_once_with(False)
        mock_torch.set_default_dtype.assert_called_once_with(
            mock_vllm_config.model_config.dtype)

        # Verify device setup
        mock_torch.device.assert_called_once_with("jax:0")
        assert worker.device == mock_device
        assert worker.device_config.device == mock_device

        # Verify model runner initialization
        mock_runner_cls.assert_called_once_with(
            mock_vllm_config, mock_device, worker.original_parallel_config)
        assert worker.model_runner == mock_runner_instance

        # Verify usage stats reporting based on rank
        if should_report_usage:
            mock_report_usage_stats.assert_called_once_with(mock_vllm_config)
        else:
            mock_report_usage_stats.assert_not_called()

    @patch('tpu_commons.worker.tpu_worker_torchax.TPU_HEAD_SIZE_ALIGNMENT',
           128)
    @patch('tpu_commons.worker.tpu_worker_torchax.jax')
    @patch('tpu_commons.worker.tpu_worker_torchax.logger')
    @pytest.mark.parametrize(
        "head_size,expected_padded_size,should_warn",
        [
            (64, 128, True),  # 64 gets padded to 128, warning expected
            (256, 256, False),  # 256 is already aligned to 128, no warning
        ])
    def test_determine_available_memory_with_distributed_mocks(
            self, mock_logger, mock_jax, mock_host_interface, mock_vllm_config,
            head_size, expected_padded_size, should_warn):
        """Tests determine_available_memory with distributed environment mocked and different head sizes."""
        # Setup JAX mocks
        mock_device = MagicMock()
        mock_memory_stats = {
            "bytes_limit": 2000000000,  # 2GB
            "bytes_in_use": 200000000,  # 200MB
            "peak_bytes_in_use": 400000000,  # 400MB
        }
        mock_device.memory_stats.return_value = mock_memory_stats
        mock_jax.local_devices.return_value = [mock_device]
        mock_jax.process_index.return_value = 0

        # Setup model config with specific head size
        mock_vllm_config.cache_config.gpu_memory_utilization = 0.9
        mock_vllm_config.model_config.get_head_size.return_value = head_size

        # Create worker
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test_method",
                           is_driver_worker=True)

        worker._init_tpu_worker_distributed_environment = MagicMock()

        # Mock model_runner
        worker.model_runner = MagicMock()
        worker.model_runner.max_num_tokens = 1024

        # Call determine_available_memory
        result = worker.determine_available_memory()

        # Verify warning behavior
        if should_warn:
            mock_logger.warning_once.assert_called_once_with(
                "head size is padded to %d", expected_padded_size)
        else:
            mock_logger.warning_once.assert_not_called()

        # Calculate expected result
        usable_memory = 2000000000 * 0.9  # 1800000000
        base_kv_cache = usable_memory - 400000000  # 1400000000

        if head_size != expected_padded_size:
            # Adjust for padding
            adjusted_kv_cache = base_kv_cache * head_size // expected_padded_size
            assert result == int(adjusted_kv_cache)
        else:
            assert result == int(base_kv_cache)

    #
    # --- Core Logic Tests ---
    #

    @patch('tpu_commons.worker.tpu_worker_torchax.TPUModelRunner')
    def test_execute_model(self, mock_runner_cls, mock_host_interface,
                           mock_vllm_config):
        """Tests that the driver worker executes the model and returns the concrete vLLM output."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test",
                           is_driver_worker=True)
        worker.model_runner = mock_runner_cls.return_value  # Assign mocked runner instance
        mock_scheduler_input = MagicMock(spec=AbstractSchedulerOutput)

        # The adapter has the vllm object
        mock_scheduler_input.vllm_scheduler_output = "concrete_vllm_object"

        # The model runner returns a concrete vllm output
        mock_model_output = MagicMock(spec=ModelRunnerOutput)
        worker.model_runner.execute_model.return_value = mock_model_output

        result = worker.execute_model(mock_scheduler_input)

        # Assert the runner was called with the unwrapped concrete object
        worker.model_runner.execute_model.assert_called_once_with(
            "concrete_vllm_object")
        # Assert the final result is the concrete model output, not an adapter
        assert result == mock_model_output

    @patch('tpu_commons.worker.tpu_worker_torchax.TPUModelRunner')
    def test_execute_model_non_driver_returns_none(self, mock_runner_cls,
                                                   mock_host_interface,
                                                   mock_vllm_config):
        """Tests that a non-driver worker executes the model but returns None."""
        worker = TPUWorker(
            host_interface=mock_host_interface,
            vllm_config=mock_vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="test",
            is_driver_worker=False  # Not a driver
        )
        worker.model_runner = mock_runner_cls.return_value
        mock_scheduler_input = MagicMock(spec=AbstractSchedulerOutput)
        mock_scheduler_input.vllm_scheduler_output = "concrete_vllm_object"

        result = worker.execute_model(mock_scheduler_input)

        worker.model_runner.execute_model.assert_called_once_with(
            "concrete_vllm_object")
        assert result is None

    #
    # --- Profiling and Health Check Tests ---
    #

    @patch('tpu_commons.worker.tpu_worker_torchax.jax')
    def test_profile_start(self, mock_jax, mock_host_interface,
                           mock_vllm_config):
        """Tests starting the JAX profiler."""
        mock_jax.profiler = MagicMock()

        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.profile_dir = "/tmp/profile_dir"

        worker.profile(is_start=True)

        mock_jax.profiler.start_trace.assert_called_once()
        args, kwargs = mock_jax.profiler.start_trace.call_args
        assert args[0] == "/tmp/profile_dir"

    @patch('tpu_commons.worker.tpu_worker_torchax.jax')
    def test_profile_stop(self, mock_jax, mock_host_interface,
                          mock_vllm_config):
        """Tests stopping the JAX profiler."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.profile_dir = "/tmp/profile_dir"
        worker.profile(is_start=False)
        mock_jax.profiler.stop_trace.assert_called_once()

    def test_check_health(self, mock_host_interface, mock_vllm_config):
        """Tests that check_health runs without error."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
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
                                        mock_host_interface, mock_vllm_config):
        """Tests methods that are simple pass-throughs to the TPUModelRunner."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()

        # Call the worker method and assert the underlying runner method was called
        getattr(worker, worker_method_name)(*method_args)
        mock_runner_method = getattr(worker.model_runner, runner_method_name)
        mock_runner_method.assert_called_once_with(*method_args)

    @patch(
        'tpu_commons.worker.tpu_worker_torchax.adapt_kv_cache_config_if_needed'
    )
    def test_initialize_from_config(self, mock_adapter_fn, mock_host_interface,
                                    mock_vllm_config):
        """Tests the special case pass-through for initialize_from_config."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()
        mock_input_config = MagicMock(spec=AbstractKVCacheConfig)
        mock_adapter_fn.return_value = mock_input_config
        mock_input_config.vllm_kv_cache_config = "concrete_vllm_object"

        worker.initialize_from_config(mock_input_config)

        mock_adapter_fn.assert_called_once_with(mock_input_config)
        worker.model_runner.initialize_kv_cache.assert_called_once_with(
            "concrete_vllm_object")

    def test_compile_or_warm_up_model(self, mock_host_interface,
                                      mock_vllm_config):
        """Tests the special case pass-through for model compilation/warmup."""
        worker = TPUWorker(host_interface=mock_host_interface,
                           vllm_config=mock_vllm_config,
                           local_rank=0,
                           rank=0,
                           distributed_init_method="test")
        worker.model_runner = MagicMock()
        worker.model_config.enforce_eager = False
        worker.compile_or_warm_up_model()

        # This method calls two different runner methods
        worker.model_runner.capture_model.assert_called_once()
