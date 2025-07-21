# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
import torch
import vllm.envs as envs
from vllm.config import CompilationLevel
from vllm.platforms.interface import _Backend
from vllm.sampling_params import SamplingParams, SamplingType

from tpu_commons.platforms.tpu_torchax import TpuPlatform


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    # Use mocks instead of real configs to avoid validation issues
    mock_config = Mock()

    # Mock model config
    mock_model_config = Mock()
    mock_model_config.dtype = torch.bfloat16
    mock_config.model_config = mock_model_config

    # Mock cache config
    mock_cache_config = Mock()
    mock_cache_config.block_size = None
    mock_config.cache_config = mock_cache_config

    # Mock parallel config
    mock_parallel_config = Mock()
    mock_parallel_config.worker_cls = "auto"
    mock_config.parallel_config = mock_parallel_config

    # Mock scheduler config
    mock_scheduler_config = Mock()
    mock_scheduler_config.is_multi_step = False
    mock_scheduler_config.is_multimodal_model = False
    mock_scheduler_config.disable_chunked_mm_input = False
    mock_config.scheduler_config = mock_scheduler_config

    # Mock compilation config
    mock_compilation_config = Mock()
    mock_compilation_config.level = CompilationLevel.NO_COMPILATION
    mock_compilation_config.backend = ""
    mock_config.compilation_config = mock_compilation_config

    # Mock speculative config
    mock_config.speculative_config = None

    return mock_config


class TestTpuPlatform:
    """Test class for TpuPlatform."""

    def test_platform_properties(self):
        """Test basic platform properties."""
        assert TpuPlatform.device_name == "tpu"
        assert TpuPlatform.device_type == "tpu"
        assert TpuPlatform.dispatch_key == "XLA"
        assert TpuPlatform.ray_device_key == "TPU"

        # Test supported quantization
        expected_quantization = []
        assert TpuPlatform.supported_quantization == expected_quantization

        # Test additional environment variables
        expected_env_vars = ["TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"]
        assert TpuPlatform.additional_env_vars == expected_env_vars

    def test_get_device_name(self):
        """Test get_device_name method."""
        device_name = TpuPlatform.get_device_name(0)
        assert device_name == "TPU"

    def test_get_device_total_memory_not_implemented(self):
        """Test that get_device_total_memory raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            TpuPlatform.get_device_total_memory(0)

    @patch.object(envs, 'VLLM_USE_V1', False)
    def test_is_async_output_supported_v0(self):
        """Test is_async_output_supported returns True for V0."""
        assert TpuPlatform.is_async_output_supported(None) is True
        assert TpuPlatform.is_async_output_supported(True) is True
        assert TpuPlatform.is_async_output_supported(False) is True

    @patch.object(envs, 'VLLM_USE_V1', True)
    def test_is_async_output_supported_v1(self):
        """Test is_async_output_supported returns False for V1."""
        assert TpuPlatform.is_async_output_supported(None) is False
        assert TpuPlatform.is_async_output_supported(True) is False
        assert TpuPlatform.is_async_output_supported(False) is False

    def test_get_punica_wrapper(self):
        """Test get_punica_wrapper raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            TpuPlatform.get_punica_wrapper()

    def test_get_infinity_values(self):
        """Test get_infinity_values for different dtypes."""
        # Test bfloat16
        min_val, max_val = TpuPlatform.get_infinity_values(torch.bfloat16)
        finfo = torch.finfo(torch.bfloat16)
        assert min_val == finfo.min
        assert max_val == finfo.max

        # Test float32
        min_val, max_val = TpuPlatform.get_infinity_values(torch.float32)
        finfo = torch.finfo(torch.float32)
        assert min_val == finfo.min
        assert max_val == finfo.max

    def test_can_update_inplace(self):
        """Test can_update_inplace returns False."""
        assert TpuPlatform.can_update_inplace() is False

    def test_get_lora_vocab_padding_size(self):
        """Test get_lora_vocab_padding_size raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            TpuPlatform.get_lora_vocab_padding_size()

    def test_inference_mode(self):
        """Test inference_mode returns torch.no_grad context."""
        context = TpuPlatform.inference_mode()
        assert hasattr(context, '__enter__') and hasattr(context, '__exit__')

    def test_is_pin_memory_available(self):
        """Test is_pin_memory_available returns False and logs warning."""
        with patch('tpu_commons.platforms.tpu_torchax.logger') as mock_logger:
            result = TpuPlatform.is_pin_memory_available()
            assert result is False
            mock_logger.warning.assert_called_once_with(
                "Pin memory is not supported on TPU.")

    def test_get_device_communicator_cls(self):
        """Test get_device_communicator_cls returns correct class."""
        communicator = TpuPlatform.get_device_communicator_cls()
        expected = "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"
        assert communicator == expected

    def test_use_all_gather(self):
        """Test use_all_gather returns True."""
        assert TpuPlatform.use_all_gather() is True

    def test_supports_v1(self):
        """Test supports_v1 returns True for any model config."""
        mock_model_config = Mock()
        assert TpuPlatform.supports_v1(mock_model_config) is True


class TestTpuPlatformAttnBackend:
    """Test attention backend selection logic."""

    def test_get_attn_backend_cls_pallas_v1(self):
        """Test attention backend selection for Pallas V1."""
        backend = TpuPlatform.get_attn_backend_cls(
            selected_backend=_Backend.PALLAS,
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype=None,
            block_size=16,
            use_v1=True,
            use_mla=False)

        expected = "tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend"
        assert backend == expected


#     def test_get_attn_backend_cls_pallas_v0(self):
#         """Test attention backend selection for Pallas V0 raises AssertionError."""
#         with pytest.raises(AssertionError, match="only v1 is supported"):
#             TpuPlatform.get_attn_backend_cls(
#                 selected_backend=_Backend.PALLAS,
#                 head_size=128,
#                 dtype=torch.bfloat16,
#                 kv_cache_dtype=None,
#                 block_size=16,
#                 use_v1=False,
#                 use_mla=False
#             )

#     def test_get_attn_backend_cls_unsupported_backend(self):
#         """Test attention backend selection with unsupported backend."""
#         backend = TpuPlatform.get_attn_backend_cls(
#             selected_backend=_Backend.FLASH_ATTN,  # Unsupported on TPU
#             head_size=128,
#             dtype=torch.bfloat16,
#             kv_cache_dtype=None,
#             block_size=16,
#             use_v1=True,
#             use_mla=False
#         )

#         expected = "tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend"
#         assert backend == expected

#     def test_get_attn_backend_cls_pallas_vllm_v1(self):
#         """Test attention backend selection for PALLAS_VLLM_V1."""
#         backend = TpuPlatform.get_attn_backend_cls(
#             selected_backend=_Backend.PALLAS_VLLM_V1,
#             head_size=128,
#             dtype=torch.bfloat16,
#             kv_cache_dtype=None,
#             block_size=16,
#             use_v1=True,
#             use_mla=False
#         )

#         expected = "tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend"
#         assert backend == expected


class TestTpuPlatformConfigUpdate:
    """Test config update logic."""

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend.get_page_size',
        return_value=16)
    @patch(
        'tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend.get_min_page_size',
        return_value=16)
    def test_check_and_update_config_basic(self, mock_get_min_page_size,
                                           mock_get_page_size,
                                           mock_vllm_config):
        """Test basic config update functionality."""
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.scheduler_config.is_multimodal_model = True
        mock_vllm_config.scheduler_config.disable_chunked_mm_input = False

        TpuPlatform.check_and_update_config(mock_vllm_config)

        # Check block size was set
        assert mock_vllm_config.cache_config.block_size == 16

        # Check model dtype is overwritten to bfloat16
        assert mock_vllm_config.model_config.dtype == torch.bfloat16

        # Check chunked mm input is disabled
        assert mock_vllm_config.scheduler_config.disable_chunked_mm_input

        # Verify the mocked methods were called
        mock_get_page_size.assert_called_once_with(mock_vllm_config)
        mock_get_min_page_size.assert_called_once_with(mock_vllm_config)

        # Check compilation level was forced to NO_COMPILATION
        assert mock_vllm_config.compilation_config.level == CompilationLevel.NO_COMPILATION

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend.get_page_size',
        return_value=16)
    @patch(
        'tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend.get_min_page_size',
        return_value=32)
    def test_check_and_update_config_overwrite_page_size(
            self, mock_get_min_page_size, mock_get_page_size, mock_vllm_config,
            caplog):
        """Test basic config update functionality."""
        TpuPlatform.check_and_update_config(mock_vllm_config)
        assert mock_vllm_config.cache_config.block_size == 32

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend.get_page_size',
        return_value=16)
    @patch(
        'tpu_commons.attention.backends.pallas_torchax.PallasAttentionBackend.get_min_page_size',
        return_value=16)
    @patch('tpu_commons.worker.tpu_worker_torchax.envs')
    @pytest.mark.parametrize("use_v1", [True, False])
    @pytest.mark.parametrize("is_multi_step", [True, False])
    def test_check_and_update_multi_step_scheduler(self,
                                                   mock_get_min_page_size,
                                                   mock_get_page_size,
                                                   mock_envs, mock_vllm_config,
                                                   use_v1, is_multi_step):
        """Test basic config update functionality."""
        mock_envs.VLLM_USE_V1 = use_v1
        mock_vllm_config.scheduler_config.is_multi_step = is_multi_step
        mock_vllm_config.parallel_config.worker_cls = "auto"

        if is_multi_step and use_v1:
            with pytest.raises(NotImplementedError):
                TpuPlatform.check_and_update_config(mock_vllm_config)


#     def test_check_and_update_config_compilation_level_warning(self, mock_vllm_config):
#         """Test compilation level is set to NO_COMPILATION."""
#         mock_vllm_config.compilation_config.level = CompilationLevel.DYNAMO_ONCE

#         TpuPlatform.check_and_update_config(mock_vllm_config)
#         assert mock_vllm_config.compilation_config.level == CompilationLevel.NO_COMPILATION

#     def test_check_and_update_config_speculative_assertion(self, mock_vllm_config):
#         """Test speculative config assertion."""
#         mock_vllm_config.speculative_config = SpeculativeConfig(
#             spec_decode_candidate_model="test", num_speculative_tokens=5
#         )

#         with pytest.raises(AssertionError, match="TPU does not support speculative decoding"):
#             TpuPlatform.check_and_update_config(mock_vllm_config)

#     def test_check_and_update_config_dtype_conversion(self, mock_vllm_config):
#         """Test dtype conversion from float16/float32 to bfloat16."""
#         # Test float16 conversion
#         mock_vllm_config.model_config.dtype = torch.float16

#         with patch('tpu_commons.platforms.tpu_torchax.logger') as mock_logger:
#             TpuPlatform.check_and_update_config(mock_vllm_config)
#             assert mock_vllm_config.model_config.dtype == torch.bfloat16
#             mock_logger.warning.assert_called_with(
#                 "The TPU backend currently does not support %s. "
#                 "Using bfloat16 instead.", torch.float16)

#         # Test float32 conversion
#         mock_vllm_config.model_config.dtype = torch.float32

#         with patch('tpu_commons.platforms.tpu_torchax.logger') as mock_logger:
#             TpuPlatform.check_and_update_config(mock_vllm_config)
#             assert mock_vllm_config.model_config.dtype == torch.bfloat16
#             mock_logger.warning.assert_called_with(
#                 "The TPU backend currently does not support %s. "
#                 "Using bfloat16 instead.", torch.float32)

#     def test_check_and_update_config_bfloat16_no_change(self, mock_vllm_config):
#         """Test that bfloat16 dtype is not changed."""
#         mock_vllm_config.model_config.dtype = torch.bfloat16

#         TpuPlatform.check_and_update_config(mock_vllm_config)
#         assert mock_vllm_config.model_config.dtype == torch.bfloat16

#     def test_check_and_update_config_page_size(self, mock_vllm_config):
#         """Test page size configuration."""
#         with patch('tpu_commons.platforms.tpu_torchax.PallasAttentionBackend') as mock_backend:
#             mock_backend.get_page_size.return_value = 64
#             mock_backend.get_min_page_size.return_value = 32

#             TpuPlatform.check_and_update_config(mock_vllm_config)

#             mock_backend.get_page_size.assert_called_once_with(mock_vllm_config)
#             mock_backend.get_min_page_size.assert_called_once_with(mock_vllm_config)
#             assert mock_vllm_config.cache_config.block_size == 64

#     def test_check_and_update_config_min_page_size_warning(self, mock_vllm_config):
#         """Test minimum page size warning."""
#         with patch('tpu_commons.platforms.tpu_torchax.PallasAttentionBackend') as mock_backend:
#             mock_backend.get_page_size.return_value = 32
#             mock_backend.get_min_page_size.return_value = 64  # Larger than page size

#             with patch('tpu_commons.platforms.tpu_torchax.logger') as mock_logger:
#                 TpuPlatform.check_and_update_config(mock_vllm_config)

#                 mock_logger.warning.assert_called_with(
#                     "Increase the page size from %s to %s to make sure there's"
#                     "no SMEM OOM",
#                     32,
#                     64,
#                 )
#                 assert mock_vllm_config.cache_config.block_size == 64

#     def test_check_and_update_config_worker_cls_auto_single_step_v0(self, mock_vllm_config):
#         """Test worker class selection for single-step V0."""
#         mock_vllm_config.parallel_config.worker_cls = "auto"
#         mock_vllm_config.scheduler_config.is_multi_step = False

#         with patch.object(envs, 'VLLM_USE_V1', False):
#             TpuPlatform.check_and_update_config(mock_vllm_config)
#             assert mock_vllm_config.parallel_config.worker_cls == "vllm.worker.tpu_worker.TPUWorker"

#     def test_check_and_update_config_worker_cls_auto_single_step_v1(self, mock_vllm_config):
#         """Test worker class selection for single-step V1."""
#         mock_vllm_config.parallel_config.worker_cls = "auto"
#         mock_vllm_config.scheduler_config.is_multi_step = False

#         with patch.object(envs, 'VLLM_USE_V1', True):
#             TpuPlatform.check_and_update_config(mock_vllm_config)
#             assert mock_vllm_config.parallel_config.worker_cls == "vllm.v1.worker.tpu_worker.TPUWorker"

#     def test_check_and_update_config_worker_cls_auto_multi_step_v0(self, mock_vllm_config):
#         """Test worker class selection for multi-step V0."""
#         mock_vllm_config.parallel_config.worker_cls = "auto"
#         mock_vllm_config.scheduler_config.is_multi_step = True

#         with patch.object(envs, 'VLLM_USE_V1', False):
#             TpuPlatform.check_and_update_config(mock_vllm_config)
#             expected = "vllm.worker.multi_step_tpu_worker.MultiStepTPUWorker"
#             assert mock_vllm_config.parallel_config.worker_cls == expected

#     def test_check_and_update_config_worker_cls_auto_multi_step_v1_error(self, mock_vllm_config):
#         """Test multi-step V1 raises NotImplementedError."""
#         mock_vllm_config.parallel_config.worker_cls = "auto"
#         mock_vllm_config.scheduler_config.is_multi_step = True

#         with patch.object(envs, 'VLLM_USE_V1', True):
#             with pytest.raises(NotImplementedError,
#                                match="Multi-step scheduling is not supported"):
#                 TpuPlatform.check_and_update_config(mock_vllm_config)

#     def test_check_and_update_config_worker_cls_not_auto(self, mock_vllm_config):
#         """Test that non-auto worker class is not changed."""
#         original_worker_cls = "custom.worker.CustomWorker"
#         mock_vllm_config.parallel_config.worker_cls = original_worker_cls

#         TpuPlatform.check_and_update_config(mock_vllm_config)
#         assert mock_vllm_config.parallel_config.worker_cls == original_worker_cls

#     def test_check_and_update_config_second_speculative_assertion(self, mock_vllm_config):
#         """Test second speculative config assertion."""
#         # Create a mock SpeculativeConfig that evaluates to True
#         mock_spec_config = MagicMock()
#         mock_spec_config.__bool__ = lambda self: True
#         mock_vllm_config.speculative_config = mock_spec_config

#         with pytest.raises(AssertionError,
#                            match="Speculative decoding is not yet supported for TPU backend"):
#             TpuPlatform.check_and_update_config(mock_vllm_config)

#     def test_check_and_update_config_multimodal_chunked_mm_input(self, mock_vllm_config):
#         """Test multimodal chunked MM input configuration."""
#         mock_vllm_config.scheduler_config.is_multimodal_model = True
#         mock_vllm_config.scheduler_config.disable_chunked_mm_input = False

#         with patch('tpu_commons.platforms.tpu_torchax.logger') as mock_logger:
#             TpuPlatform.check_and_update_config(mock_vllm_config)

#             assert mock_vllm_config.scheduler_config.disable_chunked_mm_input is True
#             mock_logger.warning.assert_called_with(
#                 "TPU does not support running Multimodal models"
#                 " without setting `--disable_chunked_mm_input`. "
#                 "Forcing --disable_chunked_mm_input.")

#     def test_check_and_update_config_multimodal_chunked_mm_input_already_disabled(self, mock_vllm_config):
#         """Test multimodal with chunked MM input already disabled."""
#         mock_vllm_config.scheduler_config.is_multimodal_model = True
#         mock_vllm_config.scheduler_config.disable_chunked_mm_input = True

#         with patch('tpu_commons.platforms.tpu_torchax.logger') as mock_logger:
#             TpuPlatform.check_and_update_config(mock_vllm_config)

#             # Should remain True
#             assert mock_vllm_config.scheduler_config.disable_chunked_mm_input is True
#             # No warning should be logged
#             mock_logger.warning.assert_not_called()

#     def test_check_and_update_config_non_multimodal(self, mock_vllm_config):
#         """Test non-multimodal model configuration."""
#         mock_vllm_config.scheduler_config.is_multimodal_model = False
#         mock_vllm_config.scheduler_config.disable_chunked_mm_input = False

#         TpuPlatform.check_and_update_config(mock_vllm_config)

#         # Should remain False for non-multimodal models
#         assert mock_vllm_config.scheduler_config.disable_chunked_mm_input is False


class TestTpuPlatformValidateRequest:
    """Test request validation logic."""

    # def test_validate_request_sampling_params_valid(self):
    #     """Test valid sampling params request."""
    #     prompt = "test prompt"
    #     params = SamplingParams(temperature=0.8, max_tokens=100)
    #     processed_inputs = Mock()

    #     # Should not raise any exception
    #     TpuPlatform.validate_request(prompt, params, processed_inputs)

    #     def test_validate_request_simple(self):
    #         """Test simple request validation."""
    #         prompt = "test prompt"
    #         params = SamplingParams()
    #         processed_inputs = Mock()

    #         # Should not raise any exception
    #         TpuPlatform.validate_request(prompt, params, processed_inputs)

    def test_validate_request_random_seed_error(self):
        """Test random seed sampling type error."""
        prompt = "test prompt"
        params = Mock(spec=SamplingParams)
        params.sampling_type = SamplingType.RANDOM_SEED
        processed_inputs = Mock()

        with pytest.raises(
                ValueError,
                match="Torch XLA does not support per-request seed"):
            TpuPlatform.validate_request(prompt, params, processed_inputs)


#     def test_validate_request_pooling_params(self):
#         """Test request validation with pooling params."""
#         prompt = "test prompt"
#         params = PoolingParams()
#         processed_inputs = Mock()

#         # Should not raise any exception
#         TpuPlatform.validate_request(prompt, params, processed_inputs)

#     def test_validate_request_no_guided_decoding(self):
#         """Test request validation with no guided decoding."""
#         prompt = "test prompt"
#         params = SamplingParams(guided_decoding=None)
#         processed_inputs = Mock()

#         # Should not raise any exception
#         TpuPlatform.validate_request(prompt, params, processed_inputs)

#     def test_validate_request_different_sampling_types(self):
#         """Test request validation with different sampling types."""
#         prompt = "test prompt"
#         processed_inputs = Mock()

#         # Test GREEDY sampling
#         params = Mock(spec=SamplingParams)
#         params.sampling_type = SamplingType.GREEDY
#         TpuPlatform.validate_request(prompt, params, processed_inputs)

#         # Test RANDOM sampling
#         params = Mock(spec=SamplingParams)
#         params.sampling_type = SamplingType.RANDOM
#         TpuPlatform.validate_request(prompt, params, processed_inputs)
