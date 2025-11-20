from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig, VllmConfig

from tpu_inference.platforms.tpu_platform import TpuPlatform


class TestTpuPlatform:

    @pytest.fixture
    def vllm_config(self):
        cache_config = CacheConfig(block_size=16,
                                   gpu_memory_utilization=0.9,
                                   swap_space=4,
                                   cache_dtype="fp8")

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.cache_config = cache_config
        vllm_config.model_config = MagicMock(dtype='bfloat16')
        vllm_config.scheduler_config = MagicMock(is_multimodal_model=False)
        vllm_config.parallel_config = MagicMock()
        vllm_config.compilation_config = MagicMock(mode="dynamo_trace_once",
                                                   backend="openxla")
        vllm_config.kv_transfer_config = None
        return vllm_config

    @pytest.mark.parametrize("chip_name,expected_dtype", [
        ("v6e", "fp8_e5m2"),
        ("v5e", "fp8"),
    ])
    def test_check_and_update_config_fp8(self, chip_name, expected_dtype,
                                         vllm_config):
        mock_chip_type = MagicMock()
        mock_chip_type.name = chip_name

        # Common patches
        with patch('tpu_inference.platforms.tpu_platform.init_logger'), \
             patch('tpu_inference.platforms.tpu_platform.device.get_local_chips', return_value=(mock_chip_type, None)), \
             patch('vllm.envs.VLLM_TPU_USING_PATHWAYS', False), \
             patch('tpu_inference.platforms.tpu_platform.ShardingConfigManager.from_vllm_config'), \
             patch('tpu_inference.platforms.tpu_platform.envs.MODEL_IMPL_TYPE', "vllm"), \
             patch('vllm.v1.attention.backends.pallas.PallasAttentionBackend.get_page_size', return_value=16), \
             patch('vllm.v1.attention.backends.pallas.PallasAttentionBackend.get_min_page_size', return_value=16), \
             patch('tpu_inference.models.jax.utils.quantization.quantization_utils.update_vllm_config_for_qwix_quantization'), \
             patch('tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler'):

            TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.cache_dtype == expected_dtype
