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

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import torch
from vllm.config import CacheConfig, CompilationMode, VllmConfig

from tpu_inference.platforms.tpu_platform import TpuPlatform


class TestTpuPlatform:

    @pytest.fixture
    def vllm_config(self):
        cache_config = CacheConfig(block_size=16,
                                   gpu_memory_utilization=0.9,
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
        ("v6e", torch.float8_e5m2),
        ("v5e", torch.float8_e4m3fn),
    ])
    def test_fp8_dtype(self, chip_name, expected_dtype):
        mock_chip_type = MagicMock()
        mock_chip_type.name = chip_name

        with patch('tpu_inference.platforms.tpu_platform.init_logger'), \
             patch('tpu_info.device.get_local_chips', return_value=(mock_chip_type, None)), \
             patch('vllm.envs.VLLM_TPU_USING_PATHWAYS', False):
            assert TpuPlatform.fp8_dtype() == expected_dtype

    def test_basic_properties(self):
        assert not TpuPlatform.is_async_output_supported(True)
        assert not TpuPlatform.can_update_inplace()
        assert not TpuPlatform.is_pin_memory_available()
        assert TpuPlatform.use_all_gather()
        assert TpuPlatform.supports_v1(MagicMock())
        assert TpuPlatform.is_kv_cache_dtype_supported("fp8", MagicMock())
        assert TpuPlatform.use_sync_weight_loader()
        assert TpuPlatform.support_hybrid_kv_cache()
        assert TpuPlatform.inference_mode()
        assert TpuPlatform.get_lora_vocab_padding_size() == 1
        assert TpuPlatform.current_device().type == "cpu"

        expected_punica = "tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU"
        assert TpuPlatform.get_punica_wrapper() == expected_punica

        expected_communicator = "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"
        assert TpuPlatform.get_device_communicator_cls(
        ) == expected_communicator

    def test_get_device_total_memory(self):
        with pytest.raises(NotImplementedError):
            TpuPlatform.get_device_total_memory()

    def test_get_infinity_values(self):
        min_val, max_val = TpuPlatform.get_infinity_values(jnp.float32)
        assert min_val == jnp.finfo(jnp.float32).min
        assert max_val == jnp.finfo(jnp.float32).max

    @patch('tpu_inference.platforms.tpu_platform.vllm_envs')
    def test_get_device_name_pathways(self, mock_envs):
        mock_envs.VLLM_TPU_USING_PATHWAYS = True
        assert TpuPlatform.get_device_name() == "TPU v6 lite"

    @patch('tpu_inference.platforms.tpu_platform.vllm_envs')
    def test_get_device_name_exception(self, mock_envs):
        mock_envs.VLLM_TPU_USING_PATHWAYS = False
        mock_device = MagicMock()
        mock_device.get_local_chips.side_effect = Exception("TPU Error")

        with patch.dict('sys.modules', {
                'tpu_info': MagicMock(),
                'tpu_info.device': mock_device
        }):
            assert TpuPlatform.get_device_name() == "TPU"

    def test_validate_request_random_seed(self):
        from vllm.sampling_params import SamplingParams, SamplingType
        params = MagicMock(spec=SamplingParams)
        params.sampling_type = SamplingType.RANDOM_SEED
        with pytest.raises(ValueError,
                           match="JAX does not support per-request seed"):
            TpuPlatform.validate_request(MagicMock(), params)

    def test_validate_request_valid(self):
        from vllm.sampling_params import SamplingParams, SamplingType
        params = MagicMock(spec=SamplingParams)
        params.sampling_type = SamplingType.GREEDY
        # Should execute cleanly without raising
        TpuPlatform.validate_request(MagicMock(), params)

    def test_get_attn_backend_cls_mla(self):
        with patch.dict('sys.modules',
                        {'tpu_inference.layers.vllm.backends': MagicMock()}):
            with patch(
                    "vllm.v1.attention.backends.registry.AttentionBackendEnum"
            ) as mock_attn_enum:
                mock_attn_enum.FLASH_ATTN_MLA = MagicMock()
                mock_attn_enum.FLASH_ATTN_MLA.get_path.return_value = "mla_path"

                config = MagicMock()
                config.use_mla = True

                res = TpuPlatform.get_attn_backend_cls(MagicMock(), config)
                assert res == "mla_path"

    def test_get_attn_backend_cls_fallback(self):
        with patch.dict('sys.modules',
                        {'tpu_inference.layers.vllm.backends': MagicMock()}):
            with patch(
                    "vllm.v1.attention.backends.registry.AttentionBackendEnum"
            ) as mock_attn_enum:
                mock_attn_enum.FLASH_ATTN = MagicMock()
                mock_attn_enum.FLASH_ATTN.get_path.return_value = "flash_path"

                config = MagicMock()
                config.use_mla = False

                selected = MagicMock()
                selected.name = "OTHER"

                res = TpuPlatform.get_attn_backend_cls(selected, config)
                assert res == "flash_path"

    @patch('tpu_inference.platforms.tpu_platform.vllm_envs')
    def test_check_and_update_config_pathways_assert(self, mock_envs,
                                                     vllm_config):
        mock_envs.VLLM_TPU_USING_PATHWAYS = True
        mock_envs.VLLM_ENABLE_V1_MULTIPROCESSING = True

        with pytest.raises(AssertionError,
                           match="VLLM_ENABLE_V1_MULTIPROCESSING must be 0"):
            TpuPlatform.check_and_update_config(vllm_config)

    @patch("tpu_inference.platforms.tpu_platform.envs.TPU_MULTIHOST_BACKEND",
           "")
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    def test_check_and_update_config_single_host_uni(self, mock_update,
                                                     mock_sharding,
                                                     vllm_config):
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.scheduler_config.is_multimodal_model = True
        vllm_config.scheduler_config.disable_chunked_mm_input = False
        vllm_config.compilation_config.mode = "dummy"
        vllm_config.compilation_config.backend = ""
        vllm_config.cache_config = None  # Skip block size checks to isolate logic

        TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.compilation_config.mode == CompilationMode.DYNAMO_TRACE_ONCE
        assert vllm_config.compilation_config.backend == "openxla"
        assert vllm_config.parallel_config.distributed_executor_backend == "uni"
        assert vllm_config.scheduler_config.disable_chunked_mm_input is True

        mock_sharding.from_vllm_config.assert_called_once_with(vllm_config)
        assert vllm_config.sharding_config == mock_sharding.from_vllm_config.return_value
        mock_update.assert_called_once_with(vllm_config)

    @patch("tpu_inference.platforms.tpu_platform.envs.TPU_MULTIHOST_BACKEND",
           "")
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    def test_check_and_update_config_single_host_multiproc(
            self, mock_update, mock_sharding, vllm_config):
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.cache_config = None

        with patch.dict(
                'sys.modules',
            {'tpu_inference.executors.multiproc_executor': MagicMock()}):
            from tpu_inference.executors.multiproc_executor import \
                MultiprocExecutor
            TpuPlatform.check_and_update_config(vllm_config)
            assert vllm_config.parallel_config.distributed_executor_backend == MultiprocExecutor

    @patch("tpu_inference.platforms.tpu_platform.envs.TPU_MULTIHOST_BACKEND",
           "ray")
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    def test_check_and_update_config_ray(self, mock_update, mock_sharding,
                                         vllm_config):
        vllm_config.cache_config = None

        with patch.dict(
                'sys.modules',
            {'tpu_inference.executors.ray_distributed_executor': MagicMock()}):
            from tpu_inference.executors.ray_distributed_executor import \
                RayDistributedExecutor
            TpuPlatform.check_and_update_config(vllm_config)
            assert vllm_config.parallel_config.distributed_executor_backend == RayDistributedExecutor

    @patch("tpu_inference.platforms.tpu_platform.envs.TPU_MULTIHOST_BACKEND",
           "unknown_backend")
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    def test_check_and_update_config_unknown(self, mock_update, mock_sharding,
                                             vllm_config):
        vllm_config.cache_config = None
        TpuPlatform.check_and_update_config(vllm_config)

        # Fallback behaviour defaults to `uni`
        assert vllm_config.parallel_config.distributed_executor_backend == "uni"

    @patch("tpu_inference.platforms.tpu_platform.envs.TPU_MULTIHOST_BACKEND",
           "")
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    @patch("tpu_inference.platforms.tpu_platform.logger")
    def test_check_and_update_config_block_size(self, mock_logger, mock_update,
                                                mock_sharding, vllm_config):
        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 16
        mock_pallas.get_min_page_size.return_value = 64

        vllm_config.cache_config = MagicMock()
        vllm_config.cache_config.user_specified_block_size = False
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.use_mla = False

        with patch.dict(
                'sys.modules', {
                    'tpu_inference.layers.vllm.backends.flash_attn':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
            TpuPlatform.check_and_update_config(vllm_config)
            # The test should check and update min_page_size and override block_size if min_page_size is larger
            assert vllm_config.cache_config.block_size == 64

    def test_update_block_size_for_backend(self, vllm_config):
        # Current implementation explicitly performs 'pass'. Ensure it returns safely.
        TpuPlatform.update_block_size_for_backend(vllm_config)

    def test_update_block_size_for_backend_align_hybrid_block_size(
            self, vllm_config):
        vllm_config.model_config.architecture = "Qwen3_5ForConditionalGeneration"
        vllm_config.model_config.is_hybrid = True
        vllm_config.model_config.get_num_kv_heads.return_value = 2
        vllm_config.model_config.get_head_size.return_value = 256
        vllm_config.model_config.dtype = torch.uint8
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.cache_config.block_size = -1

        mock_backend = MagicMock()
        mock_backend.get_mamba_state_shape_from_config.return_value = ((3,
                                                                        1536),
                                                                       (8, 128,
                                                                        128))
        mock_backend.get_mamba_state_dtype_from_config.return_value = (
            torch.bfloat16, torch.float32)
        mock_backend.get_supported_kernel_block_sizes.return_value = [256]

        with patch.object(TpuPlatform, '_find_non_ssm_backend', return_value=mock_backend), \
             patch('vllm.model_executor.models.ModelRegistry.resolve_model_cls', return_value=(mock_backend, MagicMock())):
            TpuPlatform.update_block_size_for_backend(vllm_config)

        assert vllm_config.cache_config.block_size == 1280
