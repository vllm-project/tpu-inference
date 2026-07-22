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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import jax.numpy as jnp
import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig

from tpu_inference.platforms.tpu_platform import TpuPlatform


class TestTpuPlatform:

    @pytest.fixture(autouse=True)
    def _restore_multiprocess_dp_env(self):
        saved = os.environ.get("TPU_MULTIPROCESS_DP")
        yield
        if saved is None:
            os.environ.pop("TPU_MULTIPROCESS_DP", None)
        else:
            os.environ["TPU_MULTIPROCESS_DP"] = saved

    @pytest.fixture
    def vllm_config(self):
        cache_config = CacheConfig(block_size=16,
                                   gpu_memory_utilization=0.9,
                                   cache_dtype="fp8")

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.cache_config = cache_config
        vllm_config.model_config = MagicMock(dtype='bfloat16')
        vllm_config.model_config.use_mla = False
        vllm_config.scheduler_config = MagicMock(is_multimodal_model=False)
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.sharding_config = MagicMock()
        vllm_config.compilation_config = MagicMock(backend="eager")
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {}
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.scheduler_config.enable_chunked_prefill = False
        return vllm_config

    @staticmethod
    def _enable_block_diffusion(vllm_config):
        vllm_config.additional_config = {
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "dgr2",
                "block_size": 32,
                "mask_token_id": 151669,
                "sub_block_size": 8,
            },
        }
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.model_config.runner_type = "generate"
        vllm_config.model_config.is_multimodal_model = False
        vllm_config.model_config.hf_config.head_dim = 128
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.scheduler_config.enable_chunked_prefill = False
        vllm_config.scheduler_config.max_num_batched_tokens = 2048
        vllm_config.scheduler_config.max_num_seqs = 128
        vllm_config.speculative_config = None
        vllm_config.lora_config = None
        vllm_config.kv_transfer_config = None
        vllm_config.cache_config = None

    @pytest.mark.parametrize("accelerator_type,expected_dtype", [
        ("v6e-8", torch.float8_e5m2),
        ("v5litepod-4", torch.float8_e4m3fn),
        ("tpu7x-128", torch.float8_e4m3fn),
    ])
    def test_fp8_dtype(self, accelerator_type, expected_dtype):
        with patch('tpu_inference.platforms.tpu_platform.init_logger'), \
             patch('tpu_inference.tpu_info.get_tpu_type', return_value=accelerator_type), \
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

    @patch('tpu_inference.platforms.tpu_platform.jax.local_devices')
    def test_mem_get_info(self, mock_local_devices):
        mock_dev1 = MagicMock()
        mock_dev1.memory_stats.return_value = {
            'bytes_limit': 1000,
            'bytes_in_use': 200
        }
        mock_dev2 = MagicMock()
        mock_dev2.memory_stats.return_value = {
            'bytes_limit': 1000,
            'bytes_in_use': 300
        }
        mock_local_devices.return_value = [mock_dev1, mock_dev2]

        free_mem, total_mem = TpuPlatform.mem_get_info()

        assert total_mem == 2000
        assert free_mem == 1500

    def test_get_infinity_values(self):
        min_val, max_val = TpuPlatform.get_infinity_values(jnp.float32)
        assert min_val == jnp.finfo(jnp.float32).min
        assert max_val == jnp.finfo(jnp.float32).max

    @patch('tpu_inference.platforms.tpu_platform.vllm_envs')
    def test_get_device_name_pathways(self, mock_envs):
        mock_envs.VLLM_TPU_USING_PATHWAYS = True
        assert TpuPlatform.get_device_name() == "TPU v6 lite"

    @pytest.mark.parametrize("accelerator_type,expected_name", [
        ("v6e-8", "TPU v6e"),
        ("v5litepod-16", "TPU v5e"),
        ("tpu7x-128", "TPU v7x"),
        ("v5p-32", "TPU v5p"),
    ])
    @patch('tpu_inference.platforms.tpu_platform.vllm_envs')
    def test_get_device_name(self, mock_envs, accelerator_type, expected_name):
        mock_envs.VLLM_TPU_USING_PATHWAYS = False
        with patch('tpu_inference.tpu_info.get_tpu_type',
                   return_value=accelerator_type):
            assert TpuPlatform.get_device_name() == expected_name

    @patch('tpu_inference.platforms.tpu_platform.vllm_envs')
    def test_get_device_name_exception(self, mock_envs):
        mock_envs.VLLM_TPU_USING_PATHWAYS = False
        with patch('tpu_inference.tpu_info.get_tpu_type',
                   side_effect=Exception("TPU Error")):
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

        assert vllm_config.parallel_config.distributed_executor_backend == "uni"
        assert vllm_config.scheduler_config.disable_chunked_mm_input is False

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
                                         vllm_config, monkeypatch):
        vllm_config.cache_config = None
        monkeypatch.setattr(
            "tpu_inference.platforms.tpu_platform.vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND",
            False,
        )

        with patch.dict(
                'sys.modules',
            {'tpu_inference.executors.ray_distributed_executor': MagicMock()}):
            from tpu_inference.executors.ray_distributed_executor import \
                RayDistributedExecutor
            TpuPlatform.check_and_update_config(vllm_config)
            assert vllm_config.parallel_config.distributed_executor_backend == RayDistributedExecutor

    @patch("tpu_inference.platforms.tpu_platform.envs.TPU_MULTIHOST_BACKEND",
           "ray")
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    def test_check_and_update_config_ray_v2(self, mock_update, mock_sharding,
                                            vllm_config, monkeypatch):
        vllm_config.cache_config = None
        monkeypatch.setattr(
            "tpu_inference.platforms.tpu_platform.vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND",
            True,
        )

        with patch.dict(
                'sys.modules', {
                    'tpu_inference.executors.ray_distributed_executor_v2':
                    MagicMock(),
                    'vllm.v1.executor.ray_executor_v2': MagicMock()
                }):
            from tpu_inference.executors.ray_distributed_executor_v2 import \
                RayDistributedExecutorV2
            TpuPlatform.check_and_update_config(vllm_config)
            assert vllm_config.parallel_config.distributed_executor_backend == RayDistributedExecutorV2

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

    @pytest.mark.parametrize("enable_batch_rpa,expected_block_size", [
        (True, 64),
        (False, 64),
    ])
    def test_update_block_size_for_backend_align_hybrid_block_size(
            self, vllm_config, enable_batch_rpa, expected_block_size):
        vllm_config.model_config.architecture = "Qwen3_5ForConditionalGeneration"
        vllm_config.model_config.is_hybrid = True
        vllm_config.model_config.use_mla = True
        vllm_config.model_config.get_num_kv_heads.return_value = 2
        vllm_config.model_config.get_head_size.return_value = 256
        vllm_config.model_config.dtype = torch.uint8
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.cache_config.block_size = 64

        mock_backend = MagicMock()
        mock_backend.get_mamba_state_shape_from_config.return_value = ((3,
                                                                        1536),
                                                                       (8, 128,
                                                                        128))
        mock_backend.get_mamba_state_dtype_from_config.return_value = (
            torch.bfloat16, torch.float32)
        mock_backend.get_supported_kernel_block_sizes.return_value = [256]

        with patch.object(TpuPlatform, '_find_non_ssm_backend', return_value=mock_backend), \
             patch('vllm.model_executor.models.ModelRegistry.resolve_model_cls', return_value=(mock_backend, MagicMock())), \
             patch("tpu_inference.envs.USE_BATCHED_RPA_KERNEL", enable_batch_rpa):
            TpuPlatform.update_block_size_for_backend(vllm_config)

        assert vllm_config.cache_config.block_size == expected_block_size

    def test_check_and_update_config_mla_checks(self):
        vllm_config = MagicMock()
        vllm_config.model_config.use_mla = True
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.additional_config = {}

        expected_msg = r"MLA models require both the NEW_MODEL_DESIGN=1 environment.*"

        # Test both unset NEW_MODEL_DESIGN and unset additional_config to ensure they are both
        # required
        with patch("tpu_inference.envs.NEW_MODEL_DESIGN", False):
            with pytest.raises(ValueError, match=expected_msg):
                TpuPlatform.check_and_update_config(vllm_config)

        # Next, test set NEW_MODEL_DESIGN and unset additional_config to ensure it is required
        with patch("tpu_inference.envs.NEW_MODEL_DESIGN", True):
            vllm_config.additional_config = {}
            with pytest.raises(ValueError, match=expected_msg):
                TpuPlatform.check_and_update_config(vllm_config)

            # Test where sharding_strategy exists but missing `enable_dp_attention`
            vllm_config.additional_config = {
                "sharding": {
                    "sharding_strategy": {
                        "debug_key": True
                    }
                }
            }
            with pytest.raises(ValueError, match=expected_msg):
                TpuPlatform.check_and_update_config(vllm_config)

            # Test where `enable_dp_attention` exists but is explicitly set to False
            vllm_config.additional_config = {
                "sharding": {
                    "sharding_strategy": {
                        "enable_dp_attention": False
                    }
                }
            }
            with pytest.raises(ValueError, match=expected_msg):
                TpuPlatform.check_and_update_config(vllm_config)

        # Lastly, test both set NEW_MODEL_DESIGN and set additional_config -> success
        with patch("tpu_inference.envs.NEW_MODEL_DESIGN", True):
            with patch.object(TpuPlatform, "_initialize_sharding_config"):
                with patch("vllm.config.CompilationMode"):
                    vllm_config.additional_config = {
                        "sharding": {
                            "sharding_strategy": {
                                "enable_dp_attention": True
                            }
                        }
                    }
                    try:
                        TpuPlatform.check_and_update_config(vllm_config)
                    except Exception as e:
                        # We only care if it's the germane ValueError
                        if isinstance(e, ValueError) and ("MLA models require"
                                                          in str(e)):
                            pytest.fail(
                                f"MLA check failed even when config was correct: {e}"
                            )

    def test_check_and_update_config_mla_disabled_via_env(self, vllm_config):
        vllm_config.additional_config = {}

        mock_model_config = MagicMock(spec=ModelConfig)
        mock_model_config.is_deepseek_mla = True

        real_use_mla_getter = ModelConfig.use_mla.fget
        type(mock_model_config).use_mla = PropertyMock(
            side_effect=lambda: real_use_mla_getter(mock_model_config))

        vllm_config.model_config = mock_model_config

        with patch("vllm.envs.VLLM_MLA_DISABLE", True):
            # Assert vLLM logic evaluates to False as expected when env var is set
            assert vllm_config.model_config.use_mla is False

            # Neither NEW_MODEL_DESIGN nor enable_dp_attention are set
            # which should be fine since MLA is disabled
            with patch("tpu_inference.envs.NEW_MODEL_DESIGN", False):
                with patch.object(TpuPlatform, "_initialize_sharding_config"):
                    with patch("vllm.config.CompilationMode"):
                        try:
                            TpuPlatform.check_and_update_config(vllm_config)
                        except Exception as e:
                            if isinstance(e, ValueError
                                          ) and "MLA models require" in str(e):
                                pytest.fail(
                                    f"MLA check failed unexpectedly when VLLM_MLA_DISABLE was set: {e}"
                                )

    @staticmethod
    def _dp_config(data_parallel_size, enable_dp_attention, api_process_rank):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = data_parallel_size
        vllm_config.parallel_config._api_process_rank = api_process_rank
        vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": enable_dp_attention
                }
            }
        }
        return vllm_config

    @pytest.mark.parametrize(
        "api_rank,dp_size,dp_attn,pathways,expected",
        [
            (-1, 2, False, False, "1"),  # online serve (rank -1), DP>1 -> on
            (0, 2, False, False, "0"),  # offline LLM() (rank 0) -> off
            (-1, 1, False, False, "0"),  # no DP -> off
            (-1, 2, True, False, "0"),  # attention DP -> off
            (-1, 2, False, True, "0"),  # Pathways -> off
        ],
    )
    def test_resolve_multiprocess_dp_derives_when_unset(
            self, monkeypatch, api_rank, dp_size, dp_attn, pathways, expected):
        monkeypatch.delenv("TPU_MULTIPROCESS_DP", raising=False)
        monkeypatch.setattr(
            "tpu_inference.platforms.tpu_platform.vllm_envs."
            "VLLM_TPU_USING_PATHWAYS", pathways)
        vllm_config = self._dp_config(dp_size, dp_attn, api_rank)

        TpuPlatform._resolve_multiprocess_dp(vllm_config)

        # DP <= 1 is a no-op (left unset); otherwise it pins a concrete value.
        assert os.environ.get("TPU_MULTIPROCESS_DP") == (expected if dp_size
                                                         > 1 else None)

    @pytest.mark.parametrize("preset", ["0", "1"])
    def test_resolve_multiprocess_dp_preserves_explicit(
            self, monkeypatch, preset):
        # An explicit setting in a supported (online) context must not be
        # overwritten.
        monkeypatch.setenv("TPU_MULTIPROCESS_DP", preset)
        monkeypatch.setattr(
            "tpu_inference.platforms.tpu_platform.vllm_envs."
            "VLLM_TPU_USING_PATHWAYS", False)
        vllm_config = self._dp_config(data_parallel_size=2,
                                      enable_dp_attention=False,
                                      api_process_rank=-1)

        TpuPlatform._resolve_multiprocess_dp(vllm_config)

        assert os.environ["TPU_MULTIPROCESS_DP"] == preset

    def test_resolve_multiprocess_dp_explicit_online_worker_ok(
            self, monkeypatch):
        # An online API-server worker has _api_process_rank >= 0 (not -1) but
        # inherits the explicit "1"; it must NOT be misread as offline and must
        # not raise.
        monkeypatch.setenv("TPU_MULTIPROCESS_DP", "1")
        monkeypatch.setattr(
            "tpu_inference.platforms.tpu_platform.vllm_envs."
            "VLLM_TPU_USING_PATHWAYS", False)
        vllm_config = self._dp_config(data_parallel_size=2,
                                      enable_dp_attention=False,
                                      api_process_rank=2)

        TpuPlatform._resolve_multiprocess_dp(vllm_config)  # must not raise

        assert os.environ["TPU_MULTIPROCESS_DP"] == "1"

    @pytest.mark.parametrize(
        "dp_attn,pathways",
        [
            (True, False),  # attention DP
            (False, True),  # Pathways
        ])
    def test_resolve_multiprocess_dp_explicit_incompatible_raises(
            self, monkeypatch, dp_attn, pathways):
        # An explicit TPU_MULTIPROCESS_DP=1 with attention DP or on Pathways
        # must fail loudly rather than be silently downgraded.
        monkeypatch.setenv("TPU_MULTIPROCESS_DP", "1")
        monkeypatch.setattr(
            "tpu_inference.platforms.tpu_platform.vllm_envs."
            "VLLM_TPU_USING_PATHWAYS", pathways)
        vllm_config = self._dp_config(data_parallel_size=2,
                                      enable_dp_attention=dp_attn,
                                      api_process_rank=-1)

        with pytest.raises(ValueError, match="TPU_MULTIPROCESS_DP=1"):
            TpuPlatform._resolve_multiprocess_dp(vllm_config)

    def test_resolve_multiprocess_dp_explicit_incompatible_no_dp_ok(
            self, monkeypatch):
        # With DP <= 1 there is no multi-process DP, so an explicit opt-in in
        # an otherwise-unsupported context is irrelevant and must not raise.
        monkeypatch.setenv("TPU_MULTIPROCESS_DP", "1")
        vllm_config = self._dp_config(data_parallel_size=1,
                                      enable_dp_attention=True,
                                      api_process_rank=0)

        TpuPlatform._resolve_multiprocess_dp(vllm_config)  # must not raise

    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    @patch(
        "tpu_inference.core.sched.utils.patch_vllm_scheduler_for_continue_decode"
    )
    def test_check_and_update_config_continue_decode_success(
            self, mock_patch, mock_dp_update, mock_sharding, vllm_config):
        vllm_config.additional_config = {"enable_continue_decode": True}
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.model_config.runner_type = "generate"  # not pooling
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.cache_config = None

        TpuPlatform.check_and_update_config(vllm_config)

        mock_patch.assert_called_once()

    @pytest.mark.parametrize(
        "pp_size, runner_type, async_sched, expected_error",
        [
            (2, "generate", False,
             "continue_decode is not supported with pipeline parallelism"),
            (1, "pooling", False,
             "continue_decode is not supported for pooling models"),
        ],
    )
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    @patch(
        "tpu_inference.core.sched.utils.patch_vllm_scheduler_for_continue_decode"
    )
    def test_check_and_update_config_continue_decode_errors(
        self,
        mock_patch,
        mock_dp_update,
        mock_sharding,
        vllm_config,
        pp_size,
        runner_type,
        async_sched,
        expected_error,
    ):
        vllm_config.additional_config = {"enable_continue_decode": True}
        vllm_config.parallel_config.pipeline_parallel_size = pp_size
        vllm_config.model_config.runner_type = runner_type
        vllm_config.scheduler_config.async_scheduling = async_sched
        vllm_config.cache_config = None

        with pytest.raises(ValueError, match=expected_error):
            TpuPlatform.check_and_update_config(vllm_config)
        mock_patch.assert_not_called()

    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    @patch(
        "tpu_inference.core.sched.utils.patch_vllm_scheduler_for_multi_token_decode"
    )
    def test_check_and_update_config_block_diffusion_success(
            self, mock_patch, mock_dp_update, mock_sharding, vllm_config):
        self._enable_block_diffusion(vllm_config)

        TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.additional_config[
            "multi_token_decode_lookahead"] == 31
        assert vllm_config.scheduler_config.max_num_seqs == 64
        mock_patch.assert_called_once()

    @pytest.mark.parametrize(
        "unsupported,expected_error",
        [
            ("kv_transfer", "does not support KV transfer"),
            ("requested_dp", "requires data_parallel_size=1"),
            ("sharding_dp", "requires data_parallel_size=1"),
            ("chunked_prefill", "requires chunked prefill to be disabled"),
            ("token_budget", "minimum padded batch"),
        ],
    )
    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    @patch(
        "tpu_inference.core.sched.dp_scheduler.update_vllm_config_for_dp_scheduler"
    )
    @patch(
        "tpu_inference.core.sched.utils.patch_vllm_scheduler_for_multi_token_decode"
    )
    def test_check_and_update_config_block_diffusion_rejects_unsupported_modes(
        self,
        mock_patch,
        mock_dp_update,
        mock_sharding,
        vllm_config,
        unsupported,
        expected_error,
    ):
        self._enable_block_diffusion(vllm_config)
        mock_sharding.from_vllm_config.return_value.total_dp_size = 1
        if unsupported == "kv_transfer":
            vllm_config.kv_transfer_config = SimpleNamespace(
                kv_connector="TPUConnector")
        elif unsupported == "requested_dp":
            vllm_config.parallel_config.data_parallel_size = 2
        elif unsupported == "sharding_dp":
            mock_sharding.from_vllm_config.return_value.total_dp_size = 2
        elif unsupported == "chunked_prefill":
            vllm_config.scheduler_config.enable_chunked_prefill = True
        elif unsupported == "token_budget":
            vllm_config.scheduler_config.max_num_batched_tokens = 255

        with pytest.raises(ValueError, match=expected_error):
            TpuPlatform.check_and_update_config(vllm_config)

        mock_patch.assert_not_called()

    @patch("tpu_inference.platforms.tpu_platform.ShardingConfigManager")
    def test_check_and_update_config_rejects_two_multi_token_modes(
            self, mock_sharding, vllm_config):
        vllm_config.additional_config = {
            "enable_continue_decode": True,
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "dgr2",
                "block_size": 32,
                "mask_token_id": 151669,
            },
        }
        vllm_config.cache_config = None

        with pytest.raises(ValueError, match="mutually exclusive"):
            TpuPlatform.check_and_update_config(vllm_config)
