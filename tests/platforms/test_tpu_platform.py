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
import torch
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
        ("v6e", torch.float8_e5m2),
        ("v5e", torch.float8_e4m3fn),
    ])
    def test_fp8_dtype(self, chip_name, expected_dtype):
        mock_chip_type = MagicMock()
        mock_chip_type.name = chip_name

        with patch('tpu_inference.platforms.tpu_platform.init_logger'), \
             patch('tpu_inference.platforms.tpu_platform.device.get_local_chips', return_value=(mock_chip_type, None)), \
             patch('vllm.envs.VLLM_TPU_USING_PATHWAYS', False):
            assert TpuPlatform.fp8_dtype() == expected_dtype
