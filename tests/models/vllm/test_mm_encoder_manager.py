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

import torch
from vllm.config import (CompilationConfig, ModelConfig, MultiModalConfig,
                         VllmConfig)
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

from tpu_inference.models.vllm.mm_encoder_manager import MMEncoderManager


def test_mm_encoder_manager_init():

    class DummyModel(SupportsEncoderCudaGraph):

        def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
            return EncoderCudaGraphConfig(
                modalities=["image", "video"],
                buffer_keys=["input_tensors"],
                out_hidden_size=768,
                max_frames_per_video=5,
            )

        def get_encoder_cudagraph_budget_range(
            self,
            vllm_config: VllmConfig,
        ) -> tuple[int, int]:
            return (128, 512)

        def prepare_encoder_cudagraph_capture_inputs(
            self,
            token_budget: int,
            max_batch_size: int,
            max_frames_per_batch: int,
            device: torch.device,
            dtype: torch.dtype,
        ):
            return {"input_tensors": torch.zeros((1, ))}

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            dtype=torch.float16,
            multimodal_config=MultiModalConfig(limit_per_prompt={
                "video": 0,
            }, ),
        ),
        compilation_config=CompilationConfig(
            encoder_cudagraph_token_budgets=[],
            encoder_cudagraph_max_vision_items_per_batch=0,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
    )

    # Initialize MMEncoderManager
    manager = MMEncoderManager(vllm_config, DummyModel())

    assert manager.dtype == torch.float16
    assert manager.token_budgets == [128, 256, 512]
    assert manager.max_batch_size == 4  # 512 // 128 = 4
    assert manager.max_frames_per_batch == 0
    assert list(manager.by_budget) == [128, 256, 512]


def test_generate_budgets():
    budgets = MMEncoderManager._generate_budgets(100, 1000)

    # Note that we always include the max budget
    assert budgets == [100, 200, 400, 800, 1000]
