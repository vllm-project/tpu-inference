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

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.utils.torch_utils import set_default_torch_num_threads


class MMEncoderManager:
    """
    The helper for multi-modal data processing.
    
    Responsible for most initialization task from vllm EncoderCudaGraphManager.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: SupportsEncoderCudaGraph,
    ):

        self.config = model.get_encoder_cudagraph_config()
        self.dtype = vllm_config.model_config.dtype

        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        token_budgets, max_batch_size = self._build_budgets_and_batch(
            vllm_config,
            model,
        )
        self.token_budgets = token_budgets
        """Sorted (ascending) list of token budgets."""
        self.max_batch_size = max_batch_size

        self.max_frames_per_batch = self._build_max_frames(
            vllm_config,
            self.max_batch_size,
            self.config,
        )

        self.by_budget = {
            budget:
            self._build_inputs(
                model,
                budget,
                self.max_batch_size,
                self.max_frames_per_batch,
                self.dtype,
            )
            for budget in self.token_budgets
        }
        """The "input tensors" for each token budget."""

    @staticmethod
    def _build_budgets_and_batch(
        vllm_config: VllmConfig,
        model: SupportsEncoderCudaGraph,
    ) -> tuple[list[int], int]:
        comp_config = vllm_config.compilation_config
        user_budgets = comp_config.encoder_cudagraph_token_budgets
        user_max_vision_items = comp_config.encoder_cudagraph_max_vision_items_per_batch
        min_budget, max_budget = model.get_encoder_cudagraph_budget_range(
            vllm_config)
        assert min_budget > 0 and max_budget > 0
        assert min_budget <= max_budget

        if user_max_vision_items > 0:
            # User provided max_vision_items only; adjust auto-inferred
            # budgets so min(budgets) >= max_batch_size.
            effective_min = max(min_budget, user_max_vision_items)
            token_budgets = MMEncoderManager._generate_budgets(
                effective_min,
                max_budget,
            )
            return token_budgets, user_max_vision_items
        elif user_budgets:
            # User provided budgets only; cap auto-inferred
            # max_batch_size to min(user_budgets).
            token_budgets = sorted(user_budgets)
            max_batch_size = min(
                max_budget // min_budget,
                min(token_budgets),
            )
            return token_budgets, max_batch_size
        else:
            # Fully auto-inferred.
            token_budgets = MMEncoderManager._generate_budgets(
                min_budget,
                max_budget,
            )
            max_batch_size = min(
                max_budget // min_budget,
                min(token_budgets),
            )
            return token_budgets, max_batch_size

    @staticmethod
    def _build_max_frames(
        vllm_config: VllmConfig,
        max_batch_size: int,
        config: Any,
    ) -> int:
        comp_config = vllm_config.compilation_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        user_max_frames = comp_config.encoder_cudagraph_max_frames_per_batch

        if multimodal_config.get_limit_per_prompt("video") == 0:
            return 0
        elif user_max_frames is not None:
            return user_max_frames
        else:
            # Set it to the model-specific value from config.
            return max_batch_size * config.max_frames_per_video

    @staticmethod
    def _build_inputs(
        model: SupportsEncoderCudaGraph,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        # Temporary overriding this value here can help mitigate an
        # assertion error in OpenMP (libgomp) shipped with PyTorch (2.10).
        # that breaks for budget 512 for model Qwen/Qwen3-VL-2B-Thinking.
        with set_default_torch_num_threads(None):
            inputs = model.prepare_encoder_cudagraph_capture_inputs(
                token_budget,
                max_batch_size,
                max_frames_per_batch,
                device=torch.device("cpu"),
                dtype=dtype,
            ).values

        return inputs

    @staticmethod
    def _generate_budgets(min_budget: int, max_budget: int) -> list[int]:
        """Generate power-of-2 token budgets from min_budget to max_budget."""
        # Copied from EncoderCudaGraphManager directly.
        budgets: list[int] = []
        b = min_budget
        while b <= max_budget:
            budgets.append(b)
            b *= 2
        # Always include max_budget if it's not already a power-of-2 boundary
        if not budgets or budgets[-1] < max_budget:
            budgets.append(max_budget)
        return budgets
