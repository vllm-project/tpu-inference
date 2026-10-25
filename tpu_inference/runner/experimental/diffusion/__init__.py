# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental block-diffusion serving contracts.

This package is opt-in and has no API compatibility guarantee.
"""

from .algorithm import CommitFn, get_commit_algorithm, low_confidence_commit
from .config import (AttentionPolicy, CanvasPolicy, DiffusionAlgorithm,
                     DiffusionConfig, DiffusionModelSpec,
                     DiffusionRuntimeConfig, GenerationStrategy,
                     GenerationStrategyConfig, LogitAlignment, NextBlockPolicy,
                     PromptRemainderPolicy, register_diffusion_model_adapter,
                     resolve_generation_strategy)
from .program import BlockForwardFn, DenoiseBlockOutput, denoise_block

from .batch import (  # isort: skip
    PendingBlockOutput, PromptBlockPlan, complete_seeded_decode_block,
    flush_partial_block_output, plan_seeded_prompt, required_cache_end,
    start_partial_block_output,
)

__all__ = [
    "AttentionPolicy",
    "BlockForwardFn",
    "CanvasPolicy",
    "CommitFn",
    "DenoiseBlockOutput",
    "DiffusionAlgorithm",
    "DiffusionConfig",
    "DiffusionModelSpec",
    "DiffusionRuntimeConfig",
    "GenerationStrategy",
    "GenerationStrategyConfig",
    "LogitAlignment",
    "NextBlockPolicy",
    "PendingBlockOutput",
    "PromptBlockPlan",
    "PromptRemainderPolicy",
    "complete_seeded_decode_block",
    "denoise_block",
    "flush_partial_block_output",
    "get_commit_algorithm",
    "low_confidence_commit",
    "plan_seeded_prompt",
    "required_cache_end",
    "register_diffusion_model_adapter",
    "resolve_generation_strategy",
    "start_partial_block_output",
]
