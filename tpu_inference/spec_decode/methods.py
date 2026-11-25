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
"""Speculative-decoding method checks that do not depend on the vLLM version.

vllm-project/vllm#68798 removed `SpeculativeConfig.use_eagle()`,
`use_gemma4_mtp()` and the other `use_*()` helpers in favor of
`SpeculativeConfig.speculator_config`. These helpers reproduce the removed
checks from `SpeculativeConfig.method` and the draft model config, so the same
code runs on vLLM versions before and after that change.
"""

from typing import Any

# Methods whose drafter consumes the target model's hidden states; the methods
# the removed `SpeculativeConfig.use_eagle()` matched.
_EAGLE_STYLE_METHODS = frozenset(
    ("eagle", "eagle3", "mtp", "dflash", "dspark"))


def uses_eagle(speculative_config: Any) -> bool:
    """Whether the method drafts from the target model's hidden states."""
    return speculative_config.method in _EAGLE_STYLE_METHODS


def uses_gemma4_mtp(speculative_config: Any) -> bool:
    """Whether the method is MTP with a Gemma4-MTP draft checkpoint."""
    if speculative_config.method != "mtp":
        return False
    draft_model_config = speculative_config.draft_model_config
    if draft_model_config is None:
        return False
    return getattr(draft_model_config.hf_config, "model_type",
                   None) == "gemma4_mtp"
