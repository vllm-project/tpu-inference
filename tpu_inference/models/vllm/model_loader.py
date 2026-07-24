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

from vllm.model_executor.models.registry import ModelRegistry

# Register DeepseekV32ForCausalLM / GLM52ForCausalLM out-of-tree model wrappers
ModelRegistry.register_model(
    "DeepseekV32ForCausalLM",
    "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
)
ModelRegistry.register_model(
    "GLM52ForCausalLM",
    "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
)
ModelRegistry.register_model(
    "Glm52ForCausalLM",
    "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
)
