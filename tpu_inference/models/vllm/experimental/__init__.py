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

# Mapping of architecture name -> "module:ClassName" for the TPU-specific
# vLLM/torchax model implementations that should override vLLM's built-in
# ones. Values use the lazy string form so the (torch/tilelang) model module
# is only imported when the architecture is actually instantiated.
_TPU_VLLM_MODELS = {
    "DeepseekV4ForCausalLM":
    "tpu_inference.models.vllm.experimental.deepseek_v4:DeepseekV4ForCausalLM",
    "DeepseekV32ForCausalLM":
    "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
    "GLM52ForCausalLM":
    "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
    "Glm52ForCausalLM":
    "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
}


def register_models():
    """Override vLLM's built-in model classes with TPU-specific ones.

    Called from the ``vllm.general_plugins`` entrypoint so it runs at vLLM
    startup, before any model is resolved from its architecture name.
    """
    from vllm import ModelRegistry

    for arch, model_cls in _TPU_VLLM_MODELS.items():
        ModelRegistry.register_model(arch, model_cls)
