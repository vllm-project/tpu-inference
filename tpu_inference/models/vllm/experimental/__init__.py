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
}


def register_models():
    """Override vLLM's built-in model classes with TPU-specific ones.

    Called from the ``vllm.general_plugins`` entrypoint so it runs at vLLM
    startup, before any model is resolved from its architecture name.
    """
    import os

    from vllm import ModelRegistry

    models = dict(_TPU_VLLM_MODELS)
    # Experiment only: opt back into the native vLLM Qwen2-MoE implementation
    # that vllm-project/vllm#66827 replaced with the Transformers backend.
    if os.environ.get("TPU_INFERENCE_NATIVE_QWEN2_MOE") == "1":
        models["Qwen2MoeForCausalLM"] = (
            "tpu_inference.models.vllm.experimental.qwen2_moe:"
            "Qwen2MoeForCausalLM")
    for arch, model_cls in models.items():
        ModelRegistry.register_model(arch, model_cls)
