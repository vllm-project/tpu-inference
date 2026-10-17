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
"""Explicit configuration and load guards for the native Gemma 4 W4A16 path."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class W4A16Kernel(str, Enum):
    JAX = "jax"


@dataclass(frozen=True)
class W4A16Options:
    kernel: W4A16Kernel = W4A16Kernel.JAX

    @classmethod
    def from_config(cls, additional_config: Mapping[str, Any]):
        value = additional_config["jax_w4a16"]
        if not isinstance(value, dict) or set(value) - {"kernel"}:
            raise ValueError(
                "jax_w4a16 must be an object containing only kernel")
        try:
            return cls(kernel=W4A16Kernel(value.get("kernel", "jax")))
        except (TypeError, ValueError) as error:
            raise ValueError("jax_w4a16 kernel must be 'jax'") from error


def validate_w4a16_request(vllm_config, *, impl: str,
                           is_draft_model: bool) -> W4A16Options:
    options = W4A16Options.from_config(vllm_config.additional_config)
    if impl != "flax_nnx":
        raise ValueError(
            "jax_w4a16 requires explicit MODEL_IMPL_TYPE=flax_nnx")
    if is_draft_model or vllm_config.speculative_config is not None:
        raise ValueError("jax_w4a16 does not support speculative decoding")
    if vllm_config.lora_config is not None:
        raise ValueError("jax_w4a16 does not support LoRA")
    model = vllm_config.model_config
    if getattr(model.hf_config, "architectures",
               None) != ["Gemma4ForCausalLM"]:
        raise ValueError("jax_w4a16 requires the Gemma4ForCausalLM text path")
    if model.quantization != "compressed-tensors":
        raise ValueError("jax_w4a16 requires compressed-tensors quantization")
    dtype = getattr(model.dtype, "__name__", str(model.dtype).split(".")[-1])
    if dtype != "bfloat16":
        raise ValueError("jax_w4a16 requires BF16 activations")
    parallel = vllm_config.parallel_config
    if (parallel.tensor_parallel_size != 8 or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1):
        raise ValueError("jax_w4a16 requires TP8, DP1 and PP1")
    text_config = getattr(model.hf_config, "text_config", model.hf_config)
    if getattr(text_config, "enable_moe_block", False):
        raise ValueError("jax_w4a16 does not support MoE")
    quant = getattr(model.hf_config, "quantization_config", {})
    if quant.get("format") != "pack-quantized":
        raise ValueError("jax_w4a16 requires pack-quantized weights")
    groups = quant.get("config_groups", {})
    if not groups:
        raise ValueError("jax_w4a16 requires quantization groups")
    for group in groups.values():
        weight = group.get("weights", {})
        if (group.get("input_activations") is not None
                or group.get("output_activations") is not None
                or weight.get("num_bits") != 4 or weight.get("type") != "int"
                or weight.get("strategy") != "group"
                or weight.get("group_size") != 32
                or weight.get("symmetric") is not True
                or weight.get("dynamic", False)
                or weight.get("actorder") is not None):
            raise ValueError(
                "jax_w4a16 requires static symmetric group-32 W4A16")
    return options


def get_w4a16_quantization_config(vllm_config, *, is_draft_model: bool):
    from tpu_inference import envs
    from tpu_inference.layers.jax.quantization.compressed_tensors_w4a16 import \
        W4A16CompressedTensorsConfig

    options = validate_w4a16_request(vllm_config,
                                     impl=envs.MODEL_IMPL_TYPE,
                                     is_draft_model=is_draft_model)
    if options.kernel is not W4A16Kernel.JAX:
        raise ValueError("Unsupported W4A16 kernel")
    return W4A16CompressedTensorsConfig(
        vllm_config.model_config.hf_config.quantization_config)
