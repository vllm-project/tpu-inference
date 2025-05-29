import flax.linen as nn
from flax.typing import PRNGKey
from jax.sharding import Mesh
from transformers import PretrainedConfig

from tpu_commons.models.jax.config import CacheConfig, LoRAConfig, ModelConfig
from tpu_commons.models.jax.quantization_config import get_quantization_config


def _get_model_architecture(config: PretrainedConfig) -> nn.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.
    from tpu_commons.models.jax.llama import LlamaForCausalLM

    _MODEL_REGISTRY = {
        "LlamaForCausalLM": LlamaForCausalLM,
        "Llama4ForConditionalGeneration": LlamaForCausalLM,
    }

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(
    model_config: ModelConfig,
    rng: PRNGKey,
    mesh: Mesh,
    lora_config: LoRAConfig,
    cache_config: CacheConfig,
) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    quantization_config = get_quantization_config(model_config)
    model = model_class(
        config=model_config.hf_config,
        rng=rng,
        dtype=model_config.dtype,
        mesh=mesh,
        lora_config=lora_config,
        cache_config=cache_config,
        quantization_config=quantization_config,
    )
    params = None
    if model_config.load_format != "dummy":
        if quantization_config is not None:
            params = model.load_quant_weights(model_config.model,
                                              quantization_config)
        else:
            params = model.load_weights(model_config.model)
    return model, params
