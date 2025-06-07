import flax.linen as nn
from flax.typing import PRNGKey
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import VllmConfig


def _get_model_architecture(config: PretrainedConfig) -> nn.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.
    from tpu_commons.models.jax.llama import LlamaForCausalLM

    _MODEL_REGISTRY = {
        "LlamaForCausalLM": LlamaForCausalLM,
    }

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(
    vllm_config: VllmConfig,
    rng: PRNGKey,
    mesh: Mesh,
) -> nn.Module:
    model_class = _get_model_architecture(vllm_config.model_config.hf_config)
    model = model_class(
        vllm_config=vllm_config,
        rng=rng,
        mesh=mesh,
    )
    params = model.load_weights(vllm_config.model_config.model)
    return model, params

def get_vllm_model(
    vllm_config: VllmConfig,
    rng: PRNGKey,
    mesh: Mesh,
):
    from tpu_commons.models.vllm.vllm_model_wrapper import VllmModelWrapper

    model = VllmModelWrapper(
        vllm_config=vllm_config,
        rng=rng,
        mesh=mesh,
    )
    params = model.load_weights()
    return model, params