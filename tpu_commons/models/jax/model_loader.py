import functools
import os

import flax.linen as nn
import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_commons.models.jax.utils.param_overview import get_parameter_overview


def _get_model_architecture(config: PretrainedConfig) -> nn.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.

    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    if impl == "flax_nn":
        from tpu_commons.models.jax.llama_nn import LlamaForCausalLM
    elif impl == "flax_nnx":
        from tpu_commons.models.jax.llama import LlamaForCausalLM
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")

    _MODEL_REGISTRY = {
        "LlamaForCausalLM": LlamaForCausalLM,
        # "Qwen2ForCausalLM": Qwen2ForCausalLM,
    }

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_nn_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
):
    model_class = _get_model_architecture(vllm_config.model_config.hf_config)
    model = model_class(
        vllm_config=vllm_config,
        rng=rng,
        mesh=mesh,
    )
    params = model.load_weights(vllm_config.model_config.model)
    if os.getenv("INSPECT_MODEL", False):
        print(
            "Model params:\n%s",
            get_parameter_overview(params, include_stats="sharding"),
        )

    kv_cache_sharding = NamedSharding(mesh, PartitionSpec("model"))
    outputs_sharding = NamedSharding(mesh, PartitionSpec(None))
    logits_cache_sharding = NamedSharding(mesh, PartitionSpec(None))
    jit_model = jax.jit(
        model.apply,
        out_shardings=(
            kv_cache_sharding,
            outputs_sharding,
            logits_cache_sharding,
        ),
        static_argnums=(1, 2),
        donate_argnums=3,
    )
    model_fn = functools.partial(jit_model, params)
    return model_fn


def get_nnx_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
):
    model_class = _get_model_architecture(vllm_config.model_config.hf_config)

    @nnx.jit
    def create_sharded_model():
        model = model_class(vllm_config, rng, mesh)
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

    # Create the model instance with inited sharded weights
    # TODO(xiang): create abstract model without weights allocated
    # https://flax.readthedocs.io/en/latest/guides/surgery.html#creating-an-abstract-model-or-state-without-memory-allocation
    with mesh:
        jit_model = create_sharded_model()

    jit_model.load_weights()

    kv_cache_sharding = NamedSharding(mesh, PartitionSpec("model"))
    outputs_sharding = NamedSharding(mesh, PartitionSpec(None))
    logits_cache_sharding = NamedSharding(mesh, PartitionSpec(None))

    # For performance consideration, refer to:
    # https://flax.readthedocs.io/en/latest/guides/performance.html
    graphdef, state = nnx.split(jit_model)

    @functools.partial(
        jax.jit,
        out_shardings=(
            kv_cache_sharding,
            outputs_sharding,
            logits_cache_sharding,
        ),
        static_argnums=(2, 3),
        donate_argnums=4,
    )
    def run_model(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        return model(*args)

    def model_fn(graphdef, state, *args):
        with mesh:
            return run_model(graphdef, state, *args)

    model_fn = functools.partial(model_fn, graphdef, state)
    return model_fn


def get_vllm_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
):
    from tpu_commons.models.vllm.vllm_model_wrapper import VllmModelWrapper

    model = VllmModelWrapper(
        vllm_config=vllm_config,
        rng=rng,
        mesh=mesh,
    )
    params = model.load_weights()

    jit_model = model.jit_step_func()
    model_fn = functools.partial(jit_model, params)
    return model_fn


def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nn.Module:
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    if impl == "flax_nn":
        return get_nn_model(vllm_config, rng, mesh)
    elif impl == "flax_nnx":
        return get_nnx_model(vllm_config, rng, mesh)
    elif impl == "vllm":
        return get_vllm_model(vllm_config, rng, mesh)
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")
