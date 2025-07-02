import functools
import os
from typing import Any

import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.model import Model

logger = init_logger(__name__)


def _get_model_architecture(config: PretrainedConfig) -> nnx.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.
    _MODEL_REGISTRY = {}

    # TODO(xiang): unify the model interface
    if os.getenv("USE_JAX_V1", True):
        from tpu_commons.models.jax.llama_v1 import LlamaForCausalLM
        from tpu_commons.models.jax.qwen2_v1 import Qwen2ForCausalLM
    else:
        from tpu_commons.models.jax.llama import LlamaForCausalLM
        from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM
    _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
    _MODEL_REGISTRY["Qwen2ForCausalLM"] = Qwen2ForCausalLM

    if os.getenv("NEW_MODEL_DESIGN", False):
        from tpu_commons.models.jax.recipes.llama3 import Llama3_8B
        from tpu_commons.models.jax.recipes.llama4 import Llama4Scout
        _MODEL_REGISTRY["Llama3_8B"] = Llama3_8B
        _MODEL_REGISTRY["Llama4Scout"] = Llama4Scout

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def _get_common_model(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:
    model = model_class(vllm_config, rng, mesh)
    model.load_weights(model)
    return model


def _get_nnx_model(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:
    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        # Create a sharded model with random inited weights.
        @nnx.jit
        def create_sharded_model():
            model = model_class(vllm_config, rng, mesh)
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return model

        with mesh:
            jit_model = create_sharded_model()
    else:
        # We first create an abstract model without allocating any weights,
        # then fill in its weigths during load_weights from HF.
        # This shows 3 advantages than the normal way:
        # 1. The model weights will only be allocated once. Otherwise the normal way
        #    will random-init the model weights first, then load the real weights.
        #    The two pass weights allocation causes model loading slow.
        # 2. The model loading won't be OOM. Otherwise the normal way will hold
        #    a full model weights after random-init, then duplicate a layer during
        #    the load_weights. This would be easy to OOM if the layer is super large.
        # 3. The model architecture definition won't need to worry about the sharding.
        #    The sharding definition is taken over by the load_weights instead.
        model = nnx.eval_shape(lambda: model_class(vllm_config, rng, mesh))
        model.load_weights(rng)
        # Although the created model can already work, we still need to jit
        # the model creation again, otherwise the model forward will have
        # non-trivial overhead in PjitFunction.
        @nnx.jit(donate_argnums=(0, ))
        def create_jit_model(model):
            state = nnx.state(model)
            nnx.update(model, state)
            return model

        with mesh:
            jit_model = create_jit_model(model)
    return jit_model


def get_flax_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:
    model_class = _get_model_architecture(vllm_config.model_config.hf_config)
    if issubclass(model_class,
                  Model):  # TODO: Get this to work for nnx.eval_shape.
        jit_model = _get_common_model(model_class, vllm_config, rng, mesh)
    else:
        jit_model = _get_nnx_model(model_class, vllm_config, rng, mesh)

    kv_cache_sharding = NamedSharding(mesh, PartitionSpec(None, None, "model"))
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

    model_fn = functools.partial(run_model, graphdef, state)
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
) -> Any:
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    logger.info(f"Loading model, implementation type={impl}")
    if impl == "flax_nnx":
        return get_flax_model(vllm_config, rng, mesh)
    elif impl == "vllm":
        return get_vllm_model(vllm_config, rng, mesh)
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")
