import functools
import os
from typing import Any

import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.utils.quantization.quantization_utils import \
    qwix_quantize_nnx_model

logger = init_logger(__name__)


def _get_model_architecture(config: PretrainedConfig) -> nnx.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.
    _MODEL_REGISTRY = {}

    from tpu_commons.models.jax.llama import LlamaForCausalLM
    from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM
    _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
    _MODEL_REGISTRY["Qwen2ForCausalLM"] = Qwen2ForCausalLM

    if os.getenv("NEW_MODEL_DESIGN", False):
        from tpu_commons.models.jax.recipes.llama3 import Llama3_8B

        # from tpu_commons.models.jax.recipes.llama4 import Llama4Scout
        _MODEL_REGISTRY["Llama3_8B"] = Llama3_8B
        # _MODEL_REGISTRY["Llama4Scout"] = Llama4Scout

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
    maybe_quant_dtype = vllm_config.additional_config.get("quantization",
                                                          {}).get(
                                                              "dtype", None)
    maybe_quant_rules_files = vllm_config.additional_config.get(
        "quantization", {}).get("rules_file", None)
    maybe_kv_cache_quant_dtype = vllm_config.additional_config.get(
        "quantization", {}).get("kv_quant_dtype", None)
    if maybe_quant_dtype or maybe_quant_rules_files:
        # NOTE: it's REALLY important this is jitted, or else you'll run into hanging
        block_size = vllm_config.cache_config.block_size
        model_config = vllm_config.model_config
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_total_num_kv_heads()
        model = nnx.jit(
            qwix_quantize_nnx_model,
            static_argnames=(
                "quant_dtype",
                "mesh",
                "num_hidden_layers",
                "kv_cache_block_size",
                "kv_cache_num_combined_kv_heads",
                "kv_cache_head_size",
                "kv_cache_quant_dtype",
                "rules_file_path",
            ))(model,
               maybe_quant_dtype,
               rng,
               mesh,
               vllm_config.model_config.hf_config.num_hidden_layers,
               block_size,
               num_kv_heads,
               head_size,
               maybe_kv_cache_quant_dtype,
               rules_file_path=maybe_quant_rules_files)
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
    if os.getenv("NEW_MODEL_DESIGN", False):
        jit_model = _get_common_model(model_class, vllm_config, rng, mesh)
    else:
        jit_model = _get_nnx_model(model_class, vllm_config, rng, mesh)

    kv_cache_sharding = NamedSharding(mesh, PartitionSpec(None, None, "model"))
    hidden_states_sharding = NamedSharding(mesh, PartitionSpec(None,
                                                               None))  # (T, D)

    # For performance consideration, refer to:
    # https://flax.readthedocs.io/en/latest/guides/performance.html
    graphdef, state = nnx.split(jit_model)

    @functools.partial(
        jax.jit,
        out_shardings=(
            kv_cache_sharding,
            hidden_states_sharding,
        ),
        donate_argnums=2,  # 0 is graphdef, 1 is state, 2 is kv_cache
    )
    def run_model(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        return model(*args)

    logits_sharding = NamedSharding(mesh, PartitionSpec(None, "model"))

    @functools.partial(
        jax.jit,
        out_shardings=(logits_sharding),
    )
    def run_compute_logits(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        return model.compute_logits(*args)

    model_fn = functools.partial(run_model, graphdef, state)
    compute_logits_fn = functools.partial(run_compute_logits, graphdef, state)
    return model_fn, compute_logits_fn


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
    compute_logits_fn = functools.partial(model.jit_compute_logits_func(),
                                          params)
    return model_fn, compute_logits_fn


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
