import functools
import os
from typing import Any

import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.utils.quantization.quantization_utils import (
    quantization_config_file_path_to_dict, qwix_quantize_nnx_model)

logger = init_logger(__name__)


def _eval_qwix_quantization(vllm_config: VllmConfig, model: nnx.Module,
                            rng: jax.Array, mesh: Mesh) -> nnx.Module:
    """
    Will apply quantization if a valid quantization config with Qwix rules is provided.  See README
    for more details.

    Args:
        vllm_config: the vllm config
        model: the model to quantize
        rng: the random number generator to use
        mesh: the mesh to use

    Returns:
        the potentially quantized model
    """
    # NOTE: we expect the value of "quantization" to be the name of a file in `tpu_commons/models/jax/utils/quantization/configs`
    # if given
    qwix_config = None
    if vllm_config.additional_config.get("quantization"):
        quantization_config = quantization_config_file_path_to_dict(
            vllm_config.additional_config["quantization"])
        qwix_config = quantization_config.get("qwix").get("rules")
    if qwix_config:
        block_size = vllm_config.cache_config.block_size
        model_config = vllm_config.model_config
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_total_num_kv_heads()
        # NOTE: it's REALLY important this is jitted, or else you'll run into hanging
        qwix_quantize_fn_for_eval = functools.partial(
            qwix_quantize_nnx_model,
            qwix_config=qwix_config,
            mesh=mesh,
            num_hidden_layers=vllm_config.model_config.hf_config.
            num_hidden_layers,
            kv_cache_block_size=block_size,
            kv_cache_num_kv_heads=num_kv_heads,
            kv_cache_head_size=head_size)

        # Now, nnx.eval_shape is only passed the function and the arguments
        # that should be traced (the model and rng).
        model = nnx.eval_shape(
            qwix_quantize_fn_for_eval,
            model=model,
            rng=rng,
        )
        # model = nnx.eval_shape(qwix_quantize_nnx_model_with_config,
        #                 static_argnames=(
        #                     "mesh",
        #                     "num_hidden_layers",
        #                     "kv_cache_block_size",
        #                     "kv_cache_num_kv_heads",
        #                     "kv_cache_head_size",
        #                 ))(model=model,
        #                    rng=rng,
        #                    mesh=mesh,
        #                    num_hidden_layers=vllm_config.model_config.
        #                    hf_config.num_hidden_layers,
        #                    kv_cache_block_size=block_size,
        #                    kv_cache_num_kv_heads=num_kv_heads,
        #                    kv_cache_head_size=head_size)

    return model


def _apply_qwix_quantization(vllm_config: VllmConfig, model: nnx.Module,
                             rng: jax.Array, mesh: Mesh) -> nnx.Module:
    """
    Will apply quantization if a valid quantization config with Qwix rules is provided.  See README
    for more details.

    Args:
        vllm_config: the vllm config
        model: the model to quantize
        rng: the random number generator to use
        mesh: the mesh to use

    Returns:
        the potentially quantized model
    """
    # NOTE: we expect the value of "quantization" to be the name of a file in `tpu_commons/models/jax/utils/quantization/configs`
    # if given
    qwix_config = None
    if vllm_config.additional_config.get("quantization"):
        quantization_config = quantization_config_file_path_to_dict(
            vllm_config.additional_config["quantization"])
        qwix_config = quantization_config.get("qwix").get("rules")
    if qwix_config:
        block_size = vllm_config.cache_config.block_size
        model_config = vllm_config.model_config
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_total_num_kv_heads()
        # NOTE: it's REALLY important this is jitted, or else you'll run into hanging
        qwix_quantize_nnx_model_with_config = functools.partial(
            qwix_quantize_nnx_model, qwix_config=qwix_config)
        model = nnx.jit(qwix_quantize_nnx_model_with_config,
                        donate_argnums=(0, ),
                        static_argnames=(
                            "mesh",
                            "num_hidden_layers",
                            "kv_cache_block_size",
                            "kv_cache_num_kv_heads",
                            "kv_cache_head_size",
                        ))(model=model,
                           rng=rng,
                           mesh=mesh,
                           num_hidden_layers=vllm_config.model_config.
                           hf_config.num_hidden_layers,
                           kv_cache_block_size=block_size,
                           kv_cache_num_kv_heads=num_kv_heads,
                           kv_cache_head_size=head_size)

    return model


def _get_model_architecture(config: PretrainedConfig) -> nnx.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.
    _MODEL_REGISTRY = {}

    if os.getenv("NEW_MODEL_DESIGN", False):
        from tpu_commons.experimental.llama3_jax_stashed import \
            LlamaForCausalLM
        from tpu_commons.models.jax.deepseek_v3 import DeepSeekV3
        from tpu_commons.models.jax.llama4 import Llama4ForCausalLM
        _MODEL_REGISTRY["DeepSeekV3"] = DeepSeekV3
        _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
        _MODEL_REGISTRY["Llama4ForCausalLM"] = Llama4ForCausalLM
    else:
        from tpu_commons.models.jax.llama3 import LlamaForCausalLM
        from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM
        from tpu_commons.models.jax.qwen2_5_vl import \
            Qwen2_5_VLForConditionalGeneration
        _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
        _MODEL_REGISTRY["Qwen2ForCausalLM"] = Qwen2ForCausalLM
        _MODEL_REGISTRY[
            "Qwen2_5_VLForConditionalGeneration"] = Qwen2_5_VLForConditionalGeneration

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def _get_nnx_model(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:
    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        # Create a sharded model with random inited weights.
        if not os.getenv("NEW_MODEL_DESIGN", False):
            # TODO: currently Qwen2ForCausalLM is using legacy model implementation
            # will merge the random init logic when all model are migrated to new model implementation
            @nnx.jit
            def create_sharded_model():
                model = model_class(vllm_config, rng, mesh)
                state = nnx.state(model)
                pspecs = nnx.get_partition_spec(state)
                sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
                nnx.update(model, sharded_state)
                # NOTE: we don't support quantization for the old Qwen2ForCausalLM implementation
                return model

            with mesh:
                jit_model = create_sharded_model()
        else:

            with mesh:
                model = model_class.create_model_with_random_weights(
                    vllm_config, rng, mesh)
                jit_model = _apply_qwix_quantization(vllm_config, model, rng,
                                                     mesh)
    else:
        # We first create an abstract model without allocating any weights,
        # then fill in its weigths during load_weights from HF.
        # This shows 2 advantages than the normal way:
        # 1. The model weights will only be allocated once. Otherwise the normal way
        #    will random-init the model weights first, then load the real weights.
        #    The two pass weights allocation causes model loading slow.
        # 2. The model loading won't be OOM. Otherwise the normal way will hold
        #    a full model weights after random-init, then duplicate a layer during
        #    the load_weights. This would be easy to OOM if the layer is super large.
        if os.getenv("NEW_MODEL_DESIGN", False):
            model = model_class.create_model_for_checkpoint_loading(
                vllm_config, rng, mesh)
        else:
            model = nnx.eval_shape(lambda: model_class(vllm_config, rng, mesh))
        model = _eval_qwix_quantization(vllm_config, model, rng, mesh)
        model.load_weights(rng)
        # Although the created model can already work, we still need to jit
        # the model creation again, otherwise the model forward will have
        # non-trivial overhead in PjitFunction.
        @nnx.jit(donate_argnums=(0, ))
        def create_jit_model(model):
            state = nnx.state(model)
            nnx.update(model, state)
            # model = _apply_qwix_quantization(vllm_config, model, rng, mesh)
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

    # Multi-modal support only
    # This function calculates the image token's embeddings by VIT
    @functools.partial(jax.jit,
                       out_shardings=(logits_sharding),
                       static_argnames=['image_grid_thw'])
    def run_get_multimodal_embeddings(graphdef, state, image_grid_thw,
                                      **kwargs):
        model = nnx.merge(graphdef, state)
        return model.get_multimodal_embeddings(image_grid_thw, **kwargs)

    # This function will calculates the embeddings of input texts and then merge with the image embeddings
    @functools.partial(
        jax.jit,
        out_shardings=(logits_sharding),
    )
    def run_get_input_embeddings(graphdef, state, *args, **kwargs):
        model = nnx.merge(graphdef, state)
        return model.get_input_embeddings(*args, **kwargs)

    model_fn = functools.partial(run_model, graphdef)
    compute_logits_fn = functools.partial(run_compute_logits, graphdef)
    get_multimodal_embeddings_fn = functools.partial(
        run_get_multimodal_embeddings, graphdef)
    get_input_embeddings_fn = functools.partial(run_get_input_embeddings,
                                                graphdef)
    return model_fn, compute_logits_fn, get_multimodal_embeddings_fn, get_input_embeddings_fn, state


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
    compute_logits_fn = model.jit_compute_logits_func()
    return jit_model, compute_logits_fn, None, None, params


def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> Any:
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    logger.info(f"Loading model with MODEL_IMPL_TYPE={impl}")
    if impl == "flax_nnx":
        return get_flax_model(vllm_config, rng, mesh)
    elif impl == "vllm":
        return get_vllm_model(vllm_config, rng, mesh)
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")
