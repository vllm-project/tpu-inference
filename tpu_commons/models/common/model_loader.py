import functools
import os
from typing import Any, Optional

import jax
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.ops.mappings import j2t_dtype
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.utils import supports_kw

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.utils.quantization.quantization_utils import (
    apply_qwix_on_abstract_model, apply_qwix_quantization,
    load_random_weights_into_qwix_abstract_model)

logger = init_logger(__name__)

_MODEL_REGISTRY = {}


class UnsupportedArchitectureError(ValueError):
    """Raised when a model architecture is not supported in the registry."""
    pass


def _get_model_architecture(config: PretrainedConfig) -> nnx.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.

    from tpu_commons.models.jax.deepseek_v3 import DeepSeekV3
    from tpu_commons.models.jax.llama4 import Llama4ForCausalLM
    from tpu_commons.models.jax.llama_eagle3 import EagleLlama3ForCausalLM
    from tpu_commons.models.jax.phi3 import Phi3ForCausalLM
    from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM
    from tpu_commons.models.jax.qwen2_5_vl import \
        Qwen2_5_VLForConditionalGeneration
    from tpu_commons.models.jax.qwen3 import Qwen3ForCausalLM

    if os.getenv("NEW_MODEL_DESIGN", False):
        from tpu_commons.experimental.llama3_jax_stashed import \
            LlamaForCausalLM
    else:
        from tpu_commons.models.jax.llama3 import LlamaForCausalLM

    _MODEL_REGISTRY["Llama4ForCausalLM"] = Llama4ForCausalLM
    _MODEL_REGISTRY["DeepseekV3ForCausalLM"] = DeepSeekV3
    _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
    _MODEL_REGISTRY["Qwen2ForCausalLM"] = Qwen2ForCausalLM
    _MODEL_REGISTRY["Qwen3ForCausalLM"] = Qwen3ForCausalLM
    _MODEL_REGISTRY[
        "Qwen2_5_VLForConditionalGeneration"] = Qwen2_5_VLForConditionalGeneration
    _MODEL_REGISTRY["Phi3ForCausalLM"] = Phi3ForCausalLM
    _MODEL_REGISTRY["Eagle3LlamaForCausalLM"] = EagleLlama3ForCausalLM

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise UnsupportedArchitectureError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def _get_nnx_model(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:

    def create_abstract_model() -> nnx.Module:
        """
        Helper class to create an abstract model for `nnx.eval_shape`.

        Returns:
            An abstract model function.
        """
        return model_class(vllm_config, rng, mesh)

    @nnx.jit(donate_argnums=(0, ),
             static_argnames=('use_qwix_on_abstract_model', ))
    def create_jit_model(
            model: nnx.Module,
            use_qwix_on_abstract_model: bool = False) -> nnx.Module:
        """
        Create a jit model.

        Args:
            model: The model to jit.
            use_qwix_on_abstract_model: Whether to apply Qwix on the abstract model.

        Returns:
            The jitted model.
        """
        state = nnx.state(model)
        nnx.update(model, state)
        if not use_qwix_on_abstract_model:
            # NOTE: if Qwix is not configured, this will be a no-op
            model = apply_qwix_quantization(vllm_config,
                                            model,
                                            rng,
                                            mesh,
                                            apply_to_abstract_model=False)
        return model

    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        # Create a sharded model with random inited weights.
        # TODO: currently Qwen2ForCausalLM is using legacy model implementation
        # will merge the random init logic when all model are migrated to new model implementation

        # Handle the case where we want to load in random weights to a Qwix-quantized model.  Here, we
        # need to run an abstract pass for Qwix first and then load in the random weights.
        if apply_qwix_on_abstract_model(vllm_config):
            abstract_model_fn = apply_qwix_quantization(
                vllm_config,
                create_abstract_model,
                rng,
                mesh,
                apply_to_abstract_model=True)

            model = nnx.eval_shape(abstract_model_fn)
            quantization_config = vllm_config.model_config.hf_config.quantization_config if hasattr(
                vllm_config.model_config.hf_config,
                "quantization_config") else {}
            load_random_weights_into_qwix_abstract_model(
                rng, model, mesh, quantization_config)
            with mesh:
                jit_model = create_jit_model(model,
                                             use_qwix_on_abstract_model=True)
            return jit_model

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
            # In this case, we are applying Qwix quantization to the true, concrete model
            jit_model = apply_qwix_quantization(vllm_config,
                                                jit_model,
                                                rng,
                                                mesh,
                                                apply_to_abstract_model=False)
            if hasattr(jit_model, 'initialize_cache'):
                jit_model.initialize_cache()
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
        abstract_model_fn = create_abstract_model
        # NOTE: only one of the abstract (this) or or concrete Qwix quantization paths should
        # be taken
        if should_apply_qwix_on_abstract_model := apply_qwix_on_abstract_model(
                vllm_config):
            # NOTE: if Qwix is not configured, this will return `create_abstract_model` and
            # thus be a no-op
            abstract_model_fn = apply_qwix_quantization(
                vllm_config,
                create_abstract_model,
                rng,
                mesh,
                apply_to_abstract_model=True)
        model = nnx.eval_shape(abstract_model_fn)
        # Although the created model can already work, we still need to jit
        # the model creation again, otherwise the model forward will have
        # non-trivial overhead in PjitFunction.
        with mesh:
            model.load_weights(rng)
            jit_model = create_jit_model(
                model,
                use_qwix_on_abstract_model=should_apply_qwix_on_abstract_model)
    return jit_model


# TODO(pooyam): We need to refactor this. This is returning a bunch of functions that do not work with all models and this is not very easy to see from the code.
def get_flax_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    is_draft_model: bool = False,
) -> nnx.Module:
    if is_draft_model:
        model_class = _get_model_architecture(
            vllm_config.speculative_config.draft_model_config.hf_config)
    else:
        model_class = _get_model_architecture(
            vllm_config.model_config.hf_config)
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
            hidden_states_sharding,  # aux hidden states
        ),
        donate_argnums=2,  # 0 is graphdef, 1 is state, 2 is kv_cache
        static_argnums=6,  #6 is layer_name_to_kvcache_index
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
        hidden_state, *_ = args
        return model.compute_logits(hidden_state)

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

    # For models that want to work with EAGLE-3 speculative decoding
    @functools.partial(
        jax.jit,
        out_shardings=(logits_sharding),
    )
    def combine_hidden_states(graphdef, state, hidden_states):
        model = nnx.merge(graphdef, state)
        return model.combine_hidden_states(hidden_states)

    model_fn = functools.partial(run_model, graphdef)
    compute_logits_fn = functools.partial(run_compute_logits, graphdef)
    get_multimodal_embeddings_fn = functools.partial(
        run_get_multimodal_embeddings, graphdef)
    get_input_embeddings_fn = functools.partial(run_get_input_embeddings,
                                                graphdef)
    lora_manager, model = None, None
    combine_hidden_states_fn = functools.partial(combine_hidden_states,
                                                 graphdef)

    return model_fn, compute_logits_fn, combine_hidden_states_fn, get_multimodal_embeddings_fn, get_input_embeddings_fn, state, lora_manager, model


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
    params, lora_manager = model.load_weights()

    jit_model = model.jit_step_func()
    compute_logits_fn = model.jit_compute_logits_func()
    # the model needs to be returned because lora weights are neither torch.nn.parameter nor torch.nn.buffer. After we load the lora weights and set it to the torch.nn.Module, we can shard it and move it to TPU.
    combine_hidden_states_fn = None
    return jit_model, compute_logits_fn, combine_hidden_states_fn, None, None, params, lora_manager, model


def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    is_draft_model: bool = False,
) -> Any:
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    logger.info(f"Loading model with MODEL_IMPL_TYPE={impl}")

    if impl == "flax_nnx":
        try:
            # Try to load the flax model first
            return get_flax_model(vllm_config, rng, mesh, is_draft_model)
        except UnsupportedArchitectureError as e:
            # Convert the error message to a string to check its contents
            error_msg = str(e)

            logger.warning(f"Flax model failed with: '{error_msg}'. "
                           "Falling back to vLLM implementation.")
            # Fall back to the vLLM model and updating the dtype accordingly
            vllm_config.model_config.dtype = j2t_dtype(
                vllm_config.model_config.dtype.dtype)
            return get_vllm_model(vllm_config, rng, mesh)
    elif impl == "vllm":
        return get_vllm_model(vllm_config, rng, mesh)
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")


def _validate_model_interface(model: Any) -> None:
    """Validates that the model class has the required methods and signatures.

    A valid model must have:
    - An __init__ method that accepts a 'vllm_config' keyword argument.
    - A __call__ method that accepts 'kv_caches', 'input_ids', and
      'attention_metadata' keyword arguments.

    Args:
        model: The model class to validate.

    Raises:
        TypeError: If the model does not meet the interface requirements.
    """
    # Check for __init__ with vllm_config
    model_init = getattr(model, "__init__", None)
    if not callable(model_init):
        raise TypeError(
            f"Model {model.__name__} must have an __init__ method.")

    if not supports_kw(model_init, "vllm_config"):
        raise TypeError(
            f"Model {model.__name__} __init__ method must accept a "
            "'vllm_config' keyword argument.")

    # Check for __call__ with required arguments
    model_call = getattr(model, "__call__", None)
    # A class object is always callable (it produces an instance).
    # We need to check if the class _explicitly_ defines a __call__ method for its
    # instance, which is different from `type.__call__`.
    has_defined_call = False
    if isinstance(model, type):
        if any("__call__" in C.__dict__ for C in model.__mro__):
            has_defined_call = True
    elif callable(model_call):
        # For an instance, a simple callable check is sufficient.
        has_defined_call = True

    if not has_defined_call:
        raise TypeError(f"Model {model.__name__} must have a __call__ method.")

    required_call_args = ("kv_caches", "input_ids", "attention_metadata")
    missing_args = tuple(arg for arg in required_call_args
                         if not supports_kw(model_call, arg))

    if missing_args:
        raise TypeError(
            f"Model {model.__name__} __call__ method is missing required "
            f"keyword arguments: {missing_args}")


def register_model(arch: str, model: Any) -> None:
    """
    Registers a model class for a given architecture name.

    This function registers the model with both the tpu_commons registry
    and the vLLM registry. For vLLM, it creates a compatible wrapper
    around the JAX model.

    Args:
        arch: The name of the architecture (e.g., "LlamaForCausalLM").
        model: The JAX model class to register (e.g., a flax.nnx.Module).
    """
    _validate_model_interface(model)

    # Register with tpu_commons registry for the JAX backend
    _MODEL_REGISTRY[arch] = model

    # Create a vLLM-compatible wrapper for the JAX model class.
    # This wrapper inherits from the JAX model and torch.nn.Module
    # to pass vLLM's type checks. It is not meant to be instantiated
    # or executed by vLLM's PyTorch backend.
    def unimplemented_forward(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional["torch.Tensor"] = None,
    ) -> None:
        raise NotImplementedError(
            "This is a JAX model and does not implement the PyTorch forward method."
        )

    # We need a custom __init__ that only calls torch.nn.Module's init,
    # to avoid triggering JAX logic when vLLM inspects the class.
    def wrapper_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    # Dynamically create the wrapper class that is a subclass of both the
    # JAX model and torch.nn.Module.
    VllmCompatibleModel = type(
        f"VllmCompatible{model.__name__}",
        (model, torch.nn.Module),
        {
            "__init__": wrapper_init,
            "forward": unimplemented_forward,
            # Prevent vLLM from trying to load weights into this dummy class.
            "load_weights": lambda self, *args, **kwargs: None,
        })

    # Register the wrapped model with vLLM's registry.
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(arch, VllmCompatibleModel)
    logger.info(
        f"Registered JAX model {arch} with tpu_commons and vLLM registries.")
