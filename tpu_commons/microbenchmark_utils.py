
import bisect
import json
import logging
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from tpu_commons.logger import init_logger
import warnings
from tpu_commons.models.jax.model_loader import _get_model_architecture
import os
import functools
from tpu_commons.models.jax.model_loader import _apply_qwix_quantization
from vllm_config_utils import VllmConfig
# from vllm_config_utils_utils import ModelConfig
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     # Import the class; all warnings will be suppressed
#     from microbenchmark_utils import ModelConfig
logger = init_logger(__name__)
power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens
@dataclass
class Sampler:
    type: str
    std: float = None
    def generate_samples(self, shape: Tuple[int], fill_val: Any) -> np.array:
        if self.type.lower() == "fixed":
            return np.full(shape, fill_val)
        elif self.type.lower() == "normal":
            return np.random.normal(loc=0.0, scale=self.std, size=shape)

def _get_nnx_model(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:
    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        # Create a sharded model with random inited weights.
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
            jit_model = _apply_qwix_quantization(vllm_config, jit_model, rng,
                                                 mesh)
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
        model = nnx.eval_shape(lambda: model_class(vllm_config, rng, mesh))
        model.load_weights(rng)
        # Although the created model can already work, we still need to jit
        # the model creation again, otherwise the model forward will have
        # non-trivial overhead in PjitFunction.
        @nnx.jit(donate_argnums=(0, ))
        def create_jit_model(model):
            state = nnx.state(model)
            nnx.update(model, state)
            model = _apply_qwix_quantization(vllm_config, model, rng, mesh)
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
    kv_cache_sharding = NamedSharding(mesh, PartitionSpec())  # replicated
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
    return model_fn, compute_logits_fn, get_multimodal_embeddings_fn, get_input_embeddings_fn, state, jit_model, 


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