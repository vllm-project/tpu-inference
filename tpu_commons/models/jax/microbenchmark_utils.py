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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Import the class; all warnings will be suppressed
    from vllm.config import ModelConfig


logger = init_logger(__name__)

power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens


@dataclass
class VllmConfig():
    additional_config: Mapping[str, Any] = field(default_factory=dict)
    # Set default max_model_len to turn off warnings.
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(max_model_len=1024))


@dataclass
class ModelConfig():
    max_model_len: int = 2048
    max_prefill_len: int = 1024
    prefill_batch_size: int = 1
    decode_batch_size: int = 1
    block_size: int = 16
    num_layers: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    vocab_size: int = 32000
    model: str = "llama3"
    hf_config: str = ""
    architectures: List[str] = field(default_factory=list)
    override_generation_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sampler:
    type: str
    std: float = None

    def generate_samples(self, shape: Tuple[int], fill_val: Any) -> np.array:
        if self.type.lower() == "fixed":
            return np.full(shape, fill_val)
        elif self.type.lower() == "normal":
            return np.random.normal(loc=0.0, scale=self.std, size=shape)



def get_jitted_model_creation_fn(model_class: Any, mode: Any, vllm_config: VllmConfig,
                                 mesh: Mesh, model=None) -> Any:
    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        # Create a sharded model with random inited weights.
        @nnx.jit
        def create_sharded_model(rng):
            model = model_class(vllm_config, rng, mesh)
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return model
        with mesh:
            return create_sharded_model
    else:
        model_cfg = model.cfg.model
        @nnx.jit(donate_argnums=(0, ))
        def create_jit_model(model):
            state = nnx.state(model)
            nnx.update(model, state)
            return model

        with mesh:
            jit_model = create_jit_model(model)
    




def _get_nnx_model_and_model_cfg(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    model=None,
) -> nnx.Module:
    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        # Create a sharded model with random inited weights.
        @nnx.jit
        def create_sharded_model():
            random_model = model_class(vllm_config, rng, mesh)
            state = nnx.state(random_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return random_model

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
    if os.getenv("JAX_RANDOM_WEIGHTS", False):
        model = model_class(vllm_config, rng, mesh)
    else:
        model = nnx.eval_shape(lambda: model_class(vllm_config, rng, mesh))
    import pdb
    pdb.set_trace()
    model_cfg = model.vllm_config.model_config
    jit_model = _get_nnx_model_and_model_cfg(model_class, vllm_config, rng, mesh, model)

    # model_cfg = model.cfg.model
    # @nnx.jit(donate_argnums=(0, ))
    # def create_jit_model(model):
    #     state = nnx.state(model)
    #     nnx.update(model, state)
    #     return model

    # with mesh:
    #     jit_model = create_jit_model(model)
    
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

    model_fn = functools.partial(run_model, graphdef)
    compute_logits_fn = functools.partial(run_compute_logits, graphdef)
    return model_fn, compute_logits_fn, state, model

def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> Any:
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    logger.info(f"Loading model with MODEL_IMPL_TYPE={impl}")
    if impl == "flax_nnx":
        return get_flax_model(vllm_config, rng, mesh)
    # elif impl == "vllm":
    #     return get_vllm_model(vllm_config, rng, mesh)
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")