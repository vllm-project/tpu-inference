import functools
import os

import flax.linen as nn
import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.model import Model
from tpu_commons.models.jax.utils.param_overview import get_parameter_overview
from tpu_commons.models.jax.utils.weight_utils import apply_sharding

from typing import List

logger = init_logger(__name__)


def _get_model_architecture(config: PretrainedConfig) -> nn.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.

    _MODEL_REGISTRY = {}
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()

    if impl == "flax_nn":
        from tpu_commons.models.jax.llama_nn import LlamaForCausalLM
        _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
    elif impl == "flax_nnx":
        from tpu_commons.models.jax.llama import LlamaForCausalLM
        _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
        from tpu_commons.models.jax.qwen2 import Qwen2ForCausalLM
        _MODEL_REGISTRY["Qwen2ForCausalLM"] = Qwen2ForCausalLM
        if os.getenv("NEW_MODEL_DESIGN", False):
            from tpu_commons.models.jax.recipes.llama3 import Llama3_8B
            from tpu_commons.models.jax.recipes.llama4 import Llama4Scout
            _MODEL_REGISTRY["Llama3_8B"] = Llama3_8B
            _MODEL_REGISTRY["Llama4Scout"] = Llama4Scout
        from tpu_commons.models.jax.gemma import GemmaForCausalLM
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")

    _MODEL_REGISTRY = {
        "LlamaForCausalLM": LlamaForCausalLM,
        "GemmaForCausalLM": GemmaForCausalLM,
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
    return model_fn, model


def get_nnx_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
):
    lora_config = vllm_config.additional_config.get("lora_config") if vllm_config.additional_config else None
    model_class = _get_model_architecture(vllm_config.model_config.hf_config)

    def maybe_apply_lora_to_model(base_model, mesh=None, lora_config=None):
        from qwix import lora

        """Apply LoRA to the base model."""
        if lora_config is None:

            return base_model

        # Use LoRA configuration from vllm_config
        # Default values for Llama 8B if not specified in lora_config
        rank = lora_config.get('rank', 16)
        alpha = lora_config.get('alpha', 32.0)
        module_path = lora_config.get('module_path',
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj")
        # dropout = getattr(lora_config, 'dropout', 0.0)
        # bias = getattr(lora_config, 'bias', "none")

        lora_provider = lora.LoraProvider(
            module_path=module_path,
            rank=rank,
            alpha=alpha,
            # dropout=dropout,
            # bias=bias,
        )

        model_input = base_model.get_model_input()

        lora_model = lora.apply_lora_to_model(
            base_model, lora_provider, **model_input
        )
        del model_input

        with mesh:
            state = nnx.state(lora_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(lora_model, sharded_state)
        return lora_model

    if issubclass(model_class, Model): # TODO: Get this to wrok for nnx.eval_shape.
        model = model_class(vllm_config, rng, mesh)
        model = maybe_apply_lora_to_model(model, mesh, lora_config)
        model.load_weights(model)

        # Apply LoRA if lora_config is present
        jit_model = model
    else:
        if os.getenv("JAX_RANDOM_WEIGHTS", False):
          def create_model():
              model = model_class(vllm_config, rng, mesh)
              # TODO(lancewang): Enable lora
              # model = maybe_apply_lora_to_model(model, mesh, lora_config)
              return model
          abstract_model = nnx.eval_shape(lambda: create_model())

          graph_def, abs_state = nnx.split(abstract_model)

          sharding_mappings = {
              "lm_head": (None, "model"),
              "embed.embedding": ("model", None),
              "model.layers.*.input_layernorm.scale": (None,),
              "model.layers.*.mlp.down_proj.kernel": ("model", None),
              "model.layers.*.mlp.gate_proj.kernel": (None, "model"),
              "model.layers.*.mlp.up_proj.kernel": (None, "model"),
              "model.layers.*.post_attention_layernorm.scale": (None,),
              "model.layers.*.self_attn.k_proj.kernel": ("model", None, None),
              "model.layers.*.self_attn.o_proj.kernel": ("model", None, None),
              "model.layers.*.self_attn.q_proj.kernel": ("model", None, None),
              "model.layers.*.self_attn.v_proj.kernel": ("model", None, None),
              "model.norm.scale": (None,),
          }
          lora_sharding_mappings = {
              # LoRA mappings
              "model.layers.*.mlp.gate_proj.kernel_lora_a": (None, None),
              "model.layers.*.mlp.gate_proj.kernel_lora_b": (None, "model"),
              "model.layers.*.mlp.up_proj.kernel_lora_a": (None, None),
              "model.layers.*.mlp.up_proj.kernel_lora_b": (None, "model"),
              "model.layers.*.mlp.down_proj.kernel_lora_a": ("model", None),
              "model.layers.*.mlp.down_proj.kernel_lora_b": (None, None),
              "model.layers.*.self_attn.q_proj.kernel_lora_a": ("model", None),
              "model.layers.*.self_attn.q_proj.kernel_lora_b": (None, None),
              "model.layers.*.self_attn.k_proj.kernel_lora_a": ("model", None),
              "model.layers.*.self_attn.k_proj.kernel_lora_b": (None, None),
              "model.layers.*.self_attn.v_proj.kernel_lora_a": ("model", None),
              "model.layers.*.self_attn.v_proj.kernel_lora_b": (None, None),
              "model.layers.*.self_attn.o_proj.kernel_lora_a": ("model", None),
              "model.layers.*.self_attn.o_proj.kernel_lora_b": (None, None),
          }

          sharded_state = apply_sharding(abs_state, sharding_mappings, rng, mesh)

          model = nnx.merge(graph_def, sharded_state)
          # We should not jit the model here, because we need to update the model weights
          jit_model = model

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

            def create_model():
                model = model_class(vllm_config, rng, mesh)
                model = maybe_apply_lora_to_model(model, mesh, lora_config)
                return model

            model = nnx.eval_shape(lambda: create_model())
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

    model_fn = functools.partial(run_model, graphdef, state)
    return model_fn, model


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
    return model_fn, model


def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> List[nn.Module]:
    impl = os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower()
    logger.info(f"Loading model, implementation type={impl}")
    if impl == "flax_nn":
        return get_nn_model(vllm_config, rng, mesh)
    elif impl == "flax_nnx":
        return get_nnx_model(vllm_config, rng, mesh)
    elif impl == "vllm":
        return get_vllm_model(vllm_config, rng, mesh)
    else:
        raise NotImplementedError("Unsupported MODEL_IMPL_TYPE")
