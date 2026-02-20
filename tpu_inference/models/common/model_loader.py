# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from collections.abc import Iterator
from typing import Any, Optional

import jax
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.runai_streamer_loader import \
    RunaiModelStreamerLoader
from vllm.utils.func_utils import supports_kw

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
    apply_qwix_on_abstract_model, apply_qwix_quantization,
    load_random_weights_into_qwix_abstract_model,
    update_vllm_config_for_qwix_quantization)
from tpu_inference.models.jax.utils.weight_utils import (BaseWeightLoader,
                                                         LoadableWithIterator)
from tpu_inference.utils import to_jax_dtype, to_torch_dtype

logger = init_logger(__name__)

_MODEL_REGISTRY = {}

# List of architectures that are preferred to use  "vllm" implementation over
# "flax_nnx" implementation due to various factors such as performance.
_VLLM_PREFERRED_ARCHITECTURES: frozenset[str] = frozenset(
    {"GptOssForCausalLM"})


class UnsupportedArchitectureError(ValueError):
    """Raised when a model architecture is not supported in the registry."""
    pass


def _get_model_architecture(config: PretrainedConfig) -> nnx.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.

    from tpu_inference.models.jax.deepseek_v3 import DeepseekV3ForCausalLM
    from tpu_inference.models.jax.gpt_oss import GptOss
    from tpu_inference.models.jax.llama3 import LlamaForCausalLM
    from tpu_inference.models.jax.llama4 import Llama4ForCausalLM
    from tpu_inference.models.jax.llama_eagle3 import EagleLlama3ForCausalLM
    from tpu_inference.models.jax.llama_guard_4 import LlamaGuard4ForCausalLM
    from tpu_inference.models.jax.qwen2 import Qwen2ForCausalLM
    from tpu_inference.models.jax.qwen2_5_vl import \
        Qwen2_5_VLForConditionalGeneration
    from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM
    from tpu_inference.models.jax.qwen3_moe import Qwen3MoeForCausalLM
    _MODEL_REGISTRY["Llama4ForCausalLM"] = Llama4ForCausalLM
    _MODEL_REGISTRY["DeepseekV3ForCausalLM"] = DeepseekV3ForCausalLM
    _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
    _MODEL_REGISTRY["Llama4ForConditionalGeneration"] = LlamaGuard4ForCausalLM
    _MODEL_REGISTRY["Qwen3ForCausalLM"] = Qwen3ForCausalLM
    _MODEL_REGISTRY["Qwen3MoeForCausalLM"] = Qwen3MoeForCausalLM
    _MODEL_REGISTRY[
        "Qwen2_5_VLForConditionalGeneration"] = Qwen2_5_VLForConditionalGeneration
    _MODEL_REGISTRY["Eagle3LlamaForCausalLM"] = EagleLlama3ForCausalLM
    _MODEL_REGISTRY["GptOssForCausalLM"] = GptOss
    _MODEL_REGISTRY["Qwen2ForCausalLM"] = Qwen2ForCausalLM

    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise UnsupportedArchitectureError(
        f"Model architectures {architectures} not "
        "registered in tpu-inference. Falling back to vLLM-native "
        f"Pytorch definition. JAX-native architectures: {list(_MODEL_REGISTRY.keys())}"
    )


_SAFETENSOR_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _read_safetensor_headers(
    hf_weights_files: list[str], ) -> list[tuple[str, dict[str, Any], str]]:
    """Read safetensors file headers and return list of (name, meta, file)."""
    import json
    import struct

    entries = []
    for st_file in sorted(hf_weights_files):
        with open(st_file, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            entries.append((name, meta, st_file))
    return entries


def _create_moe_mock_weights(model: JaxModule) -> int:
    """Create mock MoE weights directly on TPU, bypassing checkpoint loading.

    Generates random weights matching the final post-processed shapes that
    ``process_fp8_moe_weights`` would produce.  This skips the expensive
    per-expert dequant -> requant -> transpose pipeline entirely, making it
    orders of magnitude faster than loading real checkpoint data.

    Weights are created on CPU one layer at a time to avoid memory pressure,
    then transferred to TPU via ``shard_put`` with the correct shardings.

    For FP8 + GMM_TP the output shapes are:

    ============  ==============  ========
    Attribute     Shape           Dtype
    ============  ==============  ========
    w13_weight    (E, D, 2F)      float8
    w13_scale     (E, 1, 1, 2F)   float32
    w2_weight     (E, F, D)       float8
    w2_scale      (E, 1, 1, D)    float32
    ============  ==============  ========

    After creation, ``_called_process_weights_after_loading`` is set on each
    layer's quant method so ``process_weights_after_loading`` is skipped later.

    Args:
        model: The model whose ``JaxMoE`` sub-modules will receive mock
            weights.  Only modules using ``Fp8FusedMoEMethod`` are processed.

    Returns:
        The number of MoE layers processed.
    """
    import time

    import jax.numpy as jnp

    from tpu_inference.layers.jax.moe.moe import JaxMoE
    from tpu_inference.layers.jax.quantization.fp8 import Fp8FusedMoEMethod
    from tpu_inference.models.jax.utils.weight_utils import shard_put

    cpu = jax.devices('cpu')[0]
    start = time.time()
    layers_processed = 0

    for module in _iter_modules(model):
        if not isinstance(module, JaxMoE):
            continue
        quant_method = getattr(module, 'quant_method', None)
        if not isinstance(quant_method, Fp8FusedMoEMethod):
            continue

        E = module.num_local_experts
        D = module.hidden_size
        F = module.intermediate_size_moe
        scale_name = quant_method.weight_scale_name

        # Create random values matching process_fp8_moe_weights output shapes.
        # Created on CPU one at a time, then shard_put to TPU.
        key = jax.random.PRNGKey(layers_processed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        with jax.default_device(cpu):
            w13 = jax.random.normal(k1, (E, D, 2 * F), dtype=jnp.float8_e4m3fn)
            w13_s = jnp.ones((E, 1, 1, 2 * F), dtype=jnp.float32)
            w2 = jax.random.normal(k2, (E, F, D), dtype=jnp.float8_e4m3fn)
            w2_s = jnp.ones((E, 1, 1, D), dtype=jnp.float32)

        # Delete old params (same as process_moe_after_loading)
        for attr in [
                'kernel_gating_EDF', 'kernel_up_proj_EDF',
                f'kernel_gating_EDF_{scale_name}',
                f'kernel_up_proj_EDF_{scale_name}'
        ]:
            if hasattr(module, attr):
                delattr(module, attr)

        # Set new params with correct shardings
        module.kernel_gating_upproj_EDF = nnx.Param(
            shard_put(w13, shardings=module.edf_sharding))
        module.kernel_down_proj_EFD = nnx.Param(
            shard_put(w2, shardings=module.efd_sharding))
        setattr(module, f'kernel_gating_upproj_EDF_{scale_name}',
                nnx.Param(shard_put(w13_s, shardings=(None, ))))
        setattr(module, f'kernel_down_proj_EFD_{scale_name}',
                nnx.Param(shard_put(w2_s, shardings=(None, ))))

        # Prevent process_weights_after_loading from re-processing
        quant_method._called_process_weights_after_loading = True
        layers_processed += 1

        if layers_processed % 10 == 0:
            logger.info(
                "MOCK_WEIGHTS: Created MoE weights for %d layers [%.1fs]",
                layers_processed,
                time.time() - start)

    logger.info("MOCK_WEIGHTS: Created MoE weights for %d layers in %.1fs",
                layers_processed,
                time.time() - start)
    return layers_processed


def _iter_modules(module: nnx.Module) -> Any:
    """Recursively yield all submodules (depth-first)."""
    yield module
    for _, child in module.named_children():
        if isinstance(child, (list, tuple)):
            for item in child:
                yield from _iter_modules(item)
        else:
            yield from _iter_modules(child)


def _non_moe_mock_weights(
    hf_weights_files: list[str],
    prefix: str = "",
    skip_experts: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield random tensors for checkpoint entries not handled by Phase 1.

    Reads safetensor file headers (metadata only, not weight data) to discover
    every tensor name, shape, and dtype in the checkpoint.  When
    ``skip_experts`` is ``True``, entries whose name contains ``".experts."``
    are filtered out because they were already created directly on TPU by
    ``_create_moe_mock_weights``.  When ``False`` (the default), all entries
    are included so that MoE models without FP8 still get their expert weights.

    For each included entry a random ``torch.Tensor`` with the correct shape
    and dtype is yielded so that the normal weight-loading pipeline
    (reshape / transpose / sharding) still runs.

    Args:
        hf_weights_files: List of safetensor file paths that make up the
            checkpoint.
        prefix: Optional string prepended to each tensor name before
            yielding.
        skip_experts: If ``True``, skip tensors whose name contains
            ``".experts."``.  Should be set when ``_create_moe_mock_weights``
            already handled those layers.

    Yields:
        ``(name, tensor)`` pairs where *name* is the (possibly prefixed)
        checkpoint key and *tensor* is a random ``torch.Tensor`` matching the
        original shape and dtype.
    """
    import time

    entries = _read_safetensor_headers(hf_weights_files)
    total = len(entries)
    if skip_experts:
        filtered = [(n, m) for n, m, _ in entries if ".experts." not in n]
    else:
        filtered = [(n, m) for n, m, _ in entries]
    skipped = total - len(filtered)
    if skipped:
        logger.info(
            "MOCK_WEIGHTS: %d total checkpoint tensors, %d MoE expert "
            "tensors handled directly, %d remaining tensors to load", total,
            skipped, len(filtered))
    else:
        logger.info("MOCK_WEIGHTS: %d total checkpoint tensors to load", total)

    start_time = time.time()
    for i, (name, meta) in enumerate(filtered):
        dtype = _SAFETENSOR_DTYPE_MAP.get(meta["dtype"])
        if dtype is None:
            logger.warning("Skipping %s: unknown dtype %s", name,
                           meta["dtype"])
            continue
        if (i + 1) % 50 == 0 or (i + 1) == len(filtered):
            elapsed = time.time() - start_time
            logger.info(
                "MOCK_WEIGHTS: tensor %d/%d (%.1f%%) "
                "[%.1fs elapsed] - %s", i + 1, len(filtered),
                100.0 * (i + 1) / len(filtered), elapsed, prefix + name)
        yield prefix + name, torch.randn(meta["shape"]).to(dtype)


def _load_mock_weights(
    loader: BaseWeightLoader,
    model: JaxModule,
    model_config: ModelConfig,
) -> None:
    """Load random mock weights instead of real checkpoint data.

    Uses a two-phase approach so that compilation is identical to a real run
    (all weight shapes, dtypes, and shardings are preserved) while avoiding
    the cost of reading checkpoint files from disk.

    Phase 1 — MoE expert weights (``_create_moe_mock_weights``):
        Creates post-processed MoE weights directly on TPU, skipping the
        per-expert tensor conversions and the dequant -> requant pipeline.

    Phase 2 — Remaining weights (``_non_moe_mock_weights``):
        Yields random tensors through the normal ``load_weights`` pipeline so
        that reshape / transpose / sharding logic is still exercised.  If
        Phase 1 handled MoE expert layers, expert tensors are skipped here;
        otherwise all checkpoint entries (including experts) are included,
        making this path work for any model.

    Args:
        loader: A weight loader instance (e.g. ``StandardWeightLoader``)
            that provides ``_prepare_weights`` for locating checkpoint files.
        model: The model to populate.  Must implement ``LoadableWithIterator``
            and ``load_weights``.
        model_config: The ``ModelConfig`` for the current run, used to
            resolve the checkpoint path and revision.
    """
    import time
    start = time.time()

    _, hf_weights_files, _ = loader._prepare_weights(
        model_config.model,
        model_config.revision,
        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                         None),
    )

    # Phase 1: Create FP8 MoE weights directly on TPU (optimization).
    # Returns 0 for non-MoE models or models without Fp8FusedMoEMethod.
    moe_layers = _create_moe_mock_weights(model)

    # Phase 2: Load remaining weights through normal pipeline.
    # If Phase 1 handled MoE experts, skip those tensors here;
    # otherwise include everything so non-FP8 MoE models still work.
    mock_iter = _non_moe_mock_weights(hf_weights_files,
                                      skip_experts=moe_layers > 0)
    model.load_weights(mock_iter)

    logger.info("MOCK_WEIGHTS: Total mock-weight loading completed in %.1fs",
                time.time() - start)


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
        # if not use_qwix_on_abstract_model:
        #     # NOTE: if Qwix is not configured, this will be a no-op
        #     model = apply_qwix_quantization(vllm_config,
        #                                     model,
        #                                     rng,
        #                                     mesh,
        #                                     apply_to_abstract_model=False)
        return model

    if vllm_config.load_config.load_format == "dummy":
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
                jit_model.initialize_cache(mesh)
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
        with jax.set_mesh(mesh):
            # if should_apply_qwix_on_abstract_model := apply_qwix_on_abstract_model(
            #         vllm_config):
            #     # NOTE: if Qwix is not configured, this will return `create_abstract_model` and
            #     # thus be a no-op
            #     raise ValueError(f"{vllm_config}")
            #     abstract_model_fn = apply_qwix_quantization(
            #         vllm_config,
            #         create_abstract_model,
            #         rng,
            #         mesh,
            #         apply_to_abstract_model=True)
            model = nnx.eval_shape(abstract_model_fn)
        # Although the created model can already work, we still need to jit
        # the model creation again, otherwise the model forward will have
        # non-trivial overhead in PjitFunction.
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            if isinstance(model, LoadableWithIterator):
                assert isinstance(model, JaxModule)
                if envs.MOCK_WEIGHTS:
                    logger.warning(
                        "MOCK_WEIGHTS=1: Replacing checkpoint data with "
                        "random values. Weight shapes/dtypes/shardings are "
                        "preserved so compilation is identical to a real run.")
                    _load_mock_weights(loader, model, vllm_config.model_config)
                else:
                    loader.load_weights(model, vllm_config.model_config)
            elif isinstance(loader, RunaiModelStreamerLoader):
                model_weights = vllm_config.model_config.model
                if hasattr(vllm_config.model_config, "model_weights"):
                    model_weights = vllm_config.model_config.model_weights
                weights_iterator = loader._get_weights_iterator(
                    model_weights, vllm_config.model_config.revision)
                # We set the weights iterator at runtime, to prevent having to change
                # every model's load_weights signature. This also prevents us from hitting
                # a TypeError at runtime if you use the RunaiModelStreamerLoader with any
                # flax_nnx model whose load_weights function does not accept the
                # weights_iterator keyword argument.
                vllm_config.model_config.runai_model_weights_iterator = weights_iterator
                model.load_weights(rng)
                del vllm_config.model_config.runai_model_weights_iterator
            else:
                model.load_weights(rng)
            jit_model = create_jit_model(model,
                                         use_qwix_on_abstract_model=False)
    return jit_model


# TODO(pooyam): We need to refactor this. This is returning a bunch of functions that do not work with all models and this is not very easy to see from the code.
def get_flax_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    is_draft_model: bool = False,
) -> nnx.Module:
    model_dtype = to_jax_dtype(vllm_config.model_config.dtype)
    vllm_config.model_config.dtype = model_dtype
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

    # Only perform qwix quantization if it is jax model.
    if vllm_config.model_config:
        update_vllm_config_for_qwix_quantization(vllm_config)

    if is_draft_model:
        model_class = _get_model_architecture(
            vllm_config.speculative_config.draft_model_config.hf_config)
    else:
        model_class = _get_model_architecture(
            vllm_config.model_config.hf_config)
    jit_model = _get_nnx_model(model_class, vllm_config, rng, mesh)
    kv_cache_sharding = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD))
    hidden_states_sharding = NamedSharding(mesh,
                                           PartitionSpec(
                                               ShardingAxisName.ATTN_DATA,
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
        static_argnums=(
            7, 10, 11
        ),  #7 is layer_name_to_kvcache_index, 10 is is_first_rank, 11 is is_last_rank
    )
    def run_model(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        return model(*args)

    logits_sharding = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR))

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
    def run_embed_multimodal(graphdef, state, image_grid_thw, **kwargs):
        model = nnx.merge(graphdef, state)
        return model.embed_multimodal(image_grid_thw, **kwargs)

    embed_sharding = NamedSharding(mesh, PartitionSpec(None))
    # This function will calculates the embeddings of input texts and then merge with the image embeddings
    @functools.partial(
        jax.jit,
        out_shardings=(embed_sharding),
    )
    def run_embed_input_ids(graphdef, state, *args, **kwargs):
        model = nnx.merge(graphdef, state)
        return model.embed_input_ids(*args, **kwargs)

    # For models that want to work with EAGLE-3 speculative decoding
    @functools.partial(
        jax.jit,
        out_shardings=(logits_sharding),
    )
    def combine_hidden_states(graphdef, state, hidden_states):
        model = nnx.merge(graphdef, state)
        return model.combine_hidden_states(hidden_states)

    model = nnx.merge(graphdef, state)
    precompile_vision_encoder_fn = getattr(model, "precompile_vision_encoder",
                                           None)
    model_fn = functools.partial(run_model, graphdef)
    compute_logits_fn = functools.partial(run_compute_logits, graphdef)
    embed_multimodal_fn = functools.partial(run_embed_multimodal, graphdef)
    embed_input_ids_fn = functools.partial(run_embed_input_ids, graphdef)
    lora_manager, model = None, None
    combine_hidden_states_fn = functools.partial(combine_hidden_states,
                                                 graphdef)

    get_mrope_input_positions_fn = None if not hasattr(
        jit_model,
        "get_mrope_input_positions") else jit_model.get_mrope_input_positions

    multimodal_fns = {
        "precompile_vision_encoder_fn": precompile_vision_encoder_fn,
        "embed_multimodal_fn": embed_multimodal_fn,
        "embed_input_ids_fn": embed_input_ids_fn,
        "get_mrope_input_positions_fn": get_mrope_input_positions_fn,
    }

    return model_fn, compute_logits_fn, combine_hidden_states_fn, multimodal_fns, state, lora_manager, model


def get_vllm_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
):
    model_dtype = to_torch_dtype(vllm_config.model_config.dtype)
    vllm_config.model_config.dtype = model_dtype
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

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
    return jit_model, compute_logits_fn, combine_hidden_states_fn, None, params, lora_manager, model


def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    is_draft_model: bool = False,
) -> Any:
    impl = envs.MODEL_IMPL_TYPE
    logger.info(f"Loading model with MODEL_IMPL_TYPE={impl}")
    if impl == "auto":
        impl = resolve_model_architecture(vllm_config)
        logger.info(f"Resolved MODEL_IMPL_TYPE 'auto' to '{impl}'")

    match impl:
        case "flax_nnx":
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                logger.warning(
                    "PP is not fully supported on Jax flax_nnx models yet, fallback to vllm models."
                )
                return get_vllm_model(vllm_config, rng, mesh)
            try:
                # Try to load the flax model first
                return get_flax_model(vllm_config, rng, mesh, is_draft_model)
            except UnsupportedArchitectureError as e:
                # Convert the error message to a string to check its contents
                error_msg = str(e)

                logger.warning(error_msg)

                # Fall back to the vLLM model and updating the dtype accordingly
                return get_vllm_model(vllm_config, rng, mesh)
        case "vllm":
            return get_vllm_model(vllm_config, rng, mesh)
        case _:
            raise NotImplementedError(f"Unsupported MODEL_IMPL_TYPE: {impl}")


def resolve_model_architecture(vllm_config: VllmConfig) -> str:
    """Resolves the model implementation type.

    This function determines which model implementation to use based on the model
    architecture and whether the RunAI model streamer is active.

    When the RunAI model streamer is used, this function explicitly checks if
    the JAX model supports the streaming capability. It returns 'vllm' if:
    1. The JAX model class is found but does not have a `WeightLoader`.
    2. The JAX model's `WeightLoader` is not a subclass of `BaseWeightLoader`.

    If the architecture is not registered in JAX (UnsupportedArchitectureError),
    this function returns the default implementation ('flax_nnx'), allowing
    the caller to attempt loading, catch the error, log a
    warning, and handle the fallback to 'vllm'.

    Otherwise, it resolves the implementation based on whether the model's
    architecture is in the `_VLLM_PREFERRED_ARCHITECTURES` list.

    Args:
        vllm_config: The VllmConfig object containing the model and load
            configuration.

    Returns:
        The model implementation type.
    """

    is_runai_streamer = getattr(getattr(vllm_config, 'load_config', None),
                                'load_format', None) == 'runai_streamer'
    if is_runai_streamer:
        try:
            # Try to get the JAX model class
            model_class = _get_model_architecture(
                vllm_config.model_config.hf_config)

            # If found, check for WeightLoader capability
            if not hasattr(model_class, "WeightLoader") or not issubclass(
                    getattr(model_class, "WeightLoader", object),
                    BaseWeightLoader):
                return "vllm"

        except UnsupportedArchitectureError:
            # Architecture not in JAX.
            # We pass to let it fall through to the default logic below.
            # This causes it to return "flax_nnx", which caller will try,
            # fail, log a warning, and then fallback to "vllm".
            pass

    # Resolve "auto" based on architecture
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", [])
    assert len(architectures) == 1, (
        f"Expected exactly one architecture, got {len(architectures)}: "
        f"{architectures}")
    arch = architectures[0]
    impl = "vllm" if arch in _VLLM_PREFERRED_ARCHITECTURES else "flax_nnx"
    return impl


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

    This function registers the model with both the tpu_inference registry
    and the vLLM registry. For vLLM, it creates a compatible wrapper
    around the JAX model.

    Args:
        arch: The name of the architecture (e.g., "LlamaForCausalLM").
        model: The JAX model class to register (e.g., a flax.nnx.Module).
    """
    _validate_model_interface(model)

    # Register with tpu_inference registry for the JAX backend
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

    # Same as `forward`, this is a dummy method to satisfy vLLM's type checks.
    def unimplemented_embed_input_ids(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        inputs_embeds: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        raise NotImplementedError(
            "This is a JAX model and does not implement the PyTorch embed_input_ids method."
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
            "embed_input_ids": unimplemented_embed_input_ids,
            # Prevent vLLM from trying to load weights into this dummy class.
            "load_weights": lambda self, *args, **kwargs: None,
        })

    # Register the wrapped model with vLLM's registry.
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(arch, VllmCompatibleModel)
    logger.info(
        f"Registered JAX model {arch} with tpu_inference and vLLM registries.")
