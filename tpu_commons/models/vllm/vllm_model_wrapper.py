import copy
import functools
import os
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import jax
import torch
import torch.nn
import torchax
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import TORCH_DTYPE_TO_JAX
from vllm.config import LoRAConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.sequence import IntermediateTensors

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.vllm.quantization import get_tpu_quantization_config
from tpu_commons.models.vllm.sharding import (
    LORA_MODULE_TYPE_TO_WRAPPING_FUNC, get_fqn, shard_model_to_tpu)
from tpu_commons.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)

logger = init_logger(__name__)


class _VllmRunner(torch.nn.Module):

    def __init__(self, vllm_model: torch.nn.Module):
        super().__init__()
        self.vllm_model = vllm_model

    def forward(self, **kwargs) -> torch.Tensor:
        # We don't support multimodal input in Gemma3, but we need patch it to
        # None to workaround vLLM Gemma3 model bug that
        # `get_multimodal_embeddings` returns empty list but it's caller checks
        # for None.
        with patch(
                "vllm.model_executor.models.gemma3_mm."
                "Gemma3ForConditionalGeneration."
                "get_multimodal_embeddings",
                return_value=None):
            if "hidden_state" in kwargs:
                return self.compute_logits(kwargs["hidden_state"])
            else:
                return self.compute_hidden_state(
                    kwargs["input_ids"],
                    kwargs["positions"],
                    kwargs["intermediate_tensors"],
                    kwargs["inputs_embeds"],
                )

    def compute_hidden_state(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_state = self.vllm_model(input_ids, positions,
                                       intermediate_tensors, inputs_embeds)
        return hidden_state

    def compute_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.vllm_model.compute_logits(hidden_state)


class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    rng: PRNGKey
    mesh: Mesh
    model: _VllmRunner

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = rng
        self.mesh = mesh

        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config, self.mesh)

    def load_weights(self):
        # Set up to load the model into CPU first.
        vllm_config_for_load = copy.deepcopy(self.vllm_config)
        assert self.vllm_config.model_config.dtype in TORCH_DTYPE_TO_JAX, "The model_config.dtype must be a PyTorch dtype."
        vllm_config_for_load.device_config.device = "cpu"

        if os.getenv("JAX_RANDOM_WEIGHTS", False):
            vllm_config_for_load.load_config.load_format = "dummy"
            use_random_weights = True
        else:
            use_random_weights = (
                vllm_config_for_load.load_config.load_format == "dummy")
        if use_random_weights:
            logger.info(
                "Initializing vLLM model with random weights, weight loading skipped."
            )
        # The DummyModelLoader in vLLM calls torch._sync for torch_xla path when it detects the tpu platform, but we don't need it and it causes crash without proper setup.
        load_context = patch(
            "torch._sync",
            return_value=None) if use_random_weights else nullcontext()

        # Load the vLLM model and wrap it into a new model whose forward function can calculate the hidden_state and logits.
        with load_context:
            # vllm_get_model will move the model weight to TPU.
            vllm_model = vllm_get_model(vllm_config=vllm_config_for_load)
        lora_manager = None
        if vllm_config_for_load.lora_config is not None:
            # replace the layer with LoRA layers.
            with torchax.default_env():
                # the device in load_lora_model is used to set the device used in punica wrapper.
                lora_manager, vllm_model = load_lora_model(
                    vllm_model,
                    vllm_config_for_load.model_config,
                    vllm_config_for_load.scheduler_config,
                    vllm_config_for_load.lora_config,
                    device="jax")
                self._register_lora_weights_as_param(
                    vllm_model, vllm_config_for_load.lora_config)
            replace_set_lora(vllm_model)

        self.model = _VllmRunner(vllm_model)

        # jax.config.update("jax_explain_cache_misses", True)
        self.vllm_config.compilation_config.static_forward_context = vllm_config_for_load.compilation_config.static_forward_context

        params_and_buffers = shard_model_to_tpu(self.model, self.mesh,
                                                self.vllm_config)

        # Returning to the jax land, so we need to wrap it into a JaxValue.
        return jax_view(params_and_buffers), lora_manager

    def _register_lora_weights_as_param(self, model: torch.nn.Module,
                                        lora_config: LoRAConfig) -> None:

        def _process_module(module, name=None, parent=None):
            if get_fqn(module) in LORA_MODULE_TYPE_TO_WRAPPING_FUNC:
                assert parent is not None and name is not None, (
                    "Top Level module is not expected to be LoRA wrapper")
                module.lora_a_stacked = torch.nn.ParameterList([
                    torch.nn.Parameter(module.lora_a_stacked[i])
                    for i in range(module.n_slices)
                ])
                module.lora_b_stacked = torch.nn.ParameterList([
                    torch.nn.Parameter(module.lora_b_stacked[i])
                    for i in range(module.n_slices)
                ])
                if lora_config.bias_enabled:
                    module.lora_bias_stacked = torch.nn.ParameterList([
                        torch.nn.Parameter(module.lora_bias_stacked[i])
                        for i in range(module.n_slices)
                    ])

            for child_name, child_module in list(module.named_children()):
                _process_module(child_module, child_name, module)

        _process_module(model)

    def jit_step_func(self):

        @functools.partial(
            jax.jit,
            donate_argnums=(1, ),  # donate kv_cache
            compiler_options={
                "xla_tpu_all_gather_collective_matmul_mode":
                "post_spmd_conservative",
                "xla_tpu_reduce_scatter_collective_matmul_mode":
                "post_spmd_conservative"
            },
            static_argnames=('layer_name_to_kvcache_index', ))
        def step_fun(
            params_and_buffers,  # this has been wrapped into a torchax TorchValue
            kv_caches: List[jax.Array],
            input_ids: jax.Array,
            attn_metadata: AttentionMetadata,
            input_embeds: jax.Array,
            layer_name_to_kvcache_index: Sequence[Tuple[str, int]],
            *args,
        ) -> Tuple[List[jax.Array], jax.Array]:
            layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=kv_caches,
                    mesh=self.mesh,
                    layer_name_to_kvcache_index=layer_name_to_kvcache_index
            ), set_forward_context(attn_metadata=attn_metadata,
                                   vllm_config=self.vllm_config):
                # We need to wrap args from jax land into TorchValue with
                # torch_view in order to call the Torch function.
                hidden_states = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "input_ids": torch_view(input_ids),
                        "positions": torch_view(attn_metadata.input_positions),
                        "intermediate_tensors": None,
                        "inputs_embeds": None,
                    },
                    tie_weights=False,
                )
                vllm_model_wrapper_context = get_vllm_model_wrapper_context()
                new_kv_caches = vllm_model_wrapper_context.kv_caches
            # Wrap the hidden_states from torch land into a JaxValue for the jax
            # code to consume.
            hidden_states = jax_view(hidden_states)

            return new_kv_caches, hidden_states, []

        return step_fun

    def jit_compute_logits_func(self):

        @functools.partial(
            jax.jit,
            out_shardings=(NamedSharding(self.mesh,
                                         PartitionSpec(None, "model"))),
        )
        def compute_logits_func(
            params_and_buffers: Any,
            hidden_states: jax.Array,
        ) -> jax.Array:
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=None, mesh=self.mesh):
                logits = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "hidden_state": torch_view(hidden_states),
                    },
                    tie_weights=False,
                )
            return jax_view(logits)

        return compute_logits_func


def load_lora_model(model: torch.nn.Module, model_config: ModelConfig,
                    scheduler_config: SchedulerConfig, lora_config: LoRAConfig,
                    device: str) -> torch.nn.Module:
    if not supports_lora(model):
        raise ValueError(
            f"{model.__class__.__name__} does not support LoRA yet.")

    if supports_multimodal(model):
        logger.warning("Regarding multimodal models, vLLM currently "
                       "only supports adding LoRA to language model.")

    # Use get_text_config() in case of multimodal models
    text_config = model_config.hf_config.get_text_config()

    # Add LoRA Manager to the Model Runner
    lora_manager = LRUCacheWorkerLoRAManager(
        scheduler_config.max_num_seqs,
        scheduler_config.max_num_batched_tokens,
        model_config.get_vocab_size(),
        lora_config,
        device,
        model.embedding_modules,
        model.embedding_padding_modules,
        max_position_embeddings=text_config.max_position_embeddings,
    )
    return lora_manager, lora_manager.create_lora_manager(model)


# The reason why replace the method is that the set_lora and reset_lora need to run under torchax env.
def replace_set_lora(model):

    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        with torchax.default_env():
            self._original_set_lora(index, lora_a, lora_b, embeddings_tensor,
                                    bias)

    def _tpu_reset_lora(self, index: int):
        with torchax.default_env():
            self._original_reset_lora(index)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(
                module, module.__class__)
