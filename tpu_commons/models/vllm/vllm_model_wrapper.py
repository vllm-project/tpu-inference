import functools
import tempfile
from typing import List, Optional, Tuple

import jax
import torch
import torch.nn
import torchax
from flax.typing import PRNGKey
from jax.sharding import Mesh
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers, jax_view, torch_view
from torchax.ops.mappings import j2t_dtype
from vllm.attention import Attention as VllmAttention
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.sequence import IntermediateTensors

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.sampling import sample
from tpu_commons.models.vllm.jax_attention_wrapper import JaxAttentionWrapper
from tpu_commons.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)

KVCache = Tuple[jax.Array, jax.Array]


def swap_attention_module(model: torch.nn.Module, mesh: Mesh) -> None:
    """
    Swap the Attention torhc.nn.Module used in the model with an implementation
    in JAX, which uses the KVCache management and attention kernels for TPU.

    Args:
        model: A vLLM model
    """

    def _process_module(module, name=None, parent=None):
        if isinstance(module, VllmAttention):
            wrapped_module = JaxAttentionWrapper(module, mesh)
            assert parent is not None and name is not None, (
                "Top Level module is not expected to be wrapped.")
            setattr(parent, name, wrapped_module)
        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)


class ModelForLogits(torch.nn.Module):

    def __init__(self, vllm_model: torch.nn.Module):
        super().__init__()

        self.vllm_model = vllm_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_state = self.vllm_model(input_ids, positions,
                                       intermediate_tensors, inputs_embeds)
        return self.vllm_model.compute_logits(hidden_state,
                                              sampling_metadata=None)


class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    rng: PRNGKey
    mesh: Mesh

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        vllm_config.model_config.dtype = j2t_dtype(
            vllm_config.model_config.dtype.dtype)
        vllm_config.device_config.device = "xla"

        self.vllm_config = vllm_config
        self.rng = rng
        self.mesh = mesh

    def load_weights(self):
        # Initialize the vLLM distribution layer as a single chip environment,
        # we'll swap the model's parallel modules with TPU SPMD equivalents.
        with set_current_vllm_config(self.vllm_config):
            temp_file = tempfile.mkstemp()[1]
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{temp_file}",
                backend="gloo",
            )
            ensure_model_parallel_initialized(
                self.vllm_config.parallel_config.tensor_parallel_size,
                self.vllm_config.parallel_config.pipeline_parallel_size,
            )

        # Load the vLLM model and wrap it into a new model whose forward
        # function calculates the hidden_state and logits in one go.
        model = ModelForLogits(vllm_get_model(vllm_config=self.vllm_config))

        swap_attention_module(model, self.mesh)

        # jax.config.update("jax_explain_cache_misses", True)

        # Move the model into torchax 'jax' device so that jax code can interop
        # with it and the overall program can be traced and compiled in XLA.
        with torchax.default_env():
            self.model_for_logits = model.to('jax')
            params, buffers = extract_all_buffers(self.model_for_logits)
            params, buffers = pytree.tree_map_only(torch.Tensor,
                                                   lambda x: x.to('jax'),
                                                   (params, buffers))
            params_and_buffers = {**params, **buffers}

        # Returning to the jax land, so we need to wrap it into a JaxValue.
        return jax_view(params_and_buffers)

    def jit_step_func(self):

        @functools.partial(
            jax.jit,
            static_argnums=(1, 2),  # is_prefill, do_sampling
            donate_argnums=(3, ),  # donate kv_cache
        )
        def step_fun(
            params_and_buffers,  # this has been wrapped into a torchax TorchValue
            is_prefill: bool,
            do_sampling: bool,
            kv_caches: List[KVCache],
            input_ids: jax.Array,
            attention_metadata: AttentionMetadata,
            temperatures: jax.Array,
            top_ps: jax.Array,
            top_ks: jax.Array,
            *args,
        ) -> Tuple[List[KVCache], jax.Array, jax.Array]:

            with torchax.default_env(), set_vllm_model_wrapper_context(
                    is_prefill=is_prefill,
                    kv_caches=kv_caches,
                    attention_metadata=attention_metadata,
            ):
                # We need to wrap args from jax land into TorchValue with
                # torch_view in order to call the Torch function.
                logits = torch.func.functional_call(
                    self.model_for_logits,
                    torch_view(params_and_buffers),
                    kwargs={
                        "input_ids": torch_view(input_ids),
                        "positions":
                        torch_view(attention_metadata.input_positions),
                        "intermediate_tensors": None,
                        "inputs_embeds": None,
                    },
                    tie_weights=False,
                    strict=True)
                vllm_model_wrapper_context = get_vllm_model_wrapper_context()
                new_kv_caches = vllm_model_wrapper_context.kv_caches
            # Wrap the logits from torch land into a JaxValue for the jax code
            # to consume.
            logits = jax_view(logits)

            next_tokens = sample(
                is_prefill,
                do_sampling,
                self.rng,
                self.mesh,
                logits,
                attention_metadata.seq_lens,
                temperatures,
                top_ps,
                top_ks,
                attention_metadata.chunked_prefill_enabled,
            )

            return new_kv_caches, next_tokens, logits

        return step_fun
