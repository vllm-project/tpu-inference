from typing import List, Optional, Tuple
import tempfile
import functools
import humanize


import jax
from jax.sharding import Mesh
from flax.typing import PRNGKey

import torch
import torch.nn
from torch.utils import _pytree as pytree

import torchax
from torchax.interop import extract_all_buffers, jax_jit
from torchax.ops.mappings import j2t_dtype


from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.sampling import sample
from tpu_commons.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
    set_vllm_model_wrapper_context,
)
from tpu_commons.models.vllm.jax_attention_wrapper import JaxAttentionWrapper


from vllm.attention import Attention as VllmAttention
from vllm.config import VllmConfig
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized
from vllm.sequence import IntermediateTensors


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
        model_output = self.vllm_model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return self.vllm_model.compute_logits(model_output, sampling_metadata=None)


class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    rng: PRNGKey
    mesh: Mesh

    def __init__(
            self,
            vllm_config: VllmConfig,
            rng: PRNGKey,
            mesh: Mesh):
        vllm_config.model_config.dtype = j2t_dtype(vllm_config.model_config.dtype.dtype)
        vllm_config.device_config.device = "xla"

        self.vllm_config = vllm_config
        self.rng = rng
        self.mesh = mesh

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
                self.vllm_config.parallel_config.pipeline_parallel_size,)

        model = ModelForLogits(get_model(vllm_config=self.vllm_config))
        swap_attention_module(model, mesh)

        # model.model.layers = model.model.layers[0:2]
        jax.config.update("jax_explain_cache_misses", True)

        with torchax.default_env():
            self.model_for_logits = model.to('jax')
            params, buffers = extract_all_buffers(self.model_for_logits)
            params, buffers = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
                                                (params, buffers))
            self.model_params_and_buffers = {**params, **buffers}


    def init_jit(self):
        self.model_func = self.get_jitted_model_func()


    def get_jitted_model_func(self):
        @functools.partial(
            jax_jit,
            kwargs_for_jax_jit={
                "static_argnums": (1, ),
                "donate_argnums": (2, ),  # kv_caches
            },
        )
        def func(
            model_params_and_buffers,
            is_prefill: bool,
            kv_caches: List[KVCache],
            input_ids: jax.Array,
            attention_metadata: AttentionMetadata,
        ):
            with set_vllm_model_wrapper_context(
                    is_prefill=is_prefill,
                    kv_caches=kv_caches,
                    attention_metadata=attention_metadata,):
                logits = torch.func.functional_call(
                    self.model_for_logits,
                    model_params_and_buffers,
                    kwargs={
                        "input_ids": input_ids,
                        "positions": attention_metadata.input_positions,
                        "intermediate_tensors": None,
                        "inputs_embeds": None,
                    },
                    tie_weights=False,
                    strict=True)
                vllm_model_wrapper_context = get_vllm_model_wrapper_context()
                new_kv_caches = vllm_model_wrapper_context.kv_caches
            return new_kv_caches, logits

        # import inspect
        # print(f"signature={inspect.signature(func)}")
        return func


    def step_func(
        self,
        is_prefill: bool,
        do_sampling: bool,
        kv_caches: List[KVCache],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        temperatures: jax.Array = None,
        top_ps: jax.Array = None,
        top_ks: jax.Array = None,
        *args,
    ) -> Tuple[List[KVCache], jax.Array, jax.Array]:

        print("is_prefill={}".format(is_prefill))
        print("input_ids={}".format(input_ids))
        print("input_ids.shape={}".format(input_ids.shape))
        print("input_positions={}".format(attention_metadata.input_positions))
        print("input_positions.shape={}".format(attention_metadata.input_positions.shape))
        print(f"attention_metadata.seq_lens={attention_metadata.seq_lens}")

        fmt_size = functools.partial(humanize.naturalsize, binary=True)
        for d in jax.local_devices():
            stats = d.memory_stats()
            used = stats['bytes_in_use']
            limit = stats['bytes_limit']
            print(f"TPU device using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

        print(f"type(kv_caches[0][0])={type(kv_caches[0][0])}")
        print(f"type(kv_caches[0][1])={type(kv_caches[0][1])}")
        n_elems = 0
        for kv_cache in kv_caches:
            n_elems += kv_cache[0].size
            n_elems += kv_cache[1].size
        print(f"kv_caches n_elems={n_elems}")
        print(f"kv_caches dtype = {kv_caches[0][0].dtype}")


        new_kv_caches, logits = self.model_func(
            self.model_params_and_buffers,
            is_prefill,
            kv_caches,
            input_ids,
            attention_metadata,)
        new_kv_caches = [(k_cache.jax(), v_cache.jax()) for (k_cache, v_cache) in new_kv_caches]
        print(f"logits.shape={logits.shape}")
        print(f"type(logits)={type(logits)}")
        print(f"logits={logits}")
        print(f"new_kv_caches dtype = {new_kv_caches[0][0].dtype}")
        print(f"type(new_kv_caches[0][0]) = {type(new_kv_caches[0][0])}")
        print(f"type(new_kv_caches[0][1]) = {type(new_kv_caches[0][1])}")

        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng,
            self.mesh,
            logits.jax(),
            attention_metadata.seq_lens,
            temperatures,
            top_ps,
            top_ks,
            attention_metadata.chunked_prefill_enabled,
        )

        return new_kv_caches, next_tokens, logits