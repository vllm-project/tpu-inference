from typing import List, Optional, Tuple, Dict, Any
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import humanize


import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import linen as nn
from flax.typing import PRNGKey

import torch
import torch.nn as torch_nn
from torch.utils import _pytree as pytree

import torchax
from torchax.interop import call_jax, extract_all_buffers, jax_jit
from torchax.ops.mappings import j2t_dtype

from transformers import PretrainedConfig

from tpu_commons.models.jax import layers
from tpu_commons.models.jax.sampling import sample
# from hex_llm.models.jax.kv_cache_eviction import KVCacheUpdater
from tpu_commons.models.jax.utils.weight_utils import (
    get_num_kv_heads_by_tp, get_num_q_heads_by_tp, hf_model_weights_iterator)

from vllm.attention import Attention as VllmAttention
from vllm.config import VllmConfig
from vllm.config import CacheConfig
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import (extract_layer_index)
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized


KVCache = Tuple[jax.Array, jax.Array]



def _jax_attn_func(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        attention_metadata: layers.AttentionMetadata,
) -> Tuple[KVCache, jax.Array]:

    md = attention_metadata
    k_cache, v_cache = kv_cache
    k_cache = layers.update_cache(is_prefill, k_cache,
                                    md.kv_cache_write_indices, k)
    v_cache = layers.update_cache(is_prefill, v_cache,
                                    md.kv_cache_write_indices, v)

    if is_prefill:
        # (B, N, T, H)
        # TODO(xiang): support MQA and GQA
        if self.num_kv_heads != self.num_heads:
            k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=1)
            v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=1)
        outputs = self.flash_attention(q, k, v)
    else:
        # (B, N, H)
        q = jnp.squeeze(q, 2)
        outputs = self.paged_attention(q, k_cache, v_cache, md.seq_lens+1,
                                        md.block_indices)
        # (B, N, 1, H)
        outputs = jnp.expand_dims(outputs, 2)
    
    new_kv_cache = (k_cache, v_cache)
    return new_kv_cache, outputs


class JaxAttentionWrapper(torch_nn.Module):
    def __init__(
        self,
        vllm_attn: VllmAttention,
        mesh: Mesh,
    ) -> None:
        super().__init__()

        self.num_heads = vllm_attn.num_heads
        self.head_size = vllm_attn.head_size
        self.scale = vllm_attn.impl.scale
        self.num_kv_heads = vllm_attn.num_kv_heads
        self.layer_idx = extract_layer_index(vllm_attn.layer_name)

        self.flash_attention = layers.sharded_flash_attention(mesh)
        self.paged_attention = layers.sharded_paged_attention(mesh)

    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the q shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        # if self.layer_idx <= 1:
        #     jax.debug.print("layer_idx={layer_idx}: before reshape q={q}\nk={k}\nv={v}",
        #                     layer_idx=self.layer_idx, q=q.jax(), k=k.jax(), v=v.jax())
            #jax.debug.breakpoint()
        
        # print("q.shape={}".format(q.shape))
        # print("k.shape={}".format(k.shape))
        # print("v.shape={}".format(v.shape))
        # print("k_cache.shape={}".format(kv_cache[0].shape))
        # print("v_cache.shape={}".format(kv_cache[1].shape))

        vllm_model_wrapper_context = get_vllm_model_wrapper_context()

        head_dim = self.head_size
        bs, q_len, q_compute_dim = q.shape
        num_heads = q_compute_dim // head_dim

        _, k_len, k_compute_dim = k.shape
        assert k.shape == v.shape
        assert k.shape[0] == bs
        num_kv_heads = k_compute_dim // head_dim

        # bs, num_heads, q_len, head_dim
        q = q.reshape(bs, q_len, num_heads, head_dim).swapaxes(1, 2)
        # vllm scales q in the common Attention class, but hex-llm scales it in each of the model code.
        q = q * self.scale
        # bs, num_kv_heads, k_len, head_dim
        k = k.reshape(bs, k_len, num_kv_heads, head_dim).swapaxes(1, 2)
        v = v.reshape(bs, k_len, num_kv_heads, head_dim).swapaxes(1, 2)

        if self.layer_idx <= 1:
            kv_cache=vllm_model_wrapper_context.kv_caches[self.layer_idx]
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]
            jax.debug.print("layer_idx={layer_idx}: after reshape\nq={q}\nk={k}\nv={v}\nk_cache={k_cache}\nv_cache={v_cache}",
                            layer_idx=self.layer_idx, q=q.jax(), k=k.jax(), v=v.jax(), k_cache=k_cache.jax(), v_cache=v_cache.jax())

        new_kv_cache, outputs = call_jax(
            _jax_attn_func,
            self,
            is_prefill=vllm_model_wrapper_context.is_prefill,
            kv_cache=vllm_model_wrapper_context.kv_caches[self.layer_idx],
            q=q, k=k, v=v,
            attention_metadata=vllm_model_wrapper_context.attention_metadata,)
        vllm_model_wrapper_context.kv_caches[self.layer_idx] = new_kv_cache

        # print("outputs.shape={}".format(outputs.shape))
        if self.layer_idx <= 1:
            kv_cache=vllm_model_wrapper_context.kv_caches[self.layer_idx]
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]
            jax.debug.print("layer_idx={layer_idx}: before reshape\noutputs={outputs}\nk_cache={k_cache}\nv_cache={v_cache}",
                            layer_idx=self.layer_idx, outputs=outputs.jax(), k_cache=k_cache.jax(), v_cache=v_cache.jax())

        assert outputs.shape[0] == bs
        assert outputs.shape[1] == num_heads
        assert outputs.shape[2] == q_len
        assert outputs.shape[3] == head_dim
        outputs = outputs.swapaxes(1, 2)  # bs, q_len, num_heads, head_dim
        outputs = outputs.reshape(bs, q_len, num_heads*head_dim)

        # if self.layer_idx <= 1:
        #     jax.debug.print("layer_idx={layer_idx}: after reshape outputs={outputs}", layer_idx=self.layer_idx, outputs=outputs.jax())

        return outputs



@dataclass
class VllmModelWrapperContext:
    is_prefill: bool
    kv_caches: List[KVCache]
    attention_metadata: layers.AttentionMetadata


_vllm_model_wrapper_context: Optional[VllmModelWrapperContext] = None

def get_vllm_model_wrapper_context() -> VllmModelWrapperContext:
    assert _vllm_model_wrapper_context is not None, (
        "VllmModelWrapperContext is not set. "
        "Please use `set_vllm_model_wrapper_context` to set the VllmModelWrapperContext.")
    return _vllm_model_wrapper_context

@contextmanager
def set_vllm_model_wrapper_context(*,
                                   is_prefill: bool,
                                   kv_caches: List[KVCache],
                                   attention_metadata: layers.AttentionMetadata,):
    global _vllm_model_wrapper_context
    prev_context = _vllm_model_wrapper_context
    _vllm_model_wrapper_context = VllmModelWrapperContext(
        is_prefill=is_prefill,
        kv_caches=kv_caches,
        attention_metadata=attention_metadata,)

    try:
        yield
    finally:
        _vllm_model_wrapper_context = prev_context


class ComputeLogitsForVLLMModel(torch_nn.Module):
    def __init__(self, vllm_model: torch_nn.Module):
        super().__init__()

        self.vllm_model = vllm_model
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.vllm_model.compute_logits(hidden_states, sampling_metadata=None)


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

class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    hf_config: PretrainedConfig
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
        self.hf_config = vllm_config.model_config.hf_config
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

        model = get_model(vllm_config=self.vllm_config)
        swap_attention_module(model, mesh)
        compute_logits_model = ComputeLogitsForVLLMModel(model)
        # model.model.layers = model.model.layers[0:2]
        jax.config.update("jax_explain_cache_misses", True)
        with torchax.default_env():
            self.vllm_model = model.to('jax')
            params, buffers = extract_all_buffers(self.vllm_model)
            params, buffers = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
                                                (params, buffers))
            self.vllm_params_and_buffers = {**params, **buffers}

            self.compute_logits_model = compute_logits_model.to('jax')
            params, buffers = extract_all_buffers(self.compute_logits_model)
            params, buffers = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
                                                (params, buffers))
            self.compute_logits_model_params_and_buffers = {**params, **buffers}


    def init_jit(self):
        self.model_func = self.get_jitted_model_func()
        self.compute_logits_func = self.get_jitted_compute_logits_func()


    def get_jitted_model_func(self):
        @functools.partial(
            jax_jit,
            kwargs_for_jax_jit={
                "static_argnames": ["is_prefill"],
                "donate_argnames": ["kv_caches"],  # TODO: use "donate_argnums"
            },
        )
        def func(
            is_prefill: bool,
            kv_caches: List[KVCache],
            input_ids: jax.Array,
            attention_metadata: layers.AttentionMetadata,):
            with set_vllm_model_wrapper_context(
                    is_prefill=is_prefill,
                    kv_caches=kv_caches,
                    attention_metadata=attention_metadata,):
                model_output = torch.func.functional_call(
                    self.vllm_model,
                    self.vllm_params_and_buffers,
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
            return new_kv_caches, model_output

        return func
    

    def get_jitted_compute_logits_func(self):
        @jax_jit
        def func(hidden_states: jax.Array):
            logits = torch.func.functional_call(
                self.compute_logits_model,
                self.compute_logits_model_params_and_buffers,
                kwargs={ "hidden_states": hidden_states },
                tie_weights=False,
                strict=True)
            return logits
        
        return func


    def step_func(
        self,
        is_prefill: bool,
        do_sampling: bool,
        kv_caches: List[KVCache],
        input_ids: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        temperatures: jax.Array = None,
        top_ps: jax.Array = None,
        top_ks: jax.Array = None,
        *args,
    ) -> Tuple[List[KVCache], jax.Array, jax.Array]:

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

        new_kv_caches, model_output = self.model_func(
            is_prefill=is_prefill,
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,)
        print(f"model_output.shape={model_output.shape}")
        print(f"type(model_output)={type(model_output)}")
        print(f"model_output={model_output}")

        sample_hidden_states = model_output
        print(f"sample_hidden_states.shape={sample_hidden_states.shape}")
        print(f"type(sample_hidden_states)={type(sample_hidden_states)}")
        print(f"sample_hidden_states={sample_hidden_states}")

        logits = self.compute_logits_func(sample_hidden_states)
        print(f"logits.shape={logits.shape}")
        print(f"type(logits)={type(logits)}")
        print(f"logits={logits}")

        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng,
            self.mesh,
            logits.jax(),
            attention_metadata.seq_lens if is_prefill else attention_metadata.seq_lens+1,
            temperatures,
            top_ps,
            top_ks,
        )

        return new_kv_caches, next_tokens, logits