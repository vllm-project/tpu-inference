from typing import List, Optional, Tuple
import tempfile
import functools
import humanize


import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.typing import PRNGKey

import torch
import torch.nn as torch_nn
from torch.utils import _pytree as pytree

import torchax
from torchax.interop import call_jax, extract_all_buffers, jax_jit
from torchax.ops.mappings import j2t_dtype


from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.attention import (sharded_flash_attention,
                                                     sharded_paged_attention,
                                                     update_cache)
from tpu_commons.models.jax.layers.chunked_prefill_attention import (
    sharded_chunked_prefill_attention, sharded_chunked_prefill_update_cache)
from tpu_commons.models.jax.layers.sampling import sample

from tpu_commons.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
    set_vllm_model_wrapper_context,
)


from vllm.attention import Attention as VllmAttention
from vllm.config import VllmConfig
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import (extract_layer_index)
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized
from vllm.sequence import IntermediateTensors


KVCache = Tuple[jax.Array, jax.Array]


@functools.partial(
        jax.jit,
        static_argnums=(0, 6, 7, 8),  # is_prefill, mesh, num_heads, num_kv_heads
        donate_argnums=(1, ),  # donate kv_cache
)
def _jax_attn_func(
        is_prefill: bool,
        kv_cache: KVCache,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
        num_heads: int,
        num_kv_heads: int,
) -> Tuple[KVCache, jax.Array]:
    # TODO: fix
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)

    md = attention_metadata
    k_cache, v_cache = kv_cache
    if md.chunked_prefill_enabled:
        k_cache = sharded_chunked_prefill_update_cache(mesh)(
            k_cache, md.kv_cache_write_indices, k, md.num_decode_seqs)
        v_cache = sharded_chunked_prefill_update_cache(mesh)(
            v_cache, md.kv_cache_write_indices, v, md.num_decode_seqs)
        outputs = sharded_chunked_prefill_attention(mesh)(
            q,
            k_cache,
            v_cache,
            attention_metadata.decode_lengths,
            attention_metadata.decode_page_indices,
            attention_metadata.num_decode_seqs,
            attention_metadata.prefill_lengths,
            attention_metadata.prefill_page_indices,
            attention_metadata.prefill_query_start_offsets,
            attention_metadata.num_prefill_seqs,
        )
    else:
        k_cache = update_cache(is_prefill, k_cache,
                               md.kv_cache_write_indices, k)
        v_cache = update_cache(is_prefill, v_cache,
                               md.kv_cache_write_indices, v)
        if is_prefill:
            # (B, N, T, H)
            # TODO(xiang): support MQA and GQA
            if num_kv_heads != num_heads:
                k = jnp.repeat(k, num_heads // num_kv_heads, axis=1)
                v = jnp.repeat(v, num_heads // num_kv_heads, axis=1)
            outputs = sharded_flash_attention(mesh)(q, k, v)
        else:
            # (B, N, H)
            q = jnp.squeeze(q, 2)
            outputs = sharded_paged_attention(mesh)(q, k_cache, v_cache, md.seq_lens,
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
        self.mesh = mesh

    
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
            vllm_model_wrapper_context.is_prefill,
            vllm_model_wrapper_context.kv_caches[self.layer_idx],
            q, k, v,
            vllm_model_wrapper_context.attention_metadata,
            self.mesh,
            self.num_heads,
            self.num_kv_heads)
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


class ModelForLogits(torch_nn.Module):
    def __init__(self, vllm_model: torch_nn.Module):
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