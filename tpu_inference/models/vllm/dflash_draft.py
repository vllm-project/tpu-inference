# Copyright 2026 Google LLC
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
"""DFlash draft forward for vLLM (torchax) DFlash draft models.

vLLM's ``DFlashQwen3ForCausalLM`` expects the context K/V of every draft layer
to be pre-inserted into the paged KV cache (via CUDA custom ops in
``precompute_and_store_context_kv``) before a plain forward over the noise
block runs. Neither step maps onto the TPU attention backend: the custom ops
do not exist and the backend's decoder attention is always causal.

This module re-expresses the same computation with the vLLM model's own
sub-modules (projections, norms, RoPE, MLP) and tpu-inference's paged
attention, mirroring the JAX-native ``tpu_inference.models.jax.dflash``:

1. For each draft layer, project the (``fc``-combined, ``hidden_norm``-ed)
   target hidden states of the newly accepted tokens to K/V, apply K-norm and
   RoPE, and write them into the layer's KV cache.
2. Run the noise block through the layer, writing its K/V after the context
   and attending over the whole cache (non-causal unless the layer says
   otherwise).
"""

from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view

from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

_REQUIRED_ATTN_ATTRS = ("qkv_proj", "o_proj", "q_norm", "k_norm", "rotary_emb",
                        "attn", "num_heads", "num_kv_heads", "head_dim",
                        "q_size", "kv_size", "scaling")


def validate_dflash_draft_model(vllm_model: torch.nn.Module,
                                mesh: Optional[Mesh] = None) -> None:
    """Raises if ``vllm_model`` can't run as a DFlash draft on this path."""
    inner = getattr(vllm_model, "model", None)
    layers = getattr(inner, "layers", None)
    if inner is None or layers is None or not hasattr(inner, "hidden_norm"):
        raise NotImplementedError(
            f"{type(vllm_model).__name__} is not supported as a DFlash draft "
            "model on the vllm (torchax) path: expected a vLLM DFlash model "
            "with `model.layers` and `model.hidden_norm`.")
    uses_dcp = mesh is not None and mesh.shape.get("dcp", 1) > 1
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        missing = [
            name for name in _REQUIRED_ATTN_ATTRS
            if attn is None or not hasattr(attn, name)
        ]
        if missing:
            raise NotImplementedError(
                f"DFlash draft layer {i} of {type(vllm_model).__name__} is "
                f"missing {missing}; unsupported on the vllm (torchax) path.")
        if getattr(attn, "sliding_window", None) is not None:
            raise NotImplementedError(
                "Sliding-window DFlash draft layers are not supported on the "
                "vllm (torchax) path yet.")
        if getattr(attn, "attention_sink_bias", None) is not None:
            raise NotImplementedError(
                "DFlash draft layers with attention sinks are not supported "
                "on the vllm (torchax) path yet.")
        # The head_dim=64 RPA kernel and the DCP path ignore
        # use_causal_mask=False, which would silently make the noise block
        # causal and degrade the drafts.
        if not getattr(attn, "causal", False) and (attn.head_dim == 64
                                                   or uses_dcp):
            raise NotImplementedError(
                f"DFlash draft layer {i} needs non-causal attention, which "
                "the head_dim=64 attention kernel and decode context "
                "parallelism do not support yet.")


def _split_heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    return x.view(*x.shape[:-1], x.shape[-1] // head_dim, head_dim)


def _project_kv(attn: torch.nn.Module, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Context K/V for one layer: qkv projection, K-norm, RoPE, v-scale."""
    qkv, _ = attn.qkv_proj(hidden_states)
    _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    k_shape = k.shape
    k = attn.k_norm(_split_heads(k, attn.head_dim)).view(k_shape)
    # RoPE is applied to K only; K stands in for the query argument.
    _, k = attn.rotary_emb(positions, k, k)
    if getattr(attn, "v_scale", None) is not None:
        v = v * attn.v_scale
    return k, v


def _project_qkv(
    attn: torch.nn.Module, hidden_states: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise Q/K/V for one layer, matching ``DFlashQwen3Attention.forward``."""
    qkv, _ = attn.qkv_proj(hidden_states)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = attn.q_norm(_split_heads(q, attn.head_dim)).view(q_shape)
    k = attn.k_norm(_split_heads(k, attn.head_dim)).view(k_shape)
    q, k = attn.rotary_emb(positions, q, k)
    if getattr(attn, "v_scale", None) is not None:
        v = v * attn.v_scale
    return q, k, v


def _paged_attention(
    attn: torch.nn.Module,
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    use_causal_mask: bool,
) -> tuple[jax.Array, jax.Array]:
    """Writes K/V into ``kv_cache`` and attends ``q`` over the cache.

    Returns the updated cache and the (T, num_heads, head_dim) output.
    """
    mesh = get_vllm_model_wrapper_context().mesh
    q = q.reshape(q.shape[0], attn.num_heads, attn.head_dim)
    k = k.reshape(k.shape[0], attn.num_kv_heads, attn.head_dim)
    v = v.reshape(v.shape[0], attn.num_kv_heads, attn.head_dim)

    k_scale = v_scale = None
    impl = getattr(attn.attn, "impl", None)
    kv_cache_quantized_dtype = getattr(impl, "kv_cache_quantized_dtype", None)
    if kv_cache_quantized_dtype:
        k_scale = attn.attn._k_scale_float
        v_scale = attn.attn._v_scale_float
        k, v = quantize_kv(kv_cache_quantized_dtype, k, v, k_scale, v_scale)

    return attention(
        kv_cache,
        q,
        k,
        v,
        attention_metadata,
        mesh,
        head_dim_original=attn.head_dim,
        sm_scale=attn.scaling,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=True,
        use_causal_mask=use_causal_mask,
    )


def dflash_draft_forward(
    vllm_model: torch.nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    target_hidden: torch.Tensor,
    target_positions: torch.Tensor,
    target_query_start_loc: jax.Array,
    attention_metadata: AttentionMetadata,
) -> torch.Tensor:
    """Runs one DFlash draft step of a vLLM DFlash model under torchax.

    Must be called inside ``set_vllm_model_wrapper_context`` with the draft
    layers present in ``layer_name_to_kvcache_index``; the updated KV caches
    are written back to that context.

    Args:
        vllm_model: vLLM DFlash draft model (e.g. ``DFlashQwen3ForCausalLM``).
        input_ids: (T_noise,) noise block token ids, ``block_size`` per request.
        positions: (T_noise,) positions of the noise block tokens.
        target_hidden: (T_ctx, D) target aux hidden states already combined by
            ``combine_hidden_states`` (``fc``), not yet ``hidden_norm``-ed.
        target_positions: (T_ctx,) positions of the context tokens.
        target_query_start_loc: per-request offsets of the newly accepted
            context tokens within ``target_hidden``.
        attention_metadata: draft attention metadata; ``seq_lens`` is the
            context length after appending the accepted tokens and
            ``query_start_loc`` delimits the noise blocks.

    Returns:
        (T_noise, D) final-norm hidden states of the noise block.
    """
    context = get_vllm_model_wrapper_context()
    inner = vllm_model.model
    md = attention_metadata

    num_reqs = md.seq_lens.shape[0]
    block_size = input_ids.shape[0] // num_reqs
    context_md = replace(md, query_start_loc=target_query_start_loc)
    noise_md = replace(md, seq_lens=md.seq_lens + block_size)

    context_states = inner.hidden_norm(target_hidden)
    num_ctx_tokens = context_states.shape[0]

    hidden_states = inner.embed_input_ids(input_ids)
    residual = None
    for layer in inner.layers:
        attn = layer.self_attn
        kv_cache_index = context.layer_name_to_kvcache_index[
            attn.attn.layer_name]
        kv_cache = context.kv_caches[kv_cache_index]

        # 1. Insert the context K/V of the newly accepted tokens. The query is
        # a dummy: only the cache update of this call is used.
        k_ctx, v_ctx = _project_kv(attn, context_states, target_positions)
        k_ctx, v_ctx = jax_view(k_ctx), jax_view(v_ctx)
        q_ctx = jnp.zeros((num_ctx_tokens, attn.num_heads, attn.head_dim),
                          dtype=k_ctx.dtype)
        kv_cache, _ = _paged_attention(attn,
                                       kv_cache,
                                       q_ctx,
                                       k_ctx,
                                       v_ctx,
                                       context_md,
                                       use_causal_mask=True)

        # 2. Noise block: write its K/V after the context and attend over all.
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual)
        q, k, v = _project_qkv(attn, hidden_states, positions)
        q, k, v = jax_view(q), jax_view(k), jax_view(v)
        kv_cache, attn_out = _paged_attention(
            attn,
            kv_cache,
            q,
            k,
            v,
            noise_md,
            use_causal_mask=bool(getattr(attn, "causal", False)))
        context.kv_caches[kv_cache_index] = kv_cache

        attn_out = attn_out.reshape(attn_out.shape[0],
                                    attn.num_heads * attn.head_dim)
        hidden_states, _ = attn.o_proj(torch_view(attn_out.astype(q.dtype)))

        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

    hidden_states, _ = inner.norm(hidden_states, residual)
    return hidden_states


def dflash_draft_call_kwargs(
    input_ids: jax.Array,
    target_hidden_states: Any,
    attention_metadata: AttentionMetadata,
) -> dict[str, Any]:
    """Builds the ``dflash_draft_forward`` kwargs from JAX-land inputs."""
    target_hidden, target_query_start_loc, target_positions = (
        target_hidden_states)
    return {
        "input_ids": torch_view(input_ids),
        "positions": torch_view(attention_metadata.input_positions),
        "target_hidden": torch_view(target_hidden),
        "target_positions": torch_view(target_positions),
        "target_query_start_loc": target_query_start_loc,
        "attention_metadata": attention_metadata,
    }
