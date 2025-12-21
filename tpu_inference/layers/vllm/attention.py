# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        #TODO (kyuyeunk): Shard the sinks along num_heads dim
        if self.sinks is not None:
            sinks = t2j(self.sinks, use_dlpack=False)
            sinks = torch_view(sinks.astype(jnp.float32))
            self.sinks = torch.nn.Parameter(sinks, requires_grad=False)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "PallasAttentionBackendImpl")

        if kv_cache.numel():
            raise RuntimeError(
                "KV cache from vLLM Attention layer should be empty but has "
                "the size of %s.", kv_cache.numel())

        del kv_cache  # Use kv_cache from vllm wrapper context values instead.

        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
            layer.layer_name]
        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]

        mesh = vllm_model_wrapper_context.mesh

        query, key, value = jax_view(query), jax_view(key), jax_view(value)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            key, value = quantize_kv(self.kv_cache_quantized_dtype, key, value,
                                     layer._k_scale_float,
                                     layer._v_scale_float)
            # TODO(kyuyeunk): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = layer._q_scale_float
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float

        sinks = jax_view(self.sinks)

        new_kv_cache, outputs = _jax_attn_func(
            kv_cache,
            query,
            key,
            value,
            sinks,
            attn_metadata,
            mesh,
            self.scale,
            self.head_size,
            self.num_heads,
            self.num_kv_heads,
            q_scale,
            k_scale,
            v_scale,
            self.sliding_window,
        )
        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        return torch_view(outputs)


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "scale",
        "head_size",
        "num_heads",
        "num_kv_heads",
        "q_scale",
        "k_scale",
        "v_scale",
        "sliding_window",
    ),
    donate_argnames=("kv_cache"),
)
def _jax_attn_func(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sinks: jax.Array | None,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    scale: float,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sliding_window: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    del scale  # Unused for now, as the attention function applies a default scale.

    # Get shapes from vllm
    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_size * num_heads
    assert k_compute_dim == head_size * num_kv_heads

    # Convert the shapes from vLLM's convetion to what the attention function expects
    # bs, num_heads, q_len, head_size
    q = q.reshape(q_len, num_heads, head_size)
    # bs, num_kv_heads, k_len, head_size
    k = k.reshape(k_len, num_kv_heads, head_size)
    v = v.reshape(k_len, num_kv_heads, head_size)

    new_kv_cache, outputs = attention(
        kv_cache,
        q,
        k,
        v,
        attention_metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sinks=sinks,
        attention_chunk_size=sliding_window,
    )

    # Convert the shape back to vLLM's convention
    assert outputs.shape[0] == q_len
    assert outputs.shape[1] == num_heads
    assert outputs.shape[2] == head_size
    outputs = outputs.reshape(q_len, q_compute_dim)

    return new_kv_cache, outputs
