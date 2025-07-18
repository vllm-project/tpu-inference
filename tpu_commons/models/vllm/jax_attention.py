import functools
from typing import Optional, Tuple

import jax
import torch
import torch.nn
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view
from vllm.attention import Attention as VllmAttention
from vllm.model_executor.models.utils import extract_layer_index

from tpu_commons.models.jax.attention_interface import attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


@functools.partial(
    jax.jit,
    static_argnums=(5, 6, 7, 8,
                    9),  # mesh, scale, head_dim, num_heads, num_kv_heads
    donate_argnums=(0, ),  # donate kv_cache
)
def _jax_attn_func(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    scale: float,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> Tuple[jax.Array, jax.Array]:
    del scale  # Unused for now, as the attention function applies a default scale.

    # Get shapes from vllm
    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_dim * num_heads
    assert k_compute_dim == head_dim * num_kv_heads

    # Convert the shapes from vLLM's convetion to what the attention function expects
    # bs, num_heads, q_len, head_dim
    q = q.reshape(q_len, num_heads, head_dim)
    # bs, num_kv_heads, k_len, head_dim
    k = k.reshape(k_len, num_kv_heads, head_dim)
    v = v.reshape(k_len, num_kv_heads, head_dim)

    new_kv_cache, outputs = attention(
        kv_cache,
        q,
        k,
        v,
        attention_metadata,
        mesh,
    )

    # Convert the shape back to vLLM's convention
    assert outputs.shape[0] == q_len
    assert outputs.shape[1] == num_heads
    assert outputs.shape[2] == head_dim
    outputs = outputs.reshape(q_len, q_compute_dim)

    return new_kv_cache, outputs


class JaxAttention(torch.nn.Module):

    def __init__(
        self,
        vllm_attn: VllmAttention,
        mesh: Mesh,
    ) -> None:
        super().__init__()

        self.num_heads = vllm_attn.num_heads
        self.head_dim = vllm_attn.head_size
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
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        new_kv_cache, outputs = _jax_attn_func(
            vllm_model_wrapper_context.kv_caches[self.layer_idx], jax_view(q),
            jax_view(k), jax_view(v),
            vllm_model_wrapper_context.attention_metadata, self.mesh,
            self.scale, self.head_dim, self.num_heads, self.num_kv_heads)
        vllm_model_wrapper_context.kv_caches[self.layer_idx] = new_kv_cache

        return torch_view(outputs)
