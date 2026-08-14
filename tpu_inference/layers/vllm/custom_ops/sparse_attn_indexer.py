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
"""TPU implementation of the DSA "lightning indexer".

Covers DeepSeek-V3.2 and GLM-5.2 (``glm_moe_dsa``), which share the
``deepseek_v2.py`` DSA code path in vLLM. Two things need replacing on TPU:

* ``Indexer.forward`` calls ``per_token_group_quant_fp8``, a CUDA-only op.
* ``SparseAttnIndexer.forward_native`` raises on any non-CUDA/ROCm/XPU
  platform; it is a ``CustomOp``, so ``register_oot`` is the supported hook.

Cache layout note: vLLM's ``DeepseekV32IndexerCache`` declares
``head_dim = index_head_dim + index_head_dim // quant_block_size * 4``, i.e. a
**4-byte fp32** scale per token, matching the CUDA kernels. The TPU
``streamindex_topk`` kernel instead expects a **1-byte e8m0** scale directly
after the fp8 values (see ``indexer/streamindex_topk.py`` ``load_bkv``). TPU
owns both the write and the read, so we override the spec to the layout the
Pallas kernel wants rather than converting on every access.
"""

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# fp8 e4m3 finite max; the scale is chosen so |q| / scale lands inside this.
_FP8_E4M3_MAX = 448.0


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


class VllmDsaIndexerCache(DeepseekV32IndexerCache):
    """Indexer k-cache using the layout ``streamindex_topk`` reads.

    Record is ``[head_dim fp8 bytes][1 e8m0 scale byte]`` padded to a 128
    multiple, rather than upstream's fp32-scale layout.
    """

    def get_kv_cache_spec(self, vllm_config) -> MLAAttentionSpec:
        # self.head_dim is upstream's fp32-scale width; recover the real head
        # dim, which is what the fp8 payload actually occupies.
        quant_block_size = 128
        payload = self.head_dim - (self.head_dim //
                                   (quant_block_size + 4)) * 4 \
            if self.head_dim % 128 else self.head_dim
        # For the shapes we support (index_head_dim=128, quant_block=128)
        # upstream's head_dim is 132 and the payload is 128.
        payload = 128 if self.head_dim == 132 else payload
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=align_to(payload + 1, 128),
            dtype=torch.uint8,
        )


def quantize_k_to_e8m0_record(k: jax.Array, record_width: int) -> jax.Array:
    """Pack ``k`` into the uint8 record ``streamindex_topk`` expects.

    Args:
        k: bf16/f32 ``[num_tokens, head_dim]``.
        record_width: padded record width (``align_to(head_dim + 1, 128)``).

    Returns:
        uint8 ``[num_tokens, record_width]`` laid out as ``head_dim`` fp8_e4m3
        bytes, then one e8m0 scale byte, then zero padding.
    """
    num_tokens, head_dim = k.shape
    amax = jnp.max(jnp.abs(k.astype(jnp.float32)), axis=-1, keepdims=True)
    # e8m0 can only represent powers of two, so round the exponent up. Guard
    # the all-zero row so log2 does not produce -inf.
    safe = jnp.where(amax > 0, amax, 1.0)
    exp = jnp.ceil(jnp.log2(safe / _FP8_E4M3_MAX))
    scale = jnp.exp2(exp)
    q = (k.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn)

    payload = jax.lax.bitcast_convert_type(q, jnp.uint8).reshape(
        num_tokens, head_dim)
    # e8m0 stores the raw biased exponent; 127 is the bias (2^0 -> 127).
    scale_byte = jnp.clip(exp + 127, 0, 255).astype(jnp.uint8)
    pad = record_width - head_dim - 1
    parts = [payload, scale_byte]
    if pad > 0:
        parts.append(jnp.zeros((num_tokens, pad), jnp.uint8))
    return jnp.concatenate(parts, axis=-1)


def per_token_group_quant_fp8_jax(q: jax.Array, group_size: int):
    """TPU stand-in for vLLM's CUDA ``per_token_group_quant_fp8``.

    Returns ``(q_fp8, scale)`` with one power-of-two scale per group, matching
    ``use_ue8m0=True`` on the CUDA path.
    """
    orig_shape = q.shape
    q = q.reshape(-1, group_size).astype(jnp.float32)
    amax = jnp.max(jnp.abs(q), axis=-1, keepdims=True)
    safe = jnp.where(amax > 0, amax, 1.0)
    scale = jnp.exp2(jnp.ceil(jnp.log2(safe / _FP8_E4M3_MAX)))
    q_fp8 = (q / scale).astype(jnp.float8_e4m3fn)
    return q_fp8.reshape(orig_shape), scale.reshape(orig_shape[:-1])


@SparseAttnIndexer.register_oot
class VllmSparseAttnIndexer(SparseAttnIndexer):
    """TPU top-k selection over the paged indexer k-cache.

    Unlike the CUDA op, this **returns** the indices instead of scattering into
    ``topk_indices_buffer``: under torchax the model is traced functionally, so
    in-place mutation of a shared buffer across layers is not a reliable way to
    hand values from a "full" layer to the "shared" layers that follow.
    """

    def forward_native(self, hidden_states, q_quant, k, weights):
        return self.forward_tpu(hidden_states, q_quant, k, weights)

    def forward_tpu(self, hidden_states, q_quant, k, weights):
        from vllm.forward_context import get_forward_context

        from tpu_inference.models.vllm.vllm_model_wrapper_context import \
            get_vllm_model_wrapper_context

        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh

        cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache.prefix]
        kv_cache = wrapper_ctx.kv_caches[cache_index]
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.k_cache.prefix]

        if isinstance(q_quant, tuple):
            q_values, _ = q_quant
        else:
            q_values = q_quant

        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)

        def _topk(q, w, cache_kv, seq_lens, page_indices, cu_q_lens,
                  distribution):
            return streamindex_topk(
                q=q,
                indexer_weights=w,
                cache_kv=cache_kv,
                seq_lens=seq_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                k=self.topk_tokens,
                # DSA has no compressor: the indexer scores every token.
                compression_ratio=1,
                num_kv_pages_per_block=(3, 2, 2),
                num_queries_per_block=(1, 128, 128),
            )

        topk_indices = jax.shard_map(
            _topk,
            mesh=mesh,
            in_specs=(data_spec, data_spec, cache_spec, data_spec, data_spec,
                      data_spec, data_spec),
            out_specs=data_spec,
            check_vma=False,
        )(
            jax_view(q_values),
            jax_view(weights),
            kv_cache,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )
        return torch_view(topk_indices)
