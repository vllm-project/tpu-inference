# pytype: skip-file
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-compatible DeepSeek-V4 Lightning Indexer."""

import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import (DeepseekV4Indexer,
                                               DeepseekV4IndexerCache)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# =====================================================================
# IMPORT TPU CUSTOM OPS TO TRIGGER vLLM @register_oot DECORATORS
# =====================================================================
from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.deepseek_v4.rope import rope_quant
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.custom_ops.experimental.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def fused_indexer_q_rope_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
    mesh,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE and dynamically quantizes the queries
  """
    data_spec = P(ShardingAxisName.ATTN_DATA)

    q_quant_jax, q_scales_jax = jax.shard_map(
        functools.partial(rope_quant, quant_dtype=jnp.float8_e4m3fn),
        mesh=mesh,
        in_specs=(data_spec, data_spec, P()),
        out_specs=(data_spec, data_spec),
        check_vma=False,
    )(jax_view(q), jax_view(positions), jax_view(rotary_emb.cos_sin_cache))

    # Note: vLLM's implementation rounds the scale factors up to the
    # next power of 2, but the standard division scale returned by the kernel
    # (abs_max / dtype_max, one per row) is sufficient here.
    return torch_view(q_quant_jax), torch_view(q_scales_jax)


class VllmDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    """TPU-compatible indexer KV cache.

  On TPU the indexer KV cache is allocated and tracked outside of vLLM's
  attention-spec machinery, so we suppress the base ``get_kv_cache_spec`` to
  avoid registering a spec for this layer.
  """

    def get_kv_cache_spec(self, vllm_config):
        # In DSV4 FP8 indexer cache format
        # 128 fp8, 1 e8m0 scale packed as uint8
        # TODO: consider put the scale as a separate Array / Tensor
        # to reduce large space waste due to padding.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=align_to(128 + 1, 128),
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=None,
        )


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
    """TPU-compatible DeepSeek-V4 Lightning Indexer with StreamIndex.
    It uses `streamindex_topk` to compute top-k token indices over a PagedAttention KV cache.
  """

    def __init__(self, *args, **kwargs):
        # The base ctor builds ``self.compressor`` and ``self.k_cache`` from the
        # vLLM upstream classes. Temporarily rebind the module-level symbols
        # the base ctor references so it constructs the TPU subclasses directly.
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_cache = dsv4_attention.DeepseekV4IndexerCache
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4IndexerCache = VllmDeepseekV4IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4IndexerCache = orig_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        slot_mapping: Optional[torch.Tensor] = None,
    ):

        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh

        q, _ = self.wq_b(query)
        q = q.view(-1, self.n_head, self.head_dim)

        q_quant, q_scales = fused_indexer_q_rope_quant(q, positions,
                                                       rotary_emb, mesh)

        # Fold the query quantization scales into the weights.
        weights = (indexer_weights.to(q.dtype) * self.softmax_scale *
                   (self.n_head**-0.5) * q_scales)

        self.compressor(hidden_states, positions, rotary_emb)

        kv_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache.prefix]
        kv_cache = wrapper_ctx.kv_caches[kv_cache_index]
        attn_metadata_dict = get_forward_context().attn_metadata
        attn_metadata = attn_metadata_dict[self.k_cache.prefix]

        # All array inputs and the output are sharded along the leading axis on
        # ShardingAxisName.ATTN_DATA; the kernel runs per-shard inside the
        # shard_map. Static kwargs are captured in the closure.
        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        in_specs = (
            data_spec,  # q
            data_spec,  # indexer_weights
            cache_spec,  # cache_kv
            data_spec,  # seq_lens
            data_spec,  # page_indices
            data_spec,  # cu_q_lens
            data_spec,  # distribution
        )
        out_specs = data_spec  # topk_indices

        def _streamindex_topk(q, indexer_weights, cache_kv, seq_lens,
                              page_indices, cu_q_lens, distribution):
            return streamindex_topk(
                q=q,
                indexer_weights=indexer_weights,
                cache_kv=cache_kv,
                seq_lens=seq_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                k=self.topk_tokens,
                compression_ratio=self.compress_ratio,
                # TODO(hwanginho): Tune num_kv_pages_per_block, num_queries_per_block
                num_kv_pages_per_block=(3, 2, 2),
                num_queries_per_block=(1, 128, 128),
            )

        topk_indices = jax.shard_map(
            _streamindex_topk,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            jax_view(q_quant),
            jax_view(weights),
            kv_cache,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )

        return topk_indices
