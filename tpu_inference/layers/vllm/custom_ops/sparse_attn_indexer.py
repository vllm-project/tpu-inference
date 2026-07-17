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
"""TPU-compatible DeepSeek-V2 / GLM-5.2 Sparse Attention Indexer."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import \
    quantize_fp8_ue8m0
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


@SparseAttnIndexer.register_oot
class VllmSparseAttnIndexer(SparseAttnIndexer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # Get wrapper context and kv cache
        wrapper_ctx = get_vllm_model_wrapper_context()
        kv_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache.prefix]
        kv_cache = wrapper_ctx.kv_caches[kv_cache_index]
        mesh = wrapper_ctx.mesh

        # Get JAX views
        k_jax = jax_view(k)
        
        # 1. Quantize k to FP8 and get UE8M0 scale
        q_fp8, scale_ue8m0 = quantize_fp8_ue8m0(k_jax, block_size=self.quant_block_size)
        
        # Bitcast to raw bytes (uint8)
        q_bytes = jax.lax.bitcast_convert_type(q_fp8, jnp.uint8)
        scale_bytes = jax.lax.bitcast_convert_type(scale_ue8m0, jnp.uint8)
        
        packed = jnp.concatenate([q_bytes, scale_bytes], axis=-1)  # [num_tokens, 129]

        # 2. Update KV cache with packed values at slot_mapping indices
        attn_metadata_raw = get_forward_context().attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw[self.k_cache.prefix]
        elif isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0][self.k_cache.prefix]
        else:
            attn_metadata = attn_metadata_raw
        
        slot_mapping = attn_metadata.input_positions  # In TPU AttentionMetadata, slot_mapping is input_positions
        mask = (slot_mapping >= 0)
        
        block_indices = jnp.where(mask, slot_mapping // 1024, 0)
        offsets = jnp.where(mask, slot_mapping % 1024, 0)
        offset_x = offsets // 32
        offset_y = offsets % 32
        
        packed = jnp.where(mask[:, None], packed, 0)
        
        # Update the cache array functionally
        new_kv_cache = kv_cache.at[block_indices, offset_x, offset_y, :129].set(packed)
        wrapper_ctx.kv_caches[kv_cache_index] = new_kv_cache

        # 3. Retrieve Top-K indices using streamindex_topk
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
                compression_ratio=1,  # compression ratio of indexer cache is 1
                num_kv_pages_per_block=1,
                num_queries_per_block=1,
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
            new_kv_cache,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )

        return torch_view(topk_indices)
