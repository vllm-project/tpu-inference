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

import jax
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    if not hasattr(DeepseekV32IndexerCache, "kv_sharing_target_layer_name"):
        DeepseekV32IndexerCache.kv_sharing_target_layer_name = None
except Exception:
    pass

from tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk import \
    streamindex_topk
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


@SparseAttnIndexer.register_oot
class VllmSparseAttnIndexer(SparseAttnIndexer):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        wrapper_ctx = get_vllm_model_wrapper_context()
        prefix = getattr(self.k_cache, "prefix", None)
        if prefix is None or prefix not in wrapper_ctx.layer_name_to_kvcache_index:
            return self.topk_indices_buffer

        kv_cache_index = wrapper_ctx.layer_name_to_kvcache_index[prefix]
        kv_cache = wrapper_ctx.kv_caches[kv_cache_index]
        mesh = wrapper_ctx.mesh

        attn_metadata_dict = get_forward_context().attn_metadata
        if prefix not in attn_metadata_dict:
            return self.topk_indices_buffer
        attn_metadata = attn_metadata_dict[prefix]

        if isinstance(q_quant, tuple):
            q_values = q_quant[0]
        else:
            q_values = q_quant

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

        topk_tokens = getattr(self, "topk_tokens", 2048)
        compress_ratio = getattr(self, "compress_ratio", 1)

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
                k=topk_tokens,
                compression_ratio=compress_ratio,
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
            jax_view(q_values),
            jax_view(weights),
            kv_cache,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )

        out_tensor = torch_view(topk_indices)
        if self.topk_indices_buffer is not None:
            self.topk_indices_buffer[:out_tensor.shape[0], :out_tensor.
                                     shape[1]] = out_tensor
            return self.topk_indices_buffer
        return out_tensor
