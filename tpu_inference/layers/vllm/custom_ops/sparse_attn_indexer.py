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

import sys

import jax
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

from tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk import \
    streamindex_topk
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def _tpu_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_ub: torch.Tensor | None = None,
    use_ue8m0: bool = False,
):
    fp8_max = torch.finfo(dtype).max
    x_reshaped = x.view(-1, group_size)
    amax = torch.amax(torch.abs(x_reshaped), dim=-1,
                      keepdim=True).clamp(min=eps)
    x_s = amax / fp8_max
    x_q = (x_reshaped / x_s).to(dtype).view_as(x)
    if not column_major_scales:
        x_s = x_s.view(x.shape[:-1] + (x.shape[-1] // group_size, ))
    else:
        x_s = x_s.view(x.shape[0], x.shape[1] // group_size)
    return x_q, x_s


def _apply_fp8_utils_patch():
    if "vllm.model_executor.layers.quantization.utils.fp8_utils" in sys.modules:
        sys.modules[
            "vllm.model_executor.layers.quantization.utils.fp8_utils"].per_token_group_quant_fp8 = _tpu_per_token_group_quant_fp8
    if "vllm.model_executor.models.deepseek_v2" in sys.modules:
        sys.modules[
            "vllm.model_executor.models.deepseek_v2"].per_token_group_quant_fp8 = _tpu_per_token_group_quant_fp8


_apply_fp8_utils_patch()


@SparseAttnIndexer.register_oot
class VllmSparseAttnIndexer(SparseAttnIndexer):
    """TPU-compatible SparseAttnIndexer custom op using streamindex_topk."""

    def forward_tpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _apply_fp8_utils_patch()
        num_tokens = hidden_states.shape[0]

        if isinstance(q_quant, tuple):
            q_quant = q_quant[0]

        try:
            wrapper_ctx = get_vllm_model_wrapper_context()
            kv_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
                self.k_cache.prefix]
            kv_cache = wrapper_ctx.kv_caches[kv_cache_index]
            mesh = wrapper_ctx.mesh
            attn_metadata_dict = get_forward_context().attn_metadata
            attn_metadata = attn_metadata_dict[self.k_cache.prefix]
        except Exception:
            if self.topk_indices_buffer is not None:
                return self.topk_indices_buffer[:num_tokens]
            return torch.zeros(
                (num_tokens, self.topk_tokens),
                dtype=torch.int32,
                device=hidden_states.device,
            )

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
        out_specs = data_spec

        topk_tokens = self.topk_tokens

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
                num_kv_pages_per_block=1,
                num_queries_per_block=1,
            )

        topk_indices_jax = jax.shard_map(
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

        return torch_view(topk_indices_jax)
