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
"""TPU-compatible DeepSeek-V4 Lightning Indexer."""

import torch
import torch.nn as nn
from torchax.interop import jax_view
from torchax.interop import torch_view
from vllm.models.deepseek_v4.attention import DeepseekV4Indexer
from vllm.forward_context import get_forward_context
from tpu_inference.layers.vllm.custom_ops.fused_indexer_topk import streamindex_chunked_topk


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
    """TPU-compatible DeepSeek-V4 Lightning Indexer with StreamIndex.
    This class overrides the forward method of DeepseekV4Indexer to provide a TPU-compatible
    implementation using JAX interop. It uses `streamindex_chunked_topk` to compute
    top-k token indices over a PagedAttention KV cache with dynamic sequence padding
    and bucketization to avoid excessive JAX recompilations.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(query)
        q = q.view(-1, self.n_head, self.head_dim)

        # TODO: Implement q-rope-quant natively.
        # q_quant, weights = fused_indexer_q_rope_quant(...)

        q_quant = q

        weights = indexer_weights.to(q.dtype) * self.softmax_scale * (self.n_head**-0.5)

        attn_metadata_dict = get_forward_context().attn_metadata
        kv_cache_tensor = self.k_cache.kv_cache
        actual_num_tokens = hidden_states.shape[0]

        c_s = getattr(self, "c_s", 8192)
        c_t = getattr(self, "c_t", 2048)
        page_size = self.k_cache.cache_config.block_size

        assert kv_cache_tensor.is_contiguous(), "KV cache must be physically contiguous for TorchAX."
        kv_cache_jax = jax_view(kv_cache_tensor)

        def run_jax_kernel(q_slice, w_slice, bt_slice, sl_slice, csl_slice, bucket_tokens):
            n_tokens = q_slice.shape[0]
            n_batch, current_blocks = bt_slice.shape

            static_tokens = ((n_tokens + bucket_tokens - 1) // bucket_tokens) * bucket_tokens
            pad_tokens = static_tokens - n_tokens

            bucket_batch = 16
            static_batch = ((n_batch + bucket_batch - 1) // bucket_batch) * bucket_batch
            pad_batch = static_batch - n_batch

            bucket_blocks = max(16, 1 << (current_blocks - 1).bit_length()) if current_blocks > 0 else 16
            pad_blocks = bucket_blocks - current_blocks

            if pad_tokens > 0:
                q_slice = torch.nn.functional.pad(q_slice, (0, 0, 0, 0, 0, pad_tokens))
                w_slice = torch.nn.functional.pad(w_slice, (0, 0, 0, pad_tokens))

            if pad_batch > 0 or pad_blocks > 0:
                bt_slice = torch.nn.functional.pad(bt_slice, (0, pad_blocks, 0, pad_batch))

            if pad_batch > 0:
                sl_slice = torch.nn.functional.pad(sl_slice, (0, pad_batch))
                csl_slice = torch.nn.functional.pad(csl_slice, (0, pad_batch))
                csl_slice[-pad_batch:] = csl_slice[n_batch]

            bt_slice = bt_slice.to(torch.int32)
            sl_slice = sl_slice.to(torch.int32)
            csl_slice = csl_slice.to(torch.int32)

            q_jax = jax_view(q_slice.contiguous())
            w_jax = jax_view(w_slice.contiguous())
            bt_jax = jax_view(bt_slice.contiguous())
            sl_jax = jax_view(sl_slice.contiguous())
            csl_jax = jax_view(csl_slice.contiguous())

            topk_jax = streamindex_chunked_topk(
                query_projection=q_jax,
                kv_cache=kv_cache_jax,
                block_table=bt_jax,
                seq_lens=sl_jax,
                cu_seq_lens=csl_jax,
                indexer_weights=w_jax,
                k=self.topk_tokens,
                compression_ratio=self.compress_ratio,
                c_s=c_s,
                c_t=c_t
            )
            return torch_view(topk_jax)[:n_tokens, :]

        if not isinstance(attn_metadata_dict, dict):
            # Tracing must simulate a realistic max_blocks to reserve accurate memory.
            max_seq_len = getattr(self, "max_seq_len", 8192)
            max_possible_blocks = (max_seq_len + page_size - 1) // page_size

            bt_dummy = torch.zeros((1, max_possible_blocks), dtype=torch.int32, device=q.device)
            sl_dummy = torch.tensor([actual_num_tokens], dtype=torch.int32, device=q.device)
            csl_dummy = torch.tensor([0, actual_num_tokens], dtype=torch.int32, device=q.device)

            topk_indices = run_jax_kernel(q_quant, weights, bt_dummy, sl_dummy, csl_dummy, 256)
        else:
            attn_metadata = attn_metadata_dict[self.k_cache.prefix]

            q_lens = []
            seq_lens_list = []
            block_tables_list = []

            if attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.decode
                # Use decode_lens if available (handles speculative decoding), else default to 1 token per seq
                decode_lens = getattr(decode_meta, "decode_lens", torch.ones_like(decode_meta.seq_lens))
                q_lens.append(decode_lens.to(q.device))
                seq_lens_list.append(decode_meta.seq_lens.to(q.device))
                block_tables_list.append(decode_meta.block_table.to(q.device))

            if attn_metadata.num_prefills > 0:
                for chunk in attn_metadata.prefill.chunks:
                    q_len_per_req = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                    q_lens.append(q_len_per_req.to(q.device))
                    seq_lens_list.append(chunk.seq_lens.to(q.device))
                    block_tables_list.append(chunk.block_table.to(q.device))

            if q_lens:
                all_q_lens = torch.cat(q_lens)
                combined_csl = torch.cat([
                    torch.tensor([0], dtype=torch.int32, device=q.device),
                    torch.cumsum(all_q_lens, dim=0, dtype=torch.int32)
                ]).to(q.device)

                combined_sl = torch.cat(seq_lens_list).to(q.device)

                max_blocks = max([bt.shape[1] for bt in block_tables_list])
                padded_block_tables = []
                for bt in block_tables_list:
                    pad_size = max_blocks - bt.shape[1]
                    if pad_size > 0:
                        bt = torch.nn.functional.pad(bt, (0, pad_size))
                    padded_block_tables.append(bt)
                combined_bt = torch.cat(padded_block_tables, dim=0).to(q.device)
            else:
                combined_bt = torch.zeros((1, 1), dtype=torch.int32, device=q.device)
                combined_sl = torch.tensor([actual_num_tokens], dtype=torch.int32, device=q.device)
                combined_csl = torch.tensor([0, actual_num_tokens], dtype=torch.int32, device=q.device)

            bucket_tokens = 256 if attn_metadata.num_prefills > 0 else 16

            topk_indices = run_jax_kernel(
                q_slice=q_quant[:actual_num_tokens],
                w_slice=weights[:actual_num_tokens],
                bt_slice=combined_bt,
                sl_slice=combined_sl,
                csl_slice=combined_csl,
                bucket_tokens=bucket_tokens
            )

        # TODO: Implement a TPU-native DeepseekCompressor to compute the current 
        # token's keys, and write a native TPU scatter kernel to save those keys 
        # into `self.k_cache.kv_cache` (similar to ops.indexer_k_quant_and_cache).

        return topk_indices