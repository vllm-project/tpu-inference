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

from typing import Optional, Tuple

import jax.numpy as jnp
import torch
import torch.nn as nn
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Indexer

# =====================================================================
# IMPORT TPU CUSTOM OPS TO TRIGGER vLLM @register_oot DECORATORS
# =====================================================================
from tpu_inference.kernels.experimental.deepseek_v4.compressor import \
    compressor_forward_indexer
from tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk import \
    streamindex_topk
from tpu_inference.layers.common import quantization


def fused_indexer_q_rope_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE and dynamically quantizes the queries

  Args:
      q: Un-rotated query tensor of shape [num_tokens, num_heads, head_dim]
      positions: Token positions of shape [num_tokens]
      rotary_emb: The vLLM RoPE CustomOp module (Intercepted by TPU rope.py)

  Returns:
      q_quant: Int8 quantized, RoPE-applied queries
      q_scales: Quantization scales
  """

    # Apply the custom rotary embedding op directly in-place on the query tensor.
    q, _ = rotary_emb(positions, q)

    # Bridge to JAX, call JAX quantize_tensor, and bridge back to PyTorch
    q_jax = jax_view(q)
    q_quant_jax, q_scales_jax = quantization.quantize_tensor(
        q_jax, jnp.float8_e4m3fn)
    q_quant = torch_view(q_quant_jax)
    q_scales = torch_view(q_scales_jax)

    # Note: vLLM's implementation rounds the scale factors up to the
    # next power of 2, but the standard division scale returned by quantize_tensor
    # is sufficient here.
    return q_quant, q_scales.squeeze(-1)


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
    """TPU-compatible DeepSeek-V4 Lightning Indexer with StreamIndex.

  This class overrides the forward method of DeepseekV4Indexer to provide a
  TPU-compatible implementation using JAX interop. It uses
  `streamindex_topk` to compute top-k token indices over a
  PagedAttention KV cache.
  """

    # pylint: disable=unused-argument
    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        actual_num_tokens = hidden_states.shape[0]

        q, _ = self.wq_b(query)
        q = q.view(-1, self.n_head, self.head_dim)

        q_quant, q_scales = fused_indexer_q_rope_quant(q, positions,
                                                       rotary_emb)

        # Fold the query quantization scales into the weights
        weights = (indexer_weights.to(q.dtype) * self.softmax_scale *
                   (self.head_dim**-0.5) * q_scales)

        attn_metadata_dict = get_forward_context().attn_metadata
        attn_metadata = attn_metadata_dict[self.k_cache.prefix]

        # ---------------------------------------------------------
        # 1. EXECUTE COMPRESSOR & SCATTER TO KV CACHE
        # ---------------------------------------------------------
        # Project, norm, rope, and store states using the indexer's compressor.
        wkv_wgate = torch.cat([self.kv_proj.weight, self.gate_proj.weight],
                              dim=0)

        # Compute request index mapping for each token in the batch.
        token_to_req_indices = (
            torch.bucketize(
                torch.arange(actual_num_tokens, device=hidden_states.device),
                attn_metadata.cu_seq_lens,  # pytype: disable=attribute-error
                right=True,
            ) - 1).to(torch.int32)

        current_slot_mapping = (
            slot_mapping if slot_mapping is not None else
            attn_metadata.slot_mapping.flatten()  # pytype: disable=attribute-error
        ).to(torch.int32)
        # Key slots in compressed KV space (downsampled by compress_ratio).
        kv_slot_mapping = (current_slot_mapping[:actual_num_tokens] //
                           self.compress_ratio).to(torch.int32)

        # Pack the state cache and write back to self.k_cache.kv_cache.
        updated_cache_jax = compressor_forward_indexer(
            hidden_states=jax_view(hidden_states),
            wkv_wgate=jax_view(wkv_wgate),
            ape=jax_view(self.position_bias),
            norm_weight=jax_view(self.kv_norm.weight),
            cos_sin_cache=jax_view(self.rotary_emb.cos_sin_cache),
            positions=jax_view(positions),
            slot_mapping=jax_view(current_slot_mapping[:actual_num_tokens]),
            block_table=jax_view(attn_metadata.block_table),  # pytype: disable=attribute-error
            token_to_req_indices=jax_view(token_to_req_indices),
            kv_slot_mapping=jax_view(kv_slot_mapping),
            cache=jax_view(self.k_cache.kv_cache),
            state_block_size=self.k_cache.block_size,
            head_dim=self.head_dim,
            rope_head_dim=getattr(self.rotary_emb, "rotary_dim", 64),
            compress_ratio=self.compress_ratio,
            overlap=True,  # Overlap is True for CSA path.
            rms_eps=self.kv_norm.eps,
            quant_block=128,  # Indexer path FP8 block size is 128.
        )

        # Write back in-place to the cache tensor before read-after-write logic.
        self.k_cache.kv_cache.copy_(torch_view(updated_cache_jax))

        # ---------------------------------------------------------
        # 2. EXTRACT KV CACHE FOR JAX KERNEL
        # ---------------------------------------------------------
        kv_cache_tensor = self.k_cache.kv_cache
        # Scale factors are packed straight into kv_cache.

        # ---------------------------------------------------------
        # 3. DIRECT JAX KERNEL CALL
        # ---------------------------------------------------------
        bt_slice = attn_metadata.block_table  # pytype: disable=attribute-error
        sl_slice = attn_metadata.seq_lens  # pytype: disable=attribute-error
        csl_slice = attn_metadata.cu_seq_lens  # pytype: disable=attribute-error

        topk_indices = streamindex_topk(
            query_projection=q_quant[:actual_num_tokens],
            kv_cache=kv_cache_tensor,
            page_indices=bt_slice,
            seq_lens=sl_slice,
            cu_q_lens=csl_slice,
            indexer_weights=weights,
            k=self.topk_tokens,
            compression_ratio=self.compress_ratio,
            # TODO(hwanginho): Tune these block configurations later for performance
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )

        return topk_indices
