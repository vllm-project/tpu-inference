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

import torch
import torch.nn as nn
from torchax.interop import jax_view
from torchax.interop import torch_view
from tpu_inference.layers.vllm.custom_ops.streamindex_topk import streamindex_topk

# =====================================================================
# IMPORT TPU CUSTOM OPS TO TRIGGER vLLM @register_oot DECORATORS
# =====================================================================
import tpu_inference.layers.vllm.custom_ops.rope
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Indexer


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

  q_rope, _ = rotary_emb(positions, q.clone())

  # Find absolute max along the head dimension
  amax = torch.amax(torch.abs(q_rope), dim=-1, keepdim=True)

  # Clamp to avoid division by zero
  amax = torch.clamp(amax, min=1e-8)

  # Calculate scale for symmetric int8 range [-127, 127]
  q_scales = amax / 127.0

  # Quantize, round, and cast
  q_quant = torch.round(q_rope / q_scales)
  q_quant = torch.clamp(q_quant, -127, 127).to(torch.int8)

  return q_quant, q_scales.squeeze(-1)


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
  """TPU-compatible DeepSeek-V4 Lightning Indexer with StreamIndex.

  This class overrides the forward method of DeepseekV4Indexer to provide a
  TPU-compatible implementation using JAX interop. It uses
  `streamindex_topk` to compute top-k token indices over a
  PagedAttention KV cache with dynamic sequence padding and bucketization to
  avoid excessive JAX recompilations.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # TODO(hwanginho): Attach TPU-native compressor here.
    # Instantiate the compressor
    # self.compressor = TPUDeepseekCompressor(
    #     config=self.config,
    #     # Pass any necessary cache config or projection dimensions
    # )
    self.compressor = getattr(self, "compressor", None)  # Placeholder

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

    if actual_num_tokens == 0:
      return torch.empty(
          (0, self.topk_tokens), dtype=torch.int32, device=query.device
      )

    q, _ = self.wq_b(query)
    q = q.view(-1, self.n_head, self.head_dim)

    q_quant, q_scales = fused_indexer_q_rope_quant(q, positions, rotary_emb)

    # Do not fold q_scales into weights. Keep weights static to save HBM
    weights = (
        indexer_weights.to(q.dtype) * self.softmax_scale * (self.head_dim**-0.5)
    )

    attn_metadata_dict = get_forward_context().attn_metadata
    attn_metadata = attn_metadata_dict[self.k_cache.prefix]

    # ---------------------------------------------------------
    # 1. EXECUTE COMPRESSOR & SCATTER TO KV CACHE
    # ---------------------------------------------------------
    # TODO(hwanginho): Execute the TPU-Native Compressor to compute the current
    # token's keys, and write a native TPU scatter kernel to save those keys
    # into `self.k_cache.kv_cache`.
    # (Note: MUST happen before jax_view to guarantee XLA read-after-write ordering!)

    if self.compressor is not None:
      # Use provided slot_mapping, fallback to metadata if None
      if slot_mapping is None:
        current_slot_mapping = attn_metadata.slot_mapping.flatten()
      else:
        current_slot_mapping = slot_mapping

    #   active_slot_mapping = current_slot_mapping[:actual_num_tokens]

    #   self.compressor(
    #       hidden_states=hidden_states[:actual_num_tokens],
    #       positions=positions[:actual_num_tokens],
    #       rotary_emb=rotary_emb,
    #       kv_cache=self.k_cache.kv_cache,
    #       slot_mapping=active_slot_mapping,
    #   )

    # ---------------------------------------------------------
    # 2. EXTRACT KV CACHE FOR JAX KERNEL
    # ---------------------------------------------------------
    kv_cache_tensor = self.k_cache.kv_cache
    assert (
        kv_cache_tensor.is_contiguous()
    ), "KV cache must be physically contiguous for TorchAX."
    kv_cache_jax = jax_view(kv_cache_tensor)

    # TODO(hwanginho): Support DSv4 FP8 index cache format.
    # Once vLLM's FP8 cache is enabled, extract the scale tensor here if it
    # is a separate tensor, or pack it with cache_kv. Need to align with
    # alynie@ on https://github.com/vllm-project/tpu-inference/pull/2858
    # (The exact attribute name depends on vLLM's FP8 implementation, e.g., `kv_scale`)
    # kv_scales_tensor = self.k_cache.kv_scale
    # assert kv_scales_tensor.is_contiguous(), "Scales must be contiguous"
    # kv_scales_jax = jax_view(kv_scales_tensor)

    c_s = getattr(self, "c_s", 8192)
    c_t = getattr(self, "c_t", 2048)

    def run_jax_kernel(
        q_slice, static_weights, bt_slice, sl_slice, csl_slice, bucket_tokens
    ):
      n_tokens = q_slice.shape[0]
      n_batch, current_blocks = bt_slice.shape

      static_tokens = (
          (n_tokens + bucket_tokens - 1) // bucket_tokens
      ) * bucket_tokens
      pad_tokens = static_tokens - n_tokens

      bucket_batch = 16
      static_batch = (
          (n_batch + bucket_batch - 1) // bucket_batch
      ) * bucket_batch
      pad_batch = static_batch - n_batch

      bucket_blocks = (
          max(16, 1 << (current_blocks - 1).bit_length())
          if current_blocks > 0
          else 16
      )
      pad_blocks = bucket_blocks - current_blocks

      if pad_tokens > 0:
        q_slice = torch.nn.functional.pad(q_slice, (0, 0, 0, 0, 0, pad_tokens))

      if pad_batch > 0 or pad_blocks > 0:
        bt_slice = torch.nn.functional.pad(
            bt_slice, (0, pad_blocks, 0, pad_batch)
        )

      if pad_batch > 0:
        sl_slice = torch.nn.functional.pad(sl_slice, (0, pad_batch))
        csl_slice = torch.nn.functional.pad(csl_slice, (0, pad_batch))

        # Ensures valid offset pointing to the end of the padded batch,
        # preventing out-of-bounds JAX errors
        csl_slice[-pad_batch:] = csl_slice[n_batch]

      bt_slice = bt_slice.to(torch.int32)
      sl_slice = sl_slice.to(torch.int32)
      csl_slice = csl_slice.to(torch.int32)

      q_jax = jax_view(q_slice.contiguous())
      w_jax = jax_view(static_weights.contiguous())
      bt_jax = jax_view(bt_slice.contiguous())
      sl_jax = jax_view(sl_slice.contiguous())
      csl_jax = jax_view(csl_slice.contiguous())

      topk_jax = streamindex_topk(
          query_projection=q_jax,
          kv_cache=kv_cache_jax,
          block_table=bt_jax,
          seq_lens=sl_jax,
          cu_q_lens=csl_jax,
          indexer_weights=w_jax,
          k=self.topk_tokens,
          compression_ratio=self.compress_ratio,
          c_s=c_s,
          c_t=c_t,
      )
      return torch_view(topk_jax)[:n_tokens, :]

    # ---------------------------------------------------------
    # 3. PREPARE METADATA & BUCKETS FOR JAX
    # ---------------------------------------------------------
    q_lens = []
    seq_lens_list = []
    block_tables_list = []

    if attn_metadata.num_decodes > 0:
      decode_meta = attn_metadata.decode
      decode_lens = getattr(
          decode_meta, "decode_lens", torch.ones_like(decode_meta.seq_lens)
      )
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

      # Use torch.zeros instead of python literal [0] to avoid XLA graph breaks
      combined_csl = torch.cat([
          torch.zeros((1,), dtype=torch.int32, device=q.device),
          torch.cumsum(all_q_lens, dim=0, dtype=torch.int32),
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
      combined_sl = torch.tensor(
          [actual_num_tokens], dtype=torch.int32, device=q.device
      )
      combined_csl = torch.tensor(
          [0, actual_num_tokens], dtype=torch.int32, device=q.device
      )

    bucket_tokens = 256 if attn_metadata.num_prefills > 0 else 16

    # ---------------------------------------------------------
    # 4. EXECUTE JAX KERNEL
    # ---------------------------------------------------------
    topk_indices = run_jax_kernel(
        q_slice=q_quant[:actual_num_tokens],
        static_weights=weights,
        bt_slice=combined_bt,
        sl_slice=combined_sl,
        csl_slice=combined_csl,
        bucket_tokens=bucket_tokens,
    )

    return topk_indices
