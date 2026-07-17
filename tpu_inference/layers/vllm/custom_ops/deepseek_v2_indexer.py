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
"""TPU-compatible DeepSeek-V2/GLM custom indexer."""

import torch
import jax.numpy as jnp
from torchax.interop import jax_view, torch_view
from vllm.model_executor.models.deepseek_v2 import Indexer, DeepseekV32IndexerCache
from tpu_inference.layers.vllm.custom_ops.sparse_attn_indexer import VllmSparseAttnIndexer
from tpu_inference.layers.common.quantization import quantize_tensor

class VllmDeepseekV32IndexerCache(DeepseekV32IndexerCache):
    """TPU-compatible DeepSeek-V2 custom indexer cache."""
    pass

class VllmIndexer(Indexer):
    """TPU-compatible DeepSeek-V2/GLM Indexer with StreamIndex."""

    def __init__(self, *args, **kwargs):
        # Rebind to instantiate our TPU IndexerCache subclass
        import vllm.model_executor.models.deepseek_v2 as ds2_models
        orig_cache = ds2_models.DeepseekV32IndexerCache
        ds2_models.DeepseekV32IndexerCache = VllmDeepseekV32IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            ds2_models.DeepseekV32IndexerCache = orig_cache

        # Bind the JAX-native SparseAttnIndexer custom op
        self.indexer_op = VllmSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: torch.nn.Module,
    ) -> torch.Tensor:
        # 1. Project input queries
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        # 2. Project keys and routing weights
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        # Apply RoPE
        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # 3. Quantize q using JAX-native quantization
        q = q.view(-1, self.head_dim)
        q_jax = jax_view(q)
        q_quant_jax, q_scales_jax = quantize_tensor(
            jnp.float8_e4m3fn, q_jax, axis=-1
        )
        q_fp8 = torch_view(q_quant_jax)
        q_scale = torch_view(q_scales_jax).squeeze(-1)

        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head)

        # Fold scales into the weights
        weights = weights * q_scale * self.softmax_scale * self.n_head_scale

        # 4. Invoke the JAX custom-op indexer
        return self.indexer_op(hidden_states, q_fp8, k, weights)
