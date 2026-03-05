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
from typing import Tuple

import jax.numpy as jnp
import torch
from jax.sharding import Mesh
from torchax.interop import jax_view
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.attention.backend import (AttentionBackend, AttentionLayer,
                                       MLAAttentionImpl)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

from tpu_inference.layers.common.attention_interface import mla_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv


class PallasMLAttentionBackendImpl(MLAAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_head_dim: int | None = None,
        v_head_dim: int | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """
        Needed because this is abstract in the base class but we don't use it (instead, favoring a single `forward`).
        """
        pass

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Needed because this is abstract in the base class but we don't use it (instead, favoring a single `forward`).
        """
        pass

    def forward(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor, kv_cache: jnp.ndarray,
                attn_metadata: AttentionMetadata, mesh: Mesh,
                layer: MLAAttention,
                **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Runs the fundamental MLA forward pass.

        NOTE: the base `MLAAttentionImpl` doesn't actually have this method, but we only need
        a single `forward` for now and this is called by the bespoke MLAAttention class
        below anyways.

        Args:
            q: torch.Tensor
            kv_c_normed: torch.Tensor
            k_pe: torch.Tensor
            kv_cache: jnp.ndarray
            attn_metadata: AttentionMetadata
            mesh: Mesh
            layer: MLAAttention instance

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: (outputs, new_kv_cache)
        """
        q = jax_view(q)
        kv_c_normed = jax_view(kv_c_normed)
        k_pe = jax_view(k_pe)

        # Prepare inputs
        q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=2)

        # (B, N, P) x (N, P, L) -> (B, N, L)
        # torch nn param
        q_nope = jnp.einsum("bnp,npl->bnl", q_nope, jax_view(layer.W_UK_T))

        q_scale = k_scale = v_scale = None
        if layer.kv_cache_quantized_dtype:
            q_scale = layer._q_scale_float
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float

            kv_c_normed, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                         kv_c_normed,
                                         value=None,
                                         k_scale=k_scale)
            k_pe, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                  k_pe,
                                  value=None,
                                  k_scale=k_scale)

        k_pe = k_pe.squeeze(1)
        new_kv_cache, outputs = mla_attention(
            q_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            kv_cache,
            attn_metadata,
            mesh,
            self.num_heads,
            self.qk_nope_head_dim,
            query_tnh_sharding=None,
            keyvalue_skh_sharding=None,
            attn_o_tnh_sharding=None,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=self.scale,
        )

        outputs = outputs.reshape(-1, self.num_heads, self.kv_lora_rank)
        outputs = jnp.einsum("bnl,nlv->bnv", outputs, jax_view(layer.W_UV))
        outputs = outputs.reshape(-1, self.num_heads * self.v_head_dim)

        return outputs, new_kv_cache


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class PallasMLAttentionBackend(AttentionBackend):

    @property
    def accept_output_buffer(self) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_impl_cls() -> type["PallasMLAttentionBackend"]:
        return PallasMLAttentionBackendImpl
