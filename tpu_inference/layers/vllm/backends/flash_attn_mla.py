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
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.attention.backend import (AttentionBackend, AttentionLayer,
                                       MLAAttentionImpl)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

import tpu_inference.envs as envs
from tpu_inference.layers.common.attention_interface import mla_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import (
    quantize_kv, static_per_tensor_quantize_tensor)


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class PallasMLAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_impl_cls() -> type["PallasMLAttentionBackend"]:
        return PallasMLAttentionBackendImpl

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        return 1024


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

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                kv_cache: jnp.ndarray,
                attn_metadata: AttentionMetadata,
                mesh: Mesh,
                layer: MLAAttention,
                output: torch.Tensor | None = None,
                **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Runs the fundamental MLA forward pass.

        NOTE: the base `MLAAttentionImpl` doesn't actually have this method, but we only need
        a single `forward` for now and this is called by the bespoke MLAAttention class
        below anyways.

        Args:
            q: q_nope, q_pe tuple of torch.Tensor
            kv_c_normed: torch.Tensor
            k_pe: torch.Tensor
            kv_cache: jnp.ndarray
            attn_metadata: AttentionMetadata
            mesh: Mesh
            layer: MLAAttention instance
            output: torch.Tensor

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: (new_kv_cache, outputs)
        """

        q_nope, q_pe = q
        q_nope = jax_view(q_nope)
        q_pe = jax_view(q_pe)
        kv_c_normed = jax_view(kv_c_normed)
        k_pe = jax_view(k_pe)
        input_dtype = q_nope.dtype

        # Einsum selects 'n' as the batch axis and emits it as the major-most physical dimension.
        q_nope = jnp.einsum(
            "bnp,npl->nbl",
            q_nope,
            jax_view(layer.W_UK_T),  # torch nn param
            preferred_element_type=jnp.float32)
        scale = jax_view(layer.W_UK_T_scale)  # torch nn param
        q_nope = (q_nope * scale).astype(input_dtype)

        q_scale = k_scale = v_scale = None
        if layer.kv_cache_quantized_dtype:
            q_scale = layer._q_scale_float
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float

            if not envs.DISABLE_MLA_Q_ACTIVATION_QUANTIZATION:
                q_nope = static_per_tensor_quantize_tensor(
                    layer.kv_cache_quantized_dtype, q_nope, q_scale)
                q_pe = static_per_tensor_quantize_tensor(
                    layer.kv_cache_quantized_dtype, q_pe, q_scale)
            else:
                # Needed because q_pe comes in as FP32, so we cast down to BF16
                q_pe = q_pe.astype(input_dtype)

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
            query_nth_sharding=None,
            query_tnh_sharding=None,
            keyvalue_skh_sharding=None,
            attn_o_nth_sharding=None,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=self.scale,
        )

        # einsum selects 'n' as the major-most physical dimension again.
        outputs = outputs.reshape(self.num_heads, -1, self.kv_lora_rank)
        outputs = (jnp.einsum("nbl,nlv->bnv",
                              outputs,
                              jax_view(layer.W_UV),
                              preferred_element_type=jnp.float32) *
                   jax_view(layer.W_UV_scale)).astype(input_dtype)
        outputs = outputs.reshape(-1, self.num_heads * self.v_head_dim)

        out_torch = torch_view(outputs)
        if output is not None:
            output.copy_(out_torch)
        return out_torch, new_kv_cache
