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
from typing import Tuple, cast

import jax
import jax.numpy as jnp
import torch
import torchax
import vllm.envs as envs
from jax.sharding import Mesh
from torch.nn import Parameter
from torchax.interop import jax_view, torch_view
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant, get_attention_context)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionType, MLAAttentionImpl)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import mla_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


class PallasMLAttentionBackendImpl(AttentionImpl):

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
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

    def forward(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata, **kwargs):
        # Get the KV cache
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
            self.layer_name]
        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]
        mesh = vllm_model_wrapper_context.mesh

        # Get attention metadata
        attn_metadata, _, _, _ = get_attention_context(self.layer_name)

        q = jax_view(q)
        kv_c_normed = jax_view(kv_c_normed)
        k_pe = jax_view(k_pe)

        # Prepare inputs
        q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=2)

        # (B, N, P) x (N, P, L) -> (B, N, L)
        # torch nn param
        q_nope = jnp.einsum("bnp,npl->bnl", q_nope, jax_view(self.W_UK_T))

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            q_scale = self._q_scale_float
            k_scale = self._k_scale_float
            v_scale = self._v_scale_float

            kv_c_normed, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                         kv_c_normed,
                                         value=None,
                                         k_scale=k_scale)
            k_pe, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                  k_pe,
                                  value=None,
                                  k_scale=k_scale)

        new_kv_cache, outputs = _jax_mla_func(
            kv_cache,
            q_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            attn_metadata,
            mesh,
            self.scale,
            self.qk_nope_head_dim,
            self.num_heads,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # Update KV cache
        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        outputs = outputs.reshape(-1, self.num_heads, self.kv_lora_rank)
        outputs = jnp.einsum("bnl,nlv->bnv", outputs, jax_view(self.W_UV))
        outputs = outputs.reshape(-1, self.num_heads * self.v_head_dim)

        # Return output as Torch tensor
        return torch_view(outputs)


@register_backend(AttentionBackendEnum.FLASHMLA)
class PallasMLAttentionBackend(AttentionBackend):

    @property
    def accept_output_buffer(self) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA"

    @staticmethod
    def get_impl_cls() -> type["PallasMLAttentionBackend"]:
        return PallasMLAttentionBackendImpl


class TPUMLAAttention(MLAAttention):

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        torch.nn.Module.__init__(self)

        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        self.indexer = indexer

        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False
        self.quant_config = quant_config

        # Initialize KV cache quantization attributes
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)
        self.calculate_kv_scales = calculate_kv_scales
        _init_kv_cache_quant(self, quant_config, prefix)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config(
            ).parallel_config.pipeline_parallel_size)
        ]

        self.is_aiter_triton_fp4_bmm_enabled = False
        self.is_aiter_triton_fp8_bmm_enabled = False

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        impl_cls = cast(type[MLAAttentionImpl],
                        self.attn_backend.get_impl_cls())
        self.impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=self.attn_type,
            kv_sharing_target_layer_name=None,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
            **extra_impl_args,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        with torchax.default_env():
            super().process_weights_after_loading(act_dtype)

            # NOTE: vLLM dequantizes kv_b_proj weights which causes more memory
            # usage than expected.
            self.W_UK_T = Parameter(self.W_UK_T, requires_grad=False)
            self.W_UV = Parameter(self.W_UV, requires_grad=False)

            # Delete kv_b_proj_params as the dequantized weights are now stored
            # in self.W_UK_T and self.W_UV.
            kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
            for key in kv_b_proj_params.keys():
                delattr(self.kv_b_proj, key)

    def forward(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe,
                                                self.layer_name)

        self.impl.forward(q, kv_c_normed, k_pe, self.kv_cache, self.layer_name,
                          **kwargs)


class TPUMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        torch.nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = TPUMLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix


def _jax_mla_func(
    kv_cache: jax.Array,
    q: jax.Array,
    q_rope: jax.Array,
    k: jax.Array,
    k_rope: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    scale: float,
    qk_nope_head_dim: int,
    num_heads: int,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> Tuple[jax.Array, jax.Array]:

    k_rope = k_rope.squeeze(1)

    new_kv_cache, outputs = mla_attention(
        q,
        q_rope,
        k,
        k_rope,
        kv_cache,
        attention_metadata,
        mesh,
        num_heads,
        qk_nope_head_dim,
        query_tnh_sharding=None,
        keyvalue_skh_sharding=None,
        attn_o_tnh_sharding=None,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=scale,
    )

    return new_kv_cache, outputs
