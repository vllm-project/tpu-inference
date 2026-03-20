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
import jax.numpy as jnp
import torch
import torchax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn import Parameter
from torchax.interop import jax_view, torch_view
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.attention import \
    get_attention_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import AttentionType

from tpu_inference import utils
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


class VllmMLAAttention(MLAAttention):

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
        super().__init__(num_heads, scale, qk_nope_head_dim, qk_rope_head_dim,
                         v_head_dim, q_lora_rank, kv_lora_rank, kv_b_proj,
                         cache_config, quant_config, prefix, use_sparse,
                         indexer, **extra_impl_args)

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.to_jax_dtype(
                self.kv_cache_dtype)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if not hasattr(self.kv_b_proj, "weight"):
            # Already processed (called multiple times by vLLM and TPU inference loops)
            return

        with torchax.default_env():
            # Bypass vLLM's process_weights_after_loading which crashes on quantized TPU weights
            # Compute W_UK_T and W_UV manually in JAX
            import jax.numpy as jnp
            from torchax.interop import jax_view, torch_view

            weights = jax_view(self.kv_b_proj.weight).astype(jnp.bfloat16)
            if hasattr(self.kv_b_proj, "weight_scale"):
                weight_scale = jnp.squeeze(jax_view(self.kv_b_proj.weight_scale))

                # Handle block quantization broadcasting dynamically
                if weight_scale.shape[0] != weights.shape[0] and weight_scale.shape[1] == weights.shape[0]:
                    weight_scale = weight_scale.T

                if weight_scale.shape[1] < weights.shape[1]:
                    block_size = weights.shape[1] // weight_scale.shape[1]
                    weight_scale = jnp.repeat(weight_scale, block_size, axis=1)

                weights = weights * weight_scale

            # Transpose to match vLLM's .T
            weights = weights.T
            weights = weights.reshape((self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim))

            W_UK, W_UV = jnp.split(weights, [self.qk_nope_head_dim], axis=-1)

            # vLLM: self.W_UV = W_UV.transpose(0, 1)
            self.W_UV = jnp.transpose(W_UV, (1, 0, 2))

            # vLLM: self.W_UK_T = W_UK.permute(1, 2, 0)
            self.W_UK_T = jnp.transpose(W_UK, (1, 2, 0))

            import numpy as np
            self.W_UV = np.array(jnp.transpose(W_UV, (1, 0, 2)))
            self.W_UK_T = np.array(jnp.transpose(W_UK, (1, 2, 0)))

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

        # Get the KV cache
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
            self.layer_name]
        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]

        # Get the mesh
        mesh = vllm_model_wrapper_context.mesh

        # Get the attention metadata
        attn_metadata, _, _, _ = get_attention_context(self.layer_name)

        # Run the fundamental MLA forward pass from the impl
        outputs, new_kv_cache = self.impl.forward(q, kv_c_normed, k_pe,
                                                  kv_cache, attn_metadata,
                                                  mesh, self, **kwargs)

        # Update KV cache
        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        return torch_view(outputs)


@MultiHeadLatentAttentionWrapper.register_oot
class VllmMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

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

        self.mla_attn = VllmMLAAttention(
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
