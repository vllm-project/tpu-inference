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
import vllm.model_executor.layers.attention.mla_attention as vllm_mla_attn
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
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType

from tpu_inference import utils
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


# Provides a no-op implementation for upstream MLA prefill backend.
# This is used since upstream vllm has moved prefill backend ownership
# to the MLAAttention __init__ method:
# https://github.com/vllm-project/vllm/pull/41744.
class DummyMLAPrefillBackend:

    def __init__(self, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


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
        vllm_mla_attn.get_mla_prefill_backend = lambda config: DummyMLAPrefillBackend
        super().__init__(
            num_heads,
            scale,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            use_sparse=use_sparse,
            indexer=indexer,
            **extra_impl_args,
        )

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        # Only genuinely *quantized* kv_cache_dtype values (fp8*/nvfp4/
        # per_token_head -- see vllm's `is_quantized_kv_cache`) should route
        # through this backend's fp8-style dequant-on-gather / static
        # per-tensor quantize-before-cache-write code paths
        # (`flash_attn_mla_sparse.py`'s `_quantize_dequantize_round_trip`/
        # `quantize_kv` calls, gated on `layer.kv_cache_quantized_dtype`).
        # The previous `!= "auto"` check incorrectly also matched plain
        # float dtypes (`bfloat16`, `float16`) whenever an *explicit*
        # `--kv-cache-dtype` was passed (as opposed to `auto`), causing
        # those quantize/dequantize helpers -- which assume a genuinely
        # narrow target dtype's representable range -- to fire for a
        # bfloat16 cache. Confirmed via a live crash-free-but-NaN run on
        # the TPU VM (`--kv-cache-dtype=bfloat16`): `GLM5_DSA_DEBUG=1`
        # showed `q_combined_mean_abs=nan` from the very first sparse
        # attention call, tracing back to `_quantize_dequantize_round_trip`
        # being invoked with `dtype=jnp.bfloat16` and this checkpoint's
        # fp8-calibration-heuristic `k_scale`/`v_scale` defaults
        # (`GLM5_DSA_KV_SCALE_DEFAULT`, 0.001) -- nonsensical for a
        # non-quantized target dtype, producing NaNs that then propagated
        # through every downstream layer.
        self.kv_cache_quantized_dtype = None
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.kv_cache_quantized_dtype = utils.to_jax_dtype(
                self.kv_cache_dtype)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        with torchax.default_env():
            # tpu-inference's linear methods store layer.weight in [in_features,
            # out_features] layout. Upstream MLA's process_weights_after_loading reads
            # kv_b_proj.weight assuming [out_features, in_features] vLLM layout.
            from vllm.model_executor.layers.linear import \
                UnquantizedLinearMethod
            if isinstance(self.kv_b_proj.quant_method,
                          UnquantizedLinearMethod):
                self.kv_b_proj.weight = Parameter(
                    self.kv_b_proj.weight.transpose(0, 1).contiguous(),
                    requires_grad=False,
                )
            super().process_weights_after_loading(act_dtype)

            # NOTE: vLLM dequantizes kv_b_proj weights which causes more memory
            # usage than expected.
            # quantize W_UK_T, W_UV back to cache type and transfer
            # `W_UK_T`, `W_UV` to TPUs
            # Try to get the mesh from the first possible path.
            quant_method = getattr(self.kv_b_proj, 'quant_method', None)
            linear_config = getattr(quant_method, 'linear_config', None)
            mesh = getattr(linear_config, 'mesh', None)
            if mesh is None:
                # If the first path failed, mesh will be None. Try the second path.
                scheme = getattr(self.kv_b_proj, 'scheme', None)
                linear_config = getattr(scheme, 'linear_config', None)
                mesh = getattr(linear_config, 'mesh', None)

            if mesh is None:
                # If mesh is still None after trying all paths, raise an error.
                raise ValueError(
                    "Could not find JAX Mesh. Failed to access "
                    "'.quant_method.linear_config.mesh' or "
                    "'.scheme.linear_config.mesh' on the kv_b_proj layer.")

            sharding = NamedSharding(mesh, P(ShardingAxisName.ATTN_HEAD, ))
            # Upstream MLA registers W_UK_T/W_UV as nn.Parameters, so the
            # intermediate JAX values cannot be assigned to the attributes
            # directly; stage them in locals and assign Parameters at the end.
            #
            # Guard on `self.kv_cache_quantized_dtype is not None` -- this
            # used to call `quantize_tensor(self.kv_cache_quantized_dtype,
            # ...)` unconditionally, including when `kv_cache_quantized_dtype`
            # is `None` (an unquantized -- e.g. bf16 -- KV cache). Confirmed
            # via a live TPU VM run that this is a *second* real, distinct
            # bug of the same shape as the `__init__` one fixed just above:
            # `quantize_tensor(None, ...)` doesn't raise (`jnp.finfo(None)`
            # silently returns *float64*'s info instead of erroring, and
            # `Array.astype(None)` is a silent no-op), but computes a scale
            # of `abs_max / float64_max` -- underflows to 0 in
            # float32/bfloat16 -- whose reciprocal (`scale_inv`) becomes
            # `inf`, producing NaN/Inf directly baked into `W_UK_T`/`W_UV`
            # at weight-loading time (confirmed live: `GLM5_DSA_DEBUG=1`
            # showed `q_combined_mean_abs=nan` from the very first sparse
            # attention call even *after* fixing the `__init__` bug above,
            # traced to this second call site). When unquantized, just
            # device-put the tensors as-is with a scale of 1.0 (a no-op),
            # matching every other "unquantized" fallback convention in this
            # file (e.g. `flash_attn_mla_sparse.py`'s `k_scale_static`).
            if self.kv_cache_quantized_dtype is not None:
                w_uk_t, w_uk_t_scale = quantize_tensor(
                    self.kv_cache_quantized_dtype,
                    jax_view(self.W_UK_T),
                    axis=1)
                w_uv, w_uv_scale = quantize_tensor(
                    self.kv_cache_quantized_dtype, jax_view(self.W_UV), axis=1)
            else:
                # Fallback scale shapes must match `quantize_tensor(...,
                # axis=1)`'s own output shape convention (per-tensor max
                # reduced over axis=1, keepdims, then squeezed on that same
                # axis) so the later `jnp.expand_dims(..., 1)` /
                # `jnp.expand_dims(..., 0)` + broadcast-multiply calls below
                # behave identically to the quantized case (just a no-op
                # scale of 1.0 everywhere). Computed generically (drop axis
                # 1 from the tensor's own shape, whatever its rank) rather
                # than hardcoding the real 3-D `[H, nope/kv_lora_rank,
                # kv_lora_rank/v_head_dim]` convention, so this also works
                # against simplified/mocked lower-rank tensors (e.g. this
                # file's own unit tests).
                w_uk_t_j = jax_view(self.W_UK_T)  # [H, nope, kv_lora_rank]
                w_uk_t = w_uk_t_j
                w_uk_t_scale = jnp.ones(w_uk_t_j.shape[:1] +
                                        w_uk_t_j.shape[2:],
                                        dtype=jnp.float32)  # [H, kv_lora_rank]
                w_uv_j = jax_view(self.W_UV)  # [H, kv_lora_rank, v_head_dim]
                w_uv = w_uv_j
                w_uv_scale = jnp.ones(w_uv_j.shape[:1] + w_uv_j.shape[2:],
                                      dtype=jnp.float32)  # [H, v_head_dim]

            w_uk_t = torch_view(general_device_put(w_uk_t, sharding))
            w_uk_t_scale = torch_view(
                general_device_put(jnp.expand_dims(w_uk_t_scale, 1), sharding))

            w_uv = torch_view(general_device_put(w_uv, sharding))
            w_uv_scale = torch_view(
                general_device_put(jnp.expand_dims(w_uv_scale, 0), sharding))

            self.W_UK_T = Parameter(w_uk_t, requires_grad=False)
            self.W_UK_T_scale = Parameter(w_uk_t_scale, requires_grad=False)
            self.W_UV = Parameter(w_uv, requires_grad=False)
            self.W_UV_scale = Parameter(w_uv_scale, requires_grad=False)

            # Delete kv_b_proj_params as the dequantized weights are now stored
            # in self.W_UK_T and self.W_UV.
            kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
            for key in kv_b_proj_params.keys():
                delattr(self.kv_b_proj, key)

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                output: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
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
        outputs, new_kv_cache = self.impl.forward(q,
                                                  kv_c_normed,
                                                  k_pe,
                                                  kv_cache,
                                                  attn_metadata,
                                                  mesh,
                                                  self,
                                                  output=output,
                                                  **kwargs)

        # Update KV cache
        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        if outputs is not output and output is not None:
            output.copy_(outputs)

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
        skip_topk: bool = False,
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
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
        self.g_proj = getattr(mla_modules, "g_proj", None)
        self.skip_topk = skip_topk

        if self.indexer is not None and not self.skip_topk:
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
            # Pre-existing gap, not GLM-5.2-specific: this was never
            # threaded through, so `MLAAttention.__init__`'s
            # `topk_indices_buffer` param (and therefore every sparse MLA
            # layer's attention Impl) silently defaulted to `None` for
            # *every* layer, dense or sparse-indexer-carrying alike --
            # confirmed via a live crash on the TPU VM
            # (`PallasMLASparseAttentionBackendImpl.forward()` raising
            # "requires a topk_indices_buffer ... got None") the first time
            # any model actually exercised `use_sparse=True` through this
            # generic wrapper. Unconditional (not gated on `skip_topk`),
            # matching upstream vLLM's own (unmodified)
            # `MultiHeadLatentAttentionWrapper.__init__`, which always
            # passes `mla_modules.topk_indices_buffer` here regardless of
            # per-layer skip status -- every layer's Impl needs the *same*
            # shared buffer object to read from, even "skip" layers that
            # never write it themselves.
            topk_indices_buffer=mla_modules.topk_indices_buffer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None")
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None")
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None")

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None")
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None")
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q_nope *= llama_4_scaling
            q_pe *= llama_4_scaling

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_heads * self.v_head_dim),
        )

        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()

        return self.o_proj(attn_out)[0]
