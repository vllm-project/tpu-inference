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
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention, get_and_maybe_dequant_weights)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import AttentionType

from tpu_inference import utils
from tpu_inference.utils import t2j
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

        # Prevent vLLM's default quant_method.process_weights_after_loading
        # from running on kv_b_proj.  Upstream iterates modules parent-first,
        # so kv_b_proj's PWAL would run BEFORE ours and convert its weight to
        # a multi-host jax.Array.  Downstream .T/.view/.split/.permute then
        # produce a non-fully-addressable jax.Array with LOCAL shape, which
        # general_device_put's single-host branch rejects against the
        # multi-host sharding.  We absorb kv_b_proj's weight into
        # W_UK_T/W_UV ourselves and delete the kv_b_proj params at the end,
        # so skipping its PWAL is safe.  quant_method is a per-layer
        # instance (VllmFp8Config.get_quant_method constructs a fresh
        # VllmFp8LinearMethod), so patching the instance only affects this
        # kv_b_proj.
        import types as _types
        _qm = getattr(self.kv_b_proj, "quant_method", None)
        if _qm is not None and hasattr(_qm, "process_weights_after_loading"):
            def _mla_kv_b_proj_pwal_noop(self_qm, layer):
                return None
            _qm.process_weights_after_loading = _types.MethodType(
                _mla_kv_b_proj_pwal_noop, _qm)

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.to_jax_dtype(
                self.kv_cache_dtype)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        with torchax.default_env():
            # Inlined upstream vllm/model_executor/layers/attention/
            #   mla_attention.py:743-830 MLAAttention.process_weights_after_loading
            # with two relaxations:
            #
            # 1) num_heads_local inferred from actual kv_b_proj shape instead
            #    of asserting equal to self.num_heads — TP-selective pre-slices
            #    kv_b_proj axis-0 by jax.process_count.
            #
            # 2) Temporarily flip quant_method.use_marlin to False around the
            #    dequant call. tpu-inference force-sets use_marlin=True in
            #    VllmFp8LinearMethod.__init__ to bypass vLLM's upstream
            #    scaled_quantize path during inference. But that same flag
            #    causes get_and_maybe_dequant_weights to fall through the
            #    "simple fp8" branch and invoke the generic identity-matmul
            #    fallback, which calls quant_method.apply(layer, torch.eye(...))
            #    — torch.eye is NOT a torchax Tensor → jax_view assertion fails.
            #    Flipping use_marlin=False enables the simple-fp8 branch which
            #    just runs scaled_dequantize(weight, weight_scale_inv,
            #    group_shape=block_size) in pure torch → works on CPU fp8
            #    tensors.  After absorb, kv_b_proj's use_marlin is restored
            #    (irrelevant anyway — we delete its params below).
            _qm = self.kv_b_proj.quant_method
            _saved_use_marlin = getattr(_qm, "use_marlin", False)
            _qm.use_marlin = False
            try:
                kv_b_proj_weight = get_and_maybe_dequant_weights(
                    self.kv_b_proj, out_dtype=act_dtype).T
            finally:
                _qm.use_marlin = _saved_use_marlin

            per_head_dim = self.qk_nope_head_dim + self.v_head_dim
            assert kv_b_proj_weight.dim() == 2, (
                f"expected kv_b_proj_weight to be 2D after .T, got "
                f"shape={tuple(kv_b_proj_weight.shape)}")
            assert kv_b_proj_weight.shape[0] == self.kv_lora_rank, (
                f"kv_b_proj_weight axis-0 {kv_b_proj_weight.shape[0]} != "
                f"kv_lora_rank {self.kv_lora_rank}")
            total_out = kv_b_proj_weight.shape[1]
            assert total_out % per_head_dim == 0, (
                f"kv_b_proj_weight axis-1 {total_out} not divisible by "
                f"per_head_dim {per_head_dim} "
                f"(qk_nope={self.qk_nope_head_dim}, v_head={self.v_head_dim})")
            num_heads_local = total_out // per_head_dim
            assert num_heads_local > 0 and self.num_heads % num_heads_local == 0, (
                f"num_heads_local {num_heads_local} must evenly divide "
                f"num_heads {self.num_heads} (TP-sliced kv_b_proj)")

            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank, num_heads_local, per_head_dim)
            W_UK, W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # Match upstream transposes (mla_attention.py:828-830):
            #   W_UV: (L, N, V)  → (N, L, V)
            #   W_UK: (L, N, P)  → (N, P, L)
            self.W_UV = W_UV.transpose(0, 1)
            self.W_UK_T = W_UK.permute(1, 2, 0)

            # quantize W_UK_T, W_UV back to cache type and transfer
            # `W_UK_T`, `W_UV` to TPUs
            mesh = self.kv_b_proj.quant_method.linear_config.mesh
            sharding = NamedSharding(mesh, P(ShardingAxisName.ATTN_HEAD, ))
            scale_sharding = NamedSharding(mesh, P(None, None))

            # TP-selective loader pre-slices kv_b_proj axis 0 by jax.process_count,
            # so W_UK_T/W_UV carry local num_heads here. When pre-sliced, supply
            # explicit global shapes so general_device_put uses
            # make_array_from_process_local_data (the cross-host sharding check
            # divides the GLOBAL dim, not the per-process local one).
            _is_tp_presliced = num_heads_local < self.num_heads
            if _is_tp_presliced:
                w_uk_t_global_shape = (self.num_heads, self.qk_nope_head_dim,
                                       self.kv_lora_rank)
                w_uv_global_shape = (self.num_heads, self.kv_lora_rank,
                                     self.v_head_dim)
            else:
                w_uk_t_global_shape = None
                w_uv_global_shape = None

            if self.kv_cache_quantized_dtype is not None:
                if _is_tp_presliced:
                    # quantize_tensor(axis=1) scale shape interaction with
                    # make_array_from_process_local_data is not verified.
                    raise NotImplementedError(
                        "TP-selective loader + KV cache quantization combo "
                        "not supported yet: need verified global_shape for "
                        "W_UK_T_scale / W_UV_scale. "
                        "Set kv_cache_dtype=auto for TP-selective runs.")
                # KV cache is quantized: quantize W_UK_T/W_UV to cache dtype.
                # Use t2j instead of jax_view because scaled_dequantize outputs
                # pure torch.Tensor (not torchax Tensor/View) — jax_view would
                # raise AssertionError.  See equivalent fix in auto branch below.
                self.W_UK_T, self.W_UK_T_scale = quantize_tensor(
                    self.kv_cache_quantized_dtype,
                    t2j(self.W_UK_T, use_dlpack=False),
                    axis=1)
                self.W_UK_T = torch_view(general_device_put(self.W_UK_T, sharding))
                self.W_UK_T_scale = torch_view(
                    general_device_put(jnp.expand_dims(self.W_UK_T_scale, 0),
                                       scale_sharding))

                self.W_UV, self.W_UV_scale = quantize_tensor(
                    self.kv_cache_quantized_dtype,
                    t2j(self.W_UV, use_dlpack=False),
                    axis=1)
                self.W_UV = torch_view(general_device_put(self.W_UV, sharding))
                self.W_UV_scale = torch_view(
                    general_device_put(jnp.expand_dims(self.W_UV_scale, 0),
                                       scale_sharding))
            else:
                # No KV quantization (kv_cache_dtype=auto): move to TPU as-is
                # with scale=1.0 (identity: no dequantization needed).
                #
                # scaled_dequantize path returns a pure torch.Tensor (not a
                # torchax Tensor), so jax_view would raise AssertionError.
                # Use t2j to convert torch CPU Tensor → single-process
                # jax.Array; then general_device_put (multi-host Ray branch)
                # uses make_array_from_process_local_data with global_shape
                # to reassemble the full multi-host array.
                w_uk_t_jax = t2j(self.W_UK_T, use_dlpack=False)
                w_uv_jax = t2j(self.W_UV, use_dlpack=False)
                self.W_UK_T = torch_view(general_device_put(
                    w_uk_t_jax, sharding, global_shape=w_uk_t_global_shape))
                self.W_UV = torch_view(general_device_put(
                    w_uv_jax, sharding, global_shape=w_uv_global_shape))
                # Scalar scale = 1.0, fully replicated
                scalar_sharding = NamedSharding(mesh, P())
                one = jnp.array(1.0, dtype=jnp.float32)
                self.W_UK_T_scale = torch_view(
                    general_device_put(one, scalar_sharding))
                self.W_UV_scale = torch_view(
                    general_device_put(one, scalar_sharding))

            self.W_UK_T = Parameter(self.W_UK_T, requires_grad=False)
            self.W_UK_T_scale = Parameter(self.W_UK_T_scale,
                                          requires_grad=False)
            self.W_UV = Parameter(self.W_UV, requires_grad=False)
            self.W_UV_scale = Parameter(self.W_UV_scale, requires_grad=False)

            # Delete kv_b_proj_params as the dequantized weights are now stored
            # in self.W_UK_T and self.W_UV.
            kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
            for key in kv_b_proj_params.keys():
                delattr(self.kv_b_proj, key)

    def forward(self,
                q: torch.Tensor,
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
