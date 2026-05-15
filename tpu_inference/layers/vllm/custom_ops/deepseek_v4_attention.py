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
"""TPU overrides for DeepSeek-V4 attention:
        SWA KV Cache registry + Attention + AttentionWrapper implementations.

For reference, GPU implementations for V4 attention are spread across three 
registrations in vllm/vllm/model_executor/layers/deepseek_v4_attention.py:
  1. DeepseekV4SWACache — returns SlidingWindowMLASpec for the 
        per-layer sliding-window KV cache.
  2. DeepseekV4MLAAttention — returns MLAAttentionSpec when 
        compress_ratio > 1, else None.
  3. DeepseekV4MultiHeadLatentAttentionWrapper  —  forward pass 
        using torch.ops.vllm.deepseek_v4_attention.

TODOs: 
  - TpuDeepseekV4MLAAttention is currently a stub. 
    Will need to call TpuDeepseekV4MultiHeadLatentAttentionWrapper.forward()
  - TpuDeepseekV4MultiHeadLatentAttentionWrapper is currently a stub.
    Will need to implement forward().
"""

import torch
import torch.nn as nn
import vllm.model_executor.layers.deepseek_compressor as _compressor_module
import vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope as _rope_module
import vllm.model_executor.models.deepseek_v4 as _dsv4_module
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4MLAModules,
    DeepseekV4MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform as _real_platform
from vllm.v1.attention.backend import (AttentionBackend, 
                                       AttentionType)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from tpu_inference.layers.vllm.backends.flash_attn_mla import PallasMLAttentionBackend

class _V4TpuPlatformShim:
    """Monkey patch the current_platform to avoid unsuported device errors raised by
    running DSv4.
    This error is triggered in two observed cases:
      - torch.cuda.Stream() which cannot be wrapped by torchax.
      - 
    Two categories of TPU-incompatible calls occur during DeepseekV4 __init__:

    (A) CUDA-only APIs — cannot be intercepted by torchax or torch.device():
        torch.cuda.Stream() at deepseek_v4.py:1337.
        Fix: report is_rocm()=True → aux_stream_list = None branch taken.

    (B) A torch tensor is attempted to be constructed explictily with device=current_platform.device_type which 
        routes to "tpu".
        
    This shim allows intercepting the curent platform call to replace 'tpu' with 'cpu' in order to avoid 
    platform-related failures when defining the torch code to wrap with torchax.
    """

    def is_rocm(self) -> bool:
        return _real_platform.is_rocm() or _real_platform.is_tpu()

    @property
    def device_type(self) -> str:
        if _real_platform.is_tpu():
            return "cpu"
        return _real_platform.device_type

    def __getattr__(self, name: str):
        return getattr(_real_platform, name)


# Patch the modules that are throwing platform incompatibilities for the 'tpu' backend.
_dsv4_module.current_platform = _V4TpuPlatformShim()
_compressor_module.current_platform = _V4TpuPlatformShim()
_rope_module.current_platform = _V4TpuPlatformShim()

class _TpuMtpBufferNoop:
    """Monkey patch for DeepseekV4Model._mtp_hidden_buffer on TPU.
    This avoids a failure triggered by /vllm/vllm/model_executor/models/deepseek_v4.py:
    self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
    

    Since MTP draft is not implemented on TPU, is a torch tensor on CPU while
    hidden_states is a torchax tensor which throws an incompatibility error.
    """

    def __getitem__(self, idx):
        return self

    def copy_(self, src):
        return self


_orig_dsv4_model_init = _dsv4_module.DeepseekV4Model.__init__


def _tpu_dsv4_model_init(self, *args, **kwargs):
    _orig_dsv4_model_init(self, *args, **kwargs)
    # Replace the CPU buffer with the noop so forward() copy_ doesn't crash.
    if self._mtp_hidden_buffer is not None:
        self._mtp_hidden_buffer = _TpuMtpBufferNoop()


_dsv4_module.DeepseekV4Model.__init__ = _tpu_dsv4_model_init


class TpuDeepseekV4SWACache(MLAAttention):
    """Implements DeepseekV4SWACache for torchax.
    (vllm/vllm/v1/attention/backends/mla/sparse_swa.py).

    Inherits from MLAAttention so that we can register a KV cache for this layer.

    Shapes:
      kv_cache  [num_blocks, block_size=64, 1, head_dim]  uint8
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        prefix: str,
        cache_config: CacheConfig,
        vllm_config: VllmConfig,
    ) -> None:
        # Even though we inherit from MLAAttention, it is only used to allow
        # allocating KV cache blocks in the kv_cache_manager.
        nn.Module.__init__(self)
        self.kv_cache = torch.tensor([])
        self.layer_name = prefix

        # Attributes used by kv_cache_manager.py
        self.head_size = head_dim         # used to compute head_size_set
        self.num_kv_heads = 1             # MLA always uses 1 KV head
        self.attn_type = AttentionType.DECODER
        self.kv_sharing_target_layer_name = None

        # If window_size > 1 then kv_cache_manager needs to create SlidingWindowSpec
        self.sliding_window = window_size
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        
        self.block_size = 64 # GPU default (TODO: update for TPU)
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Implements DeepseekV4SWACache.get_kv_cache_spec 
        # GPU implementation in vllm/vllm/v1/attention/backends/mla/sparse_swa.py
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=576, # GPU settings used in vllm/vllm/v1/attention/backends/mla/sparse_swa.py
            model_version="deepseek_v4",
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        # Filler that is eeded to satisfy the AttentionLayerBase interface but is not used in practice
        return PallasMLAttentionBackend

    def process_weights_after_loading(self, _act_dtype: torch.dtype) -> None:
        # Doesn't hold weights of its own so
        # process_weights_after_loading should be a no-op.
        pass

    def forward(self) -> None:
        # This class only sets up the cache specs for the SWA cache.
        # SWA KV insertion and SWA attention are the
        # responsibility of TpuDeepseekV4MultiHeadLatentAttentionWrapper.forward.
        pass


class TpuDeepseekV4MLAAttention(MLAAttention):
    """
    Analagous to VllmMLAAttention in tpu_inference/layers/vllm/custom_ops/mla_attention.py.
    Responsibilities: 
        - Registers the KV cache for the DSv4 attention.
        - Receives the q/kv inputs from TpuDeepseekV4MultiHeadLatentAttentionWrapper and 
          calls the kernel implementation.

    (GPU reference: 
    vllm/vllm/model_executor/layers/deepseek_v4_attention.py::DeepseekV4MLAAttention).

    Responsibilities (all TODO):
      1. Orchestration: GPU deepseek_v4_attention.py::DeepseekV4MLAAttention routes tokens into _forward_prefill / _forward_decode.
         We will need to confirm whether to follow this GPU design.
         Metadata is built by TpuDeepseekSparseSWAMetadataBuilder (backends/deepseek_v4_mla.py)
         before dispatching, providing slot_mappings for the caches.
      2. Extract top-k index preparation: for decode, gather compressed block indices
         from topk_indices_buffer (filled by the Lightening Indexer).
      3. Call the DSv4 MLA kernel implementation, which reads from three caches:
           (A) swa_cache   — raw uncompressed KV, last window_size tokens (primary k_cache)
           (B) state_cache — stores the <compress_ratio> accumulated
                             (kv, score) slots, computes weighted sum, writes compressed result
                             to mla_cache.
           (C) mla_cache   — compressed KV, one slot per compress_ratio tokens (extra_k_cache)
         The Pallas kernel also writes to all three caches via start_update_kv_cache:
           (A) swa_cache   — every token, via swa_slot_mapping from metadata
           (B) state_cache — every token, via state_slot_mapping;
                             accumulates (kv, score) where score comes from fused_wkv_wgate
                             gate projection. Defined by CompressorStateCache in
                             deepseek_compressor.py.
           (C) mla_cache   — writes to mla_cache when
                             state_slot_mapping[t] % block_size == compress_ratio - 1
    """

    def __init__(
        self,
        head_dim: int,
        compress_ratio: int,
        prefix: str,
        cache_config: CacheConfig,
        vllm_config: VllmConfig,
    ) -> None:
        # We can't call super().__init__() because it expects a GPU backend.
        # Instead we call nn.Module.__init__(self) and instantiate variables manually.
        nn.Module.__init__(self)
        self.kv_cache = torch.tensor([])
        self.layer_name = prefix
        
        # Attributes read by kv_cache_manager.py
        self.head_size = head_dim         # used to compute head_size_set
        self.num_kv_heads = 1             # MLA always uses 1 KV head
        self.attn_type = AttentionType.DECODER
        self.kv_sharing_target_layer_name = None
        self.sliding_window = None        # main MLA cache has no sliding window
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        
        # SWA-only layers (compress_ratio <= 1) have no main MLA KV cache.
        if self.compress_ratio <= 1:
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=576, # GPU settings used in vllm/vllm/model_executor/layers/deepseek_v4_attention.py
            model_version="deepseek_v4",
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        # Filler that is eeded to satisfy the AttentionLayerBase interface but is not used in practice
        return PallasMLAttentionBackend

    def process_weights_after_loading(self, _act_dtype: torch.dtype) -> None:
        # No-op since this is just an orchestrator class that doesn't own any weights.
        pass

    def forward(
        self,
        q: torch.Tensor, # [T, num_heads, head_dim]
        kv: torch.Tensor, # [T, 1, head_dim]
        positions: torch.Tensor,
        output: torch.Tensor, # [T, num_heads, head_dim]
    ) -> None:
        # TODO: fill in with logic similar to PallasMLAttentionBackendImpl which will 
        # assemble all of the inputs expected by the DSv4 kernel call (e.g. collect 
        # the indexer top-K, call tpu_inference/layers/vllm/backends/deepseek_v4_mla.py::TpuDeepseekSparseSWAMetadataBuilder),
        # potentially split into prefill vs. decode batches and call mla_attention for each type of 
        # batch respectively.
        #
        # Reference GPU implementation:
        #   vllm/vllm/model_executor/layers/deepseek_v4_attention.py
        output.zero_()


@DeepseekV4MultiHeadLatentAttentionWrapper.register_oot
class TpuDeepseekV4MultiHeadLatentAttentionWrapper(
    DeepseekV4MultiHeadLatentAttentionWrapper
):
    """Implements the attention layer logic for DSv4 (currently a stub!).
    Prepares the inputs needed for calling the MLA kernel (via TpuDeepseekV4MLAAttention) and
    processes/projects the kernel outputs.

    Analagous to VllmMultiHeadLatentAttentionWrapper in tpu_inference/layers/vllm/custom_ops/mla_attention.py.
    GPU reference is in vllm/vllm/model_executor/layers/deepseek_v4_attention.py.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # We need to avoid super().__init__ because it triggers an error
        # when calling torch.cuda.Event() and asserting CUDA device capability,
        nn.Module.__init__(self)

        vllm_config: VllmConfig = mla_modules.vllm_config

        # Store dimensions that will be used by the forward implementation.
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim          # = qk_nope_head_dim + qk_rope_head_dim
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.window_size = window_size
        self.prefix = prefix

        # Projection modules (weights loaded by vLLM's normal weight-loading path).
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b
        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a
        self.wo_b = mla_modules.wo_b

        self.rotary_emb = mla_modules.rotary_emb

        # SWA KV cache helper — always present, even for full-MLA layers.
        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.swa_cache_layer = TpuDeepseekV4SWACache(
            head_dim=head_dim,
            window_size=window_size,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

        self.mla_attn = TpuDeepseekV4MLAAttention(
            head_dim=head_dim,
            compress_ratio=self.compress_ratio,
            prefix=prefix,
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

        # Register the wrapper so the custom op can retrieve it.
        # GPU does the same at:
        # vllm/vllm/model_executor/layers/deepseek_v4_attention.py:264
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Instantiate DeepseekCompressor to register the state_cache pages.
        # NOTE: the actual compression is assigned to the MLA kernel rather than
        # having a separate Compressor kernel like in the GPU implementation.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = _compressor_module.DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO: implement the V4 attention layer.
        # Refer to the GPU implmenetation vllm/vllm/model_executor/layers/deepseek_v4_attention.py:
        #   1. fused_wqa_wkv(hidden_states)                    [T, q_lora_rank + head_dim]
        #      split → qr [T, q_lora_rank], kv [T, head_dim]
        #   2. q_norm(qr) → q_c  [T, q_lora_rank]             (RMSNorm w/o weight)
        #      kv_norm(kv) → kv_c [T, head_dim]
        #   3. wq_b(q_c) → q [T, num_heads * head_dim]        → reshape [T, num_heads, head_dim]
        #   4. Split q → q_nope [T, num_heads, nope_head_dim]
        #               q_pe   [T, num_heads, rope_head_dim]
        #      Split kv_c → kv_nope [T, kv_lora_rank]
        #                   k_pe    [T, rope_head_dim]
        #   5. RoPE(q_pe, k_pe, positions) via rotary_emb
        #   7. Call MLA kernel via TpuDeepseekV4MLAAttention.forward
        #   8. wo_a (grouped BMM)
        #   9. wo_b → final output [T, hidden_size]
        return hidden_states