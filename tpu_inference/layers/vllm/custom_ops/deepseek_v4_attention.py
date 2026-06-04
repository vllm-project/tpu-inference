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
"""TPU interception for DeepSeek-V4 MLA attention (torchax path).

``DeepseekV4MLA`` is a plain ``nn.Module`` that vLLM instantiates directly
(``self.mla_attn = DeepseekV4MLA(...)`` in ``deepseek_v4/amd/model.py``). Unlike
the MHC ops or the attention-impl bases, it is NOT a vLLM ``CustomOp`` and has no
``register_oot`` hook, so there is no registry-based way to swap it. Its
constructor is also CUDA-bound (asserts a CUDA device capability and allocates
``torch.cuda.Event``), so it cannot run on TPU as-is.

Instead we substitute the class symbol before the model is built. Because
``amd/model.py`` does ``from ...attention import DeepseekV4MLA``, the name is
bound into the ``amd.model`` module namespace at import time; patching
``attention.DeepseekV4MLA`` alone would not take effect. ``patch_deepseek_v4_mla_cls``
rebinds it on ``amd.model`` directly. It is invoked from
``_maybe_patch_for_deepseek_v4`` in ``vllm_model_wrapper`` while ``is_rocm`` is
forced True and the package has been reloaded onto the AMD implementation.
"""
import torch
import torch.nn as nn
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.attention import DeepseekV4MLA
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from tpu_inference.layers.vllm.backends.flash_attn_mla import \
    PallasMLAttentionBackend
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class VllmDeepseekV4MLAAttention(nn.Module, AttentionLayerBase):

    def __init__(
        self,
        head_dim: int,
        compress_ratio: int,
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        nn.Module.__init__(self)
        self.prefix = prefix
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.cache_dtype = cache_config.cache_dtype

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def process_weights_after_loading(self, act_order: bool = False) -> None:
        pass

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PallasMLAttentionBackend

    def forward(
            self,
            q: torch.Tensor,  # [T, num_heads, head_dim]
            kv: torch.Tensor,  # [T, 1, head_dim]
            positions: torch.Tensor,
            output: torch.Tensor,  # [T, num_heads, head_dim]
    ) -> None:
        logger.error(
            "DeepseekV4MLA.forward is not implemented, just a pass-through for now"
        )
        return q


class VllmDeepseekV4MLA(DeepseekV4MLA):

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
        vllm_config: VllmConfig,
        fused_wqa_wkv: torch.nn.Module,
        q_norm: torch.nn.Module,
        wq_b: torch.nn.Module,
        kv_norm: torch.nn.Module,
        wo_a: torch.nn.Module,
        wo_b: torch.nn.Module,
        attn_sink: torch.nn.Module,
        rotary_emb: torch.nn.Module,
        indexer: torch.nn.Module | None,
        indexer_rotary_emb: torch.nn.Module,
        topk_indices_buffer: torch.Tensor | None,
        aux_stream_list: list | None,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.mla_attn = VllmDeepseekV4MLAAttention(
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            prefix=prefix,
            cache_config=cache_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logger.error(
            "VllmDeepseekV4MLA.forward is not implemented, just a pass-through for now"
        )
        return hidden_states


def patch_deepseek_v4_mla_cls() -> None:
    """Rebind ``DeepseekV4MLA`` to the TPU subclass for DS V4 model module.

    Must run after ``vllm.models.deepseek_v4.amd.model`` is imported (it holds
    its own ``from ...attention import DeepseekV4MLA`` reference) and before the
    model is constructed.
    """
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    ds_v4_amd_model.DeepseekV4MLA = VllmDeepseekV4MLA
    logger.info("Patched DeepseekV4MLA -> VllmDeepseekV4MLA for TPU.")
