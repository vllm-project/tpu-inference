"""MLA KV Cache Utilities - Inline version"""
import os
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

@dataclass
class MLAKVCacheSpec:
    """Specification for MLA KV cache allocation."""
    num_layers: int
    layer_names: List[str] = field(default_factory=list)
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    block_size: int = 16
    dtype: torch.dtype = torch.bfloat16
    
    @property
    def head_size(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim
    
    @property
    def num_kv_heads(self) -> int:
        return 1
    
    def __post_init__(self):
        if not self.layer_names:
            self.layer_names = [f"layer_{i}" for i in range(self.num_layers)]
    
    @classmethod
    def from_model_config(cls, model_config) -> Optional["MLAKVCacheSpec"]:
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(model_config, "hf_text_config", None)
        if hf_config is None:
            return None
        kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
        if kv_lora_rank is None:
            return None
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("MLA model config missing num_hidden_layers")
        return cls(
            num_layers=num_layers,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=getattr(hf_config, "qk_rope_head_dim", 64),
            dtype=getattr(model_config, "dtype", torch.bfloat16),
        )

def create_mla_kv_cache_group_spec(spec: MLAKVCacheSpec):
    """Create a KVCacheGroupSpec for MLA models."""
    from vllm.v1.core.kv_cache_utils import KVCacheGroupSpec
    from vllm.v1.kv_cache_interface import AttentionSpec
    
    attention_spec = AttentionSpec(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=spec.dtype,
    )
    return KVCacheGroupSpec(spec.layer_names, attention_spec)

def ensure_mla_cache_groups(kv_cache_config, model_config) -> bool:
    """Ensure kv_cache_config has cache groups for MLA models."""
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if groups and len(groups) > 0:
        return False
    mla_spec = MLAKVCacheSpec.from_model_config(model_config)
    if mla_spec is None:
        return False
    block_size = getattr(kv_cache_config, "block_size", 16)
    if block_size:
        mla_spec.block_size = block_size
    group_spec = create_mla_kv_cache_group_spec(mla_spec)
    kv_cache_config.kv_cache_groups = [group_spec]
    logger.info(f"[MLA] Created cache group: head_size={mla_spec.head_size}")
    return True
