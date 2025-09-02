from dataclasses import dataclass, field
from typing import Any, List, Mapping


@dataclass
class ModelConfig():
    max_model_len: int = 2048
    max_prefill_len: int = 1024
    prefill_batch_size: int = 1
    decode_batch_size: int = 1
    block_size: int = 16
    num_layers: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    vocab_size: int = 32000
    model: str = "llama3"
    hf_config: str = ""
    architectures: List[str] = field(default_factory=list)
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    hf_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class VllmConfig():
    additional_config: Mapping[str, Any] = field(default_factory=dict)
    # Set default max_model_len to turn off warnings.
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(max_model_len=1024))
