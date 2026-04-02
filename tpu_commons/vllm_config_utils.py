from dataclasses import dataclass, field
from typing import Any, List, Mapping
import numpy as np
import numpy as np
from tpu_commons.logger import init_logger
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     # Import the class; all warnings will be suppressed
#     from microbenchmark_utils import ModelConfig
logger = init_logger(__name__)
power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens
@dataclass

@dataclass
class HFConfig():
    
    # Qwen 3 - 32 B
    architectures: List[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    rope_scaling: Mapping[str, Any] = field(
        default_factory=lambda: {"factor": 2.0, "type": "linear"})
    rms_norm_eps: str = 1e-06,
    hidden_act: str = "silu"
    hidden_size: int = 5120
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    tie_word_embeddings: bool = False
    num_hidden_layers: int = 64
    intermediate_size: int = 25600
    vocab_size: int = 151936
    rope_theta: float = 100000.0
    rms_norm_eps: str = 1e-06
    # Add other HF config parameters as needed

    # DeepseekV3
    # architectures: List[str] = field(default_factory=lambda: ["DeepseekV3ForCausalLM"])
    # rope_scaling: Mapping[str, Any] = field(
    #     default_factory=lambda: {"factor": 2.0, "type": "linear"})
    # rms_norm_eps: str = 1e-06,
    # hidden_act: str = "silu"
    # hidden_size: int = 5120
    # num_attention_heads: int = 128
    # num_key_value_heads: int = 128
    # head_dim: int = 128
    # tie_word_embeddings: bool = False
    # num_hidden_layers: int = 8
    # intermediate_size: int = 25600
    # vocab_size: int = 151936
    # rope_theta: float = 100000.0
    # Add other HF config parameters as needed

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
    model: str = ""
    hf_config: HFConfig = field(
        default_factory=lambda: HFConfig())
    dtype: str = "bfloat16"
    # parallel_config = field(default_factory=dict)
    # architectures: List[str] = field(default_factory=list)
    override_generation_config: dict[str, Any] = field(default_factory=dict)

    def get_vocab_size(self) -> int:
        return self.vocab_size

@dataclass
class VllmConfig():
    additional_config: Mapping[str, Any] = field(default_factory=dict)
    # Set default max_model_len to turn off warnings.
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(max_model_len=1024))
    