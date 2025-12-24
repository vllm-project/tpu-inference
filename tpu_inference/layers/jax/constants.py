# Copyright 2025 Google LLC
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
"""
Current Used Abbreviation for Tensor Dimensions:
B: Batch size
T: Sequence Length (for Query tensors)
S: Sequence Length (for Key/Value tensors)
D: d_model, the embedding dimension of the model
F: d_ff, the hidden dimension of the feed-forward MLP layers
V: Vocab Size
H: Dimension of each attention head
N: Number of query heads in Attention
Q: Number of query heads (synonymous with N)
K: Number of Key/Value heads in Attention
C: Expert capacity in Mixture-of-Experts models
X: Number of activated experts per token in MoE
G: Number of groups in Grouped-Query Attention
E: Total number of experts in MoE
"""

import enum
from typing import Tuple, TypeAlias

import jax

KVCacheType: TypeAlias = Tuple[jax.Array, jax.Array]


class RouterType(enum.Enum):
    """Enum for router types."""
    TOP_K = 'top_k'


class OPERATION_MODE(enum.Enum):
    PREFILL = 1
    DECODE = 2


class HuggingFaceArgNames(enum.Enum):
    ## Modeling params
    HIDDEN_ACT: str = "hidden_act"
    HIDDEN_SIZE: str = "hidden_size"
    NUM_HIDDEN_LAYERS: str = "num_hidden_layers"
    RMS_NORM_EPS: str = "rms_norm_eps"
    ROPE_SCALING: str = "rope_scaling"
    ROPE_THETA: str = "rope_theta"
    VOCAB_SIZE: str = "vocab_size"

    # Block parameters
    SHARED_EXPERTS: str = "shared_experts"

    # FFW params
    INTERMEDIATE_SIZE: str = "intermediate_size"

    # Attention params
    HEAD_DIM: str = "head_dim"
    NUM_ATTENTION_HEADS: str = "num_attention_heads"
    NUM_KEY_VALUE_HEADS: str = "num_key_value_heads"
    ATTENTION_DROPOUT: str = "attention_dropout"
    ATTENTION_BIAS: str = "attention_bias"
    ATTENTION_CHUNK_SIZE: str = "attention_chunk_size"

    ## Llama4 Attention Params
    USE_QK_NORM: str = "use_qk_norm"
    TEMPERATURE_TUNING: str = "temperature_tuning"
    TEMPERATURE_TUNING_SCALE: str = "temperature_tuning_scale"
    TEMPERATURE_TUNING_FLOOR_SCALE: str = "temperature_tuning_floor_scale"

    # MLA params
    KV_LORA_RANK: str = "kv_lora_rank"
    Q_LORA_RANK: str = "q_lora_rank"
    QK_NOPE_HEAD_DIM: str = "qk_nope_head_dim"
    QK_ROPE_HEAD_DIM: str = "qk_rope_head_dim"
    V_HEAD_DIM: str = "v_head_dim"

    # MoE
    INTERMEDIATE_SIZE_MOE: str = "intermediate_size_moe"
    NUM_LOCAL_EXPERTS: str = "num_local_experts"  # Llama moe
    NUM_EXPERTS_PER_TOKEN: str = "num_experts_per_token"
    NUM_ROUTED_EXPERTS: str = "n_routed_experts"  # Deepseek moe
    NUM_SHARED_ROUTED_EXPERTS: str = "n_shared_experts"
    NUM_GROUPS: str = "n_group"
    ROUTED_SCALING_FACTOR: str = "routed_scaling_factor"
    TOPK_GROUP: str = "topk_group"
    NORM_TOPK_PROB: str = "norm_topk_prob"
    SCORING_FUNCTION: str = "scoring_func"

    ## Sampling params
    BOS_TOKEN_ID: str = "bos_token_id"
    EOS_TOKEN_ID: str = "eos_token_id"
