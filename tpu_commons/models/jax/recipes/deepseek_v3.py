from dataclasses import dataclass

from tpu_commons.models.jax.common.sharding import (
    ShardingRulesConfig, ATTN_TENSOR_AXIS_NAME, 
    ATTN_HEAD_AXIS_NAME
)

@dataclass
class DeepSeekV3ShardingRulesConfig(ShardingRulesConfig):
    # MLA Query down projection weight: (Dim, QueryLoraRank)
    attn_mla_qa_weight_da: tuple = (None, None)
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_qb_weight_anh: tuple = (None, None, None)
    # MLA KV down projection weight: (Dim, KVLoRA + QKRopeHeadDim)
    attn_mla_kva_weight_da: tuple = (None, None)
    # MLA KV up projection weight: (KVLoRA, NumHeads, QKNopeHeadDim + VHeadDim)
    attn_mla_kvb_weight_anh: tuple = (None, None, None)

    def __post_init__(self):
        super().__post_init__()
        # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
        self.prefill_rules.attn_mla_qb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)
        # MLA KV up projection weight: (KVLoRA, NumHeads, QKNopeHeadDim + VHeadDim)
        self.prefill_rules.attn_mla_kvb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)
        # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
        self.generate_rules.attn_mla_qb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)
        # MLA KV up projection weight: (KVLoRA, NumHeads, QKNopeHeadDim + VHeadDim)
        self.generate_rules.attn_mla_kvb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)

