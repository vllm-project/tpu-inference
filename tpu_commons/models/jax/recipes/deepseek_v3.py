from dataclasses import dataclass

from tpu_commons.models.jax.common.sharding import (ATTN_HEAD_AXIS_NAME,
                                                    ATTN_TENSOR_AXIS_NAME,
                                                    ShardingRulesConfig)


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


@dataclass
class DeepSeekV3PrefillShardingRulesConfig(DeepSeekV3ShardingRulesConfig):
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_qb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                     ATTN_TENSOR_AXIS_NAME)
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_kvb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                      ATTN_TENSOR_AXIS_NAME)


@dataclass
class DeepSeekV3GenerateShardingRulesConfig(DeepSeekV3ShardingRulesConfig):
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_qb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                     ATTN_TENSOR_AXIS_NAME)
    # MLA KV up projection weight: (KVLoRA, NumHeads, QKNopeHeadDim + VHeadDim)
    attn_mla_kvb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                      ATTN_TENSOR_AXIS_NAME)
