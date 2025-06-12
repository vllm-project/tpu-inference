from constants import LOGICAL_MESH_AXIS_NAME
import dataclass
from typing import Int, List, Tuple

@dataclass
class ICIParallelismSettings:
    tensor_parallelism: Int | None = None
    expert_parallelism: Int | None = None
    prefill_sequence_parallelism: Int | None = None
    decode_data_parallelism: Int | None = None
    decode_data_prefill_seq_parallesim: Int | None = None


class Mesh:
    def __init__(self, device_grid , axis_names: Tuple[str, ...]):
        self.device_grid = device_grid
        self.axis_names = axis_names
        ...


MESH_AXES = ['decode_data_prefill_seq', 'data', 'seq', 'tensor', 'expert']

@dataclass
class AttentionLogicalAxesRules:
    # demo only, suffix needs to be verified
   
   # Activation for attn:
    activation_attention_batch: List[str]

    # Activation for attn:
    activation_attention_bsd: List[str]

    # Activation for attn out:
    activation_attention_out_bsd: List[str]
    
    # Activation for q:
    activation_q_bsd: List[str]

    # Attention Out activation
    attn_o_bsqh: List[str]

    # Q vector:
    query_bshq: List[str]
    
    # K/V vector: 
    keyvalue_bsmq: List[str]

    # Attention Q weight:
    attn_q_weight_qdh: List[str]
    
    # Attention K weight:
    attn_k_weight_mdh: List[str]
    
    # Attention V weight
    attn_v_weight_mdh: List[str]
    
    # Attention Out weight.
    attn_o_weight_qhd: List[str]
    
    # K/V cache.
    keyvalue_generate_mode_cache_bsmh: List[str]
    keyvalue_prefill_mode_cache_bsmh: List[str]



@dataclass
class DenseLayerLogicalAxesRules:
    # Activation for ffw:
    activation_ffw_bsd: List[str]
    # FFW hidden activation:
    ffw_hidden_bsf: List[str]
    # FFW weight:
    ffw_weight_df: List[str]
    # MoE weights 
    moe_edf: List[str]
    ...

@dataclass
class OpSpecificLogicalAxesRules:
    # Embedding
    emb_weight_vd: List[str]
    # Activation between layer:
    activation_bsd: List[str]
    # Final activation:
    prelogit_bsd: List[str]
    # Logit activation:
    logits_bsv: List[str]
    # RMS norm scale weight
    norm_scale: List[str]

# @dataclasses.dataclass
# class ShardingConfig: 
#     prefill: OpSpecificShardingConfig
#     generate: OpSpecificShardingConfig


# class Sharding:
#   """sharding.
#   axis naming table
#   b: batch
#   s: sequence length
#   d: d_model
#   f: d_ff, hidden_dim of ffw
#   v: vocab size
#   c: expert capacity
#   k: top K per token
#   g: number of groups
#   h: head dim in Attention
#   q: number of query heads
#   m: number of k/v heads
#   e: number of experts
#   ...
#   """
#     sharding_strategy: Sharding_Strategy
#     sharding_cfg: ShardingConfig
#     logical_mesh: Mesh

#     def __init__(self, flags_cfg):
#       self.sharding_strategy = Sharding_Strategy(flags_cfg)
#       self.logical_mesh = get_logical_mesh()
#       ...
#     def get_sharding_cfg(self) -> ShardingConfig:
#       ...
#     def get_logical_mesh(self, physical_shape) -> LogicalMesh:
#       # Build the logic mesh from physical mesh
#       device_grid = np.asarray(jax.devices()).reshape(physical_shape)
#       mesh = Mesh(device_grid, LOGICAL_MESH_AXIS_NAME)
#       ...
      
#       return mesh
#       ...
#     def make_sharding_config(self, op_mode) -> ShardingConfig:

#         if op_mode == 'prefill'"
#           if shading_strategy.data_parallelism > 0:
#               ...
#           if shading_strategy.tensor_parallelism > 0:
#               ...
#           if shading_strategy.expert_parallelism > 0:
#               ...
#           if shading_strategy.sequence_parallelism > 0:
#               ...
#           prefill_sharding_cfg = OpSpecificShardingConfig(...)

#         elif op_mode == 'generate'"
#           if shading_strategy.data_parallelism > 0:
#               ...
#           if shading_strategy.tensor_parallelism > 0:
#               ...
#           if shading_strategy.expert_parallelism > 0:
#               ...
#           if shading_strategy.sequence_parallelism > 0:
#               ...
#           generate_sharding_cfg = OpSpecificShardingConfig(...)

#         self.sharding_cfg = ShardingConfig(
#           prefill=prefill_sharding_cfg,
#           generate=generate_sharding_cfg)
#         return self.sharding_cfg



  