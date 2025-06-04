
DimSharding = str | Sequence[str] | None | type(UNCONSTRAINED)
ShardingAxis = Sequence[DimSharding] | None

@dataclass
class Sharding_Strategy:
    fsdb: Int | None = None
    tensor_parrelism: Int | None = None
    expert_parrelism: Int | None = None
    sequence_parrelism: Int | None = None

@dataclass
class AttentionShardingConfig:
    # demo only suffix needs to be verified 
    # Activation for attn:
    activation_attention_bsd: ShardingAxis | None = None
    # Activation for attn out:
    activation_attention_out_bsd: ShardingAxis | None = None
    # Activation for q:
    activation_q_bsd: ShardingAxis | None = None  
    # Attention Out activation
    attn_o_bsqh: ShardingAxis | None = None 
    # Q vector:
    query_bshq: ShardingAxis | None = None 
    # K/V vector: 
    keyvalue_bsmq: ShardingAxis | None = None  

    # Attention Q weight:
    attn_q_weight_qdh: ShardingAxis | None = None  
    # Attention K weight:
    attn_k_weight_mdh: ShardingAxis | None = None  
    # Attention V weight
    attn_v_weight_mdh: ShardingAxis | None = None 
    # Attention Out weight.
    attn_o_weight_qhd: ShardingAxis | None = None

    # K/V cache.
    keyvalue_generate_mode_cache_bsmh: ShardingAxis | None = None
    keyvalue_prefill_mode_cache_bsmh: ShardingAxis | None = None

@dataclass
class MlpShardingConfig:
    # Activation for ffw:
    activation_ffw_bsd: ShardingAxis | None = None  
    # FFW hidden activation:
    ffw_hidden_bsf: ShardingAxis | None = None  
    # FFW weight:
    ffw_weight_df: ShardingAxis | None = None  
    # MoE weights 
    moe_edf: ShardingAxis | None = None  

@dataclass
class ShardingConfig:

    attention_sharding: AttentionShardingConfig
    mlp_sharding: MlpShardingConfig

    # Embedding
    emb_weight_vd: ShardingAxis | None = None  # Embed/softmax weight

    # Activation between layer:
    activation_bsd: ShardingAxis | None = None  
    # Final activation:
    prelogit_bsd: ShardingAxis | None = None  
    # Logit activation:
    logits_bsv: ShardingAxis | None = None
    # RMS norm scale weight
    norm_scale: ShardingAxis | None = None
    # Sharding on pipeline stages.
    pipeline: shd.DimSharding = None  

class Sharding:
  """sharding.
  axis naming table
  b: batch
  s: sequence length
  d: d_model
  f: d_ff, hidden_dim of ffw
  v: vocab size
  c: expert cpacity
  k: top K per token
  g: number of groups
  h: head dim in Attention
  q: number of query heads
  m: number of k/v heads
  e: number of experts
  ...
  """
    sharding_strategy: Sharding_Strategy
    sharding_cfg: ShardingConfig
    logical_mesh: LogicalMesh # ['x'=2, 'y'=4, 'z'=2]

    def __init__(self, flags_cfg):
      self.sharding_strategy = Sharding_Strategy(flags_cfg)
      ...
    def get_sharding_cfg(self) -> ShardingConfig:
      ...
    def build_logical_mesh(self) -> LogicalMesh:
      ...
    def make_sharding_config(self, name: str) -> ShardingConfig:

        # Build Logical Mesh based on parallelism yaml
        self.logical_mesh = build_logical_mesh()

        # Map Logical Mesh to Tensor Dim, i.e 'x' -> Batch
        if shading_strategy.fsdb > 0:
            ...
        if shading_strategy.tensor_parrelism > 0:
            ...
        if shading_strategy.expert_parrelism > 0:
            ...
        
        self.sharding_cfg = ShardingConfig(...)
        return self.sharding_cfg



  
