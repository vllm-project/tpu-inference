from constants import LOGICAL_MESH_AXIS_NAME
@dataclass
class Sharding_Strategy:
    tensor_parallelism: Int | None = None
    expert_parallelism:: Int | None = None
    sequence_parallelism:: Int | None = None
    data_parallelism:: Int | None = None

class Mesh:
    def __init__(self, device_grid , axis_names: Tuple[str, ...]):
        self.device_grid = device_grid
        self.axis_names = axis_names
        ...


@dataclass
class AttentionShardingConfig:
    # demo only, suffix needs to be verified
   
    # Activation for attn:
    activation_attention_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation for attn out:
    activation_attention_out_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation for q:
    activation_q_bsd: LOGICAL_MESH_AXIS_NAME | None = None  
    # Attention Out activation
    attn_o_bsqh: LOGICAL_MESH_AXIS_NAME | None = None 
    # Q vector:
    query_bshq: LOGICAL_MESH_AXIS_NAME | None = None 
    # K/V vector: 
    keyvalue_bsmq: LOGICAL_MESH_AXIS_NAME | None = None  

    # Attention Q weight:
    attn_q_weight_qdh: LOGICAL_MESH_AXIS_NAME | None = None  
    # Attention K weight:
    attn_k_weight_mdh: LOGICAL_MESH_AXIS_NAME | None = None  
    # Attention V weight
    attn_v_weight_mdh: LOGICAL_MESH_AXIS_NAME | None = None 
    # Attention Out weight.
    attn_o_weight_qhd: LOGICAL_MESH_AXIS_NAME | None = None

    # K/V cache.
    keyvalue_generate_mode_cache_bsmh: LOGICAL_MESH_AXIS_NAME | None = None
    keyvalue_prefill_mode_cache_bsmh: LOGICAL_MESH_AXIS_NAME | None = None

    ...

@dataclass
class MlpShardingConfig:
    # Activation for ffw:
    activation_ffw_bsd: LOGICAL_MESH_AXIS_NAME | None = None  
    # FFW hidden activation:
    ffw_hidden_bsf: LOGICAL_MESH_AXIS_NAME | None = None  
    # FFW weight:
    ffw_weight_df: LOGICAL_MESH_AXIS_NAME | None = None  
    # MoE weights 
    moe_edf: LOGICAL_MESH_AXIS_NAME | None = None  
    ...

@dataclass
class OpSpecificShardingConfig:

    attention_sharding: AttentionShardingConfig
    mlp_sharding: MlpShardingConfig
    # Embedding
    emb_weight_vd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation between layer:
    activation_bsd: LOGICAL_MESH_AXIS_NAME | None = None  
    # Final activation:
    prelogit_bsd: LOGICAL_MESH_AXIS_NAME | None = None  
    # Logit activation:
    logits_bsv: LOGICAL_MESH_AXIS_NAME | None = None
    # RMS norm scale weight
    norm_scale: LOGICAL_MESH_AXIS_NAME | None = None
    ...

@dataclasses.dataclass
class ShardingConfig:
    prefill: OpSpecificShardingConfig
    generate: OpSpecificShardingConfig


class Sharding:
  """sharding.
  axis naming table
  b: batch
  s: sequence length
  d: d_model
  f: d_ff, hidden_dim of ffw
  v: vocab size
  c: expert capacity
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
    logical_mesh: Mesh

    def __init__(self, flags_cfg):
      self.sharding_strategy = Sharding_Strategy(flags_cfg)
      self.logical_mesh = get_logical_mesh()
      ...
    def get_sharding_cfg(self) -> ShardingConfig:
      ...
    def get_logical_mesh(self, physical_shape) -> LogicalMesh:
      # Build the logic mesh from physical mesh
      device_grid = np.asarray(jax.devices()).reshape(physical_shape)
      mesh = Mesh(device_grid, LOGICAL_MESH_AXIS_NAME)
      ...
      
      return mesh
      ...
    def make_sharding_config(self, op_mode) -> ShardingConfig:

        if op_mode == 'prefill'"
          if shading_strategy.data_parallelism > 0:
              ...
          if shading_strategy.tensor_parallelism > 0:
              ...
          if shading_strategy.expert_parallelism > 0:
              ...
          if shading_strategy.sequence_parallelism > 0:
              ...
          prefill_sharding_cfg = OpSpecificShardingConfig(...)

        elif op_mode == 'generate'"
          if shading_strategy.data_parallelism > 0:
              ...
          if shading_strategy.tensor_parallelism > 0:
              ...
          if shading_strategy.expert_parallelism > 0:
              ...
          if shading_strategy.sequence_parallelism > 0:
              ...
          generate_sharding_cfg = OpSpecificShardingConfig(...)

        self.sharding_cfg = ShardingConfig(
          prefill=prefill_sharding_cfg,
          generate=generate_sharding_cfg)
        return self.sharding_cfg



  
