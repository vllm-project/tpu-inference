from constants import LOGICAL_MESH_AXIS_NAME
@dataclass
class Op_Sharding_Strategy:
    tensor_parallelism: Int | None = None
    expert_parallelism:: Int | None = None
    sequence_parallelism:: Int | None = None
    data_parallelism:: Int | None = None

@dataclass
class Sharding_Strategy:
    prefill_Sharding_Strategy: Op_Sharding_Strategy
    decode_Sharding_Strategy: Op_Sharding_Strategy

class OpMesh:
    def __init__(self, device_grid , axis_names: Tuple[str, ...]):
        self.device_grid = device_grid
        self.axis_names = axis_names

class Mesh:
    def __init__(self, prefill_mesh: OpMesh, decode_mesh: OpMesh):
        self.prefill_mesh = prefill_mesh
        self.decode_mesh = decode_mesh
    def get_mesh(op_mode=OPERATION_MODE.DECODE):
        if op_mode == OPERATION_MODE.PREFILL:
            return self.prefill_mesh
        elif op_mode == OPERATION_MODE.DECODE:
            return self.decode_mesh


class logical_mesh_axes:

    @classmethod
    def update(logical_mesh_axis_name, op_mode):
        if op_mode == OPERATION_MODE.PREFILL:
            self.logical_mesh_axis_name_prefill = logical_mesh_axis_name
        elif op_mode == OPERATION_MODE.DECODE:
            self.logical_mesh_axis_name_decode = logical_mesh_axis_name
    
    def get_axes(op_mode=OPERATION_MODE.DECODE):
         if op_mode == OPERATION_MODE.PREFILL:
            return self.logical_mesh_axis_name_prefill
        elif op_mode == OPERATION_MODE.DECODE:
            return self.logical_mesh_axis_name_decode      

@dataclass
class ShardingConfig:

    # Activation for attn:
    activation_attention_bsd: logical_mesh_axes | None = None
    # Activation for attn out:
    activation_attention_out_bsd: logical_mesh_axes | None = None
    # Activation for q:
    activation_q_bsd: logical_mesh_axes | None = None  
    # Attention Out activation
    attn_o_bsqh: logical_mesh_axes | None = None 
    # Q vector:
    query_bshq: logical_mesh_axes | None = None 
    # K/V vector: 
    keyvalue_bsmq: logical_mesh_axes | None = None  

    # Attention Q weight:
    attn_q_weight_qdh: logical_mesh_axes | None = None  
    # Attention K weight:
    attn_k_weight_mdh: logical_mesh_axes | None = None  
    # Attention V weight
    attn_v_weight_mdh: logical_mesh_axes | None = None 
    # Attention Out weight.
    attn_o_weight_qhd: logical_mesh_axes | None = None

    # K/V cache.
    keyvalue_generate_mode_cache_bsmh: logical_mesh_axes | None = None
    keyvalue_prefill_mode_cache_bsmh: logical_mesh_axes | None = None

    # Activation for ffw:
    activation_ffw_bsd: logical_mesh_axes | None = None  
    # FFW hidden activation:
    ffw_hidden_bsf: logical_mesh_axes | None = None  

    # FFW weight:
    ffw_weight_df: logical_mesh_axes | None = None
    # FFW weight:
    ffw_weight_fd: logical_mesh_axes | None = None  
    # MoE weights 
    moe_weights_edf: logical_mesh_axes | None = None 
    moe_weights_efd: logical_mesh_axes | None = None  
    moe_router_de: logical_mesh_axes | None = None  

    # Embedding
    emb_weight_vd: logical_mesh_axes | None = None
    # Activation between layer:
    activation_bsd: logical_mesh_axes | None = None  
    # Final activation:
    prelogit_bsd: logical_mesh_axes | None = None  
    # Logit activation:
    logits_bsv: logical_mesh_axes | None = None
    # RMS norm scale weight
    norm_scale: logical_mesh_axes | None = None
    



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
    LOGICAL_MESH_AXIS_NAME: LOGICAL_MESH_AXIS_NAME

    def __init__(self, flags_cfg):
      self.sharding_strategy = Sharding_Strategy(flags_cfg)
    def __post__init__(self):
      self.mesh(
        prefill_mesh=self.build_mesh_per_op(self.sharding_strategy.prefill_Sharding_Strategy),
        decode_mesh=self.build_mesh_per_op(self.sharding_strategy.decide_Sharding_Strategy))

    def validate_sharding_strategy(self,):
    """ 
    Validate if the sharding strategy is correct and be able to fit in devices
    """

    def get_sharding_cfg(self) -> ShardingConfig:
      return self.sharding_cfg
    
    def build_mesh_per_op(self, strategy: Op_Sharding_Strategy) -> LogicalMesh:
        # TODO to decide if we should name as x,y,z or 'data','tensor', 'expert' etc
        axis_order = {
          "seq": strategy.sequence_parallelism,
          "tensor": strategy.tensor_parallelism,
          "expert": strategy.expert_parallelism,
          "data": strategy.data_parallelism,
        }
        # TODO add logic to infer axis when the degree is -1
        axis_names = []
        mesh_shape = []
        for axis, dim in axis_order.items()
            if dim > 1:
                axis_names.append(axis)
                mesh_shape.append(dim)

        
         mesh = Mesh(
            np.asarray(jax.devices()).reshape(mesh_shape),
            axis_names=mesh_axis_names)     
         return mesh

    def make_sharding_config(self) -> ShardingConfig:
        #TODO organize into update_prefill() and update_decode for each axis
        #TODO verify the sharding axes
        activation_attention_bsd=logical_mesh_axes.update((BATCH_AXIS_NAME, None, ATTN_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        activation_attention_out_bsd=logical_mesh_axes.update((BATCH_AXIS_NAME, None, ATTN_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        activation_q_bsd=logical_mesh_axes.update((BATCH_AXIS_NAME, None, ATTN_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        attn_o_bsqh=logical_mesh_axes.update((BATCH_AXIS_NAME, None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        query_bshq=logical_mesh_axes.update((BATCH_AXIS_NAME, None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        activation_ffw_bsd=logical_mesh_axes.update((BATCH_AXIS_NAME, None, MLP_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        ffw_hidden_bsf=logical_mesh_axes.update((BATCH_AXIS_NAME, None, MLP_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE) 
        ffw_weight_df=logical_mesh_axes.update((BATCH_AXIS_NAME, None, MLP_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        ffw_weight_fd=logical_mesh_axes.update((MLP_TENSOR_AXIS_NAME, None) , OPERATION_MODE.DECODE) 
        moe_weights_edf=logical_mesh_axes.update((EXPERT_AXIS_NAME, None, MOE_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)
        moe_weights_efd=logical_mesh_axes.update((EXPERT_AXIS_NAME, MOE_TENSOR_AXIS_NAME, None) , OPERATION_MODE.DECODE)

        # PREFILL
        activation_attention_bsd=logical_mesh_axes.update((BATCH_AXIS_NAME, SEQUENCE_AXIS_NAME, ATTN_TENSOR_AXIS_NAME) , OPERATION_MODE.DECODE)

        self.sharding_cfg = ShardingConfig(
            activation_attention_bsd=activation_attention_bsd,
            activation_attention_out_bsd=activation_attention_out_bsd,
            activation_q_bsd=activation_q_bsd,
            attn_o_bsqh=attn_o_bsqh,
            query_bshq=query_bshq ,
            activation_ffw_bsd=activation_ffw_bsd, 
            ffw_hidden_bsf=ffw_hidden_bsf,
            ffw_weight_df=ffw_weight_df,
            ffw_weight_fd=ffw_weight_fd,
            moe_weights_edf=moe_weights_edf, 
            moe_weights_efd=moe_weights_efd,     
        ) 
        return self.sharding_cfg



  
