  """Current Used abbreviation. 
  #TODO follow dragon book
  B: batch
  S: sequence length
  T: token len
  D: d_model
  F: d_ff, hidden_dim of ffw/expert
  V: vocab size
  C: expert capacity
  K: top K per token
  G: number of groups
  H: head dim in Attention
  Q: number of query heads
  N: number of KV heads
  E: number of experts
  """

  """
  Sharding_Strategy -> 
  Sharding:
   self.Mesh i.e. Mesh((2,4,4,4), ('dp', 'sp', 'ep', 'tp')) 128 devices
   self.ShardingConfig 
        prefill_axes i.e. activation_attention_bsd = (None, 'dp', 'ep')
        decode_axes i.e. activation_attention_bsd = ('dp', None, 'ep')
  """

from constants import LOGICAL_MESH_AXIS_NAME

@dataclass
class Sharding_Strategy:
    tensor_parallelism: Int = 1
    expert_parallelism: Int = 1
    sequence_parallelism: Int = 1
    data_parallelism: Int = 1

class Mesh:
    def __init__(self, device_grid , axis_names: Tuple[str, ...]):
        self.device_grid = device_grid
        self.axis_names = axis_names

@dataclass
class logical_mesh_axes:
    prefill_axes: Tuple[str, ...] | None = None
    decode_axes: Tuple[str, ...] # (None, 'dp', 'ep')

    def update(self, axes: Tuple[str, ...], op_mode: OPERATION_MODE):
        if op_mode == OPERATION_MODE.PREFILL:
            self.prefill_axes = axes
        elif op_mode == OPERATION_MODE.DECODE:
            self.decode_axes = axes

    def get_axes(self, op_mode: OPERATION_MODE = OPERATION_MODE.DECODE):
        if op_mode == OPERATION_MODE.PREFILL:
            return self.prefill_axes
        elif op_mode == OPERATION_MODE.DECODE:
            return self.decode_axes  

@dataclass
class ShardingConfig:

    # Activation for attn:
    # an example: logical_mesh_axes(prefill_axes=(None, 'sp', 'tp'), decode_axes=('sp', None, 'ep')
    activation_attention_bsd: logical_mesh_axes | None = None
    # Activation for attn out:
    activation_attention_out_bsd: logical_mesh_axes | None = None
    # Activation for q:
    activation_q_bsd: logical_mesh_axes | None = None  
    # Attention Out activation
    attn_o_bsqh: logical_mesh_axes | None = None 
    # Q vector:
    query_bsqh: logical_mesh_axes | None = None 
    # K/V vector: 
    keyvalue_bsnh: logical_mesh_axes | None = None  

    # Attention Q weight:
    attn_q_weight_qdh: logical_mesh_axes | None = None  
    # Attention K weight:
    attn_k_weight_ndh: logical_mesh_axes | None = None  
    # Attention V weight
    attn_v_weight_ndh: logical_mesh_axes | None = None 
    # Attention Out weight.
    attn_o_weight_qhd: logical_mesh_axes | None = None

    # K/V cache.
    keyvalue_generate_mode_cache_bsnh: logical_mesh_axes | None = None
    keyvalue_prefill_mode_cache_bsnh: logical_mesh_axes | None = None

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
    # vocab sharding
    vocab_dv: logical_mesh_axes | None = None

    


class Sharding:
    """
    Sharding block, which store and generate the ShardingConfig
    for tensors based on the Sharding_Strategy
    """
    sharding_strategy: Sharding_Strategy
    sharding_cfg: ShardingConfig
    LOGICAL_MESH_AXIS_NAME: LOGICAL_MESH_AXIS_NAME

    def __init__(self, flags_cfg):
      self.sharding_strategy = Sharding_Strategy(flags_cfg)
      self.mesh = self.build_mesh()

    def validate_sharding_strategy(self,):
    """ 
    Validate if the sharding strategy is correct and be able to fit in devices
    """

    #TODO check num_devices % parallelism == 0

    #TODO check num_devices == multiply(parallelism(with inferred)) 

    def get_sharding_cfg(self) -> ShardingConfig:
      return self.sharding_cfg
    
    def build_mesh(self, strategy: Sharding_Strategy) -> LogicalMesh:
        # TODO to decide if we should name as x,y,z or 'data','tensor', 'expert' etc
        axis_order = {
          "sp": strategy.sequence_parallelism,
          "tp": strategy.tensor_parallelism,
          "ep": strategy.expert_parallelism,
          "dp": strategy.data_parallelism,
        }
        # TODO add logic to infer axis when the degree is -1
        mesh_axis_names = []
        mesh_shape = []
        for axis, dim in axis_order.items()
            if dim > 1:
                mesh_axis_names.append(axis)
                mesh_shape.append(dim)

        
         mesh = Mesh(
            np.asarray(jax.devices()).reshape(mesh_shape),
            axis_names=mesh_axis_names)     
         return mesh

    def _create_axes(self, decode_axis, prefill_axis=None):
        return logical_mesh_axes(
            prefill_axes=prefill_axis,
            decode_axes=decode_axis
        )

    #TODO add method to read sharding config directly user specified config file

    def make_sharding_config(self) -> ShardingConfig:
        #TODO organize into update_prefill() and update_decode for each axis
        #TODO verify the sharding axes
        self.sharding_cfg = ShardingConfig()
        
        self.sharding_cfg.activation_attention_bsd=_create_axes(
            decode_axis=(BATCH_AXIS_NAME, None, ATTN_TENSOR_AXIS_NAME),
            prefill_axis=(None, SEQUENCE_AXIS_NAME, ATTN_TENSOR_AXIS_NAME))
        self.sharding_cfg.activation_attention_out_bsd=_create_axes(
            decode_axis=(None, None, ATTN_TENSOR_AXIS_NAME))
        self.sharding_cfg.activation_q_bsd=_create_axes(
            decode_axis=(None, None, ATTN_TENSOR_AXIS_NAME))
        self.sharding_cfg.attn_o_bsqh=_create_axes(
            decode_axis=(None, None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME))
        self.sharding_cfg.query_bshq=_create_axes(
            decode_axis=(None, None, ATTN_HEAD_AXIS_NAME, ATTN_TENSOR_AXIS_NAME))
        self.sharding_cfg.activation_ffw_bsd=_create_axes(
            decode_axis=(None, None, MLP_TENSOR_AXIS_NAME))
        self.sharding_cfg.ffw_hidden_bsf=_create_axes(
            decode_axis=(None, None, MLP_TENSOR_AXIS_NAME)) 
        self.sharding_cfg.ffw_weight_df=_create_axes(
            decode_axis=(None, None, MLP_TENSOR_AXIS_NAME))
        self.sharding_cfg.ffw_weight_fd=_create_axes(
            decode_axis=(MLP_TENSOR_AXIS_NAME, None)) 
        self.sharding_cfg.moe_weights_edf=_create_axes(
            decode_axis=(EXPERT_AXIS_NAME, None, MOE_TENSOR_AXIS_NAME))
        self.sharding_cfg.moe_weights_efd=logical_mesh_axes.update(
            decode_axis=(EXPERT_AXIS_NAME, MOE_TENSOR_AXIS_NAME, None))

        return self.sharding_cfg



  
