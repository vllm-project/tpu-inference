from jaxtyping import Float, Array
import jax
import jax.numpy as jnp

@dataclass
class RuntimeParams:
# A layer wised runtime parameters that may be needed when initializing the live blocks, i.e. Attention
# That way, if a new block need to be added, i.e. lora, we won't have to update the interfaces for all blocks
    kv_cache: Optional[KVCache] = None
    sharding_cfg: Optional[ShardingConfig] = None
    quantization: Optional[Quantization] = None 

@dataclass
class Config:

  @classmethod
  def from_cfg(cls, flags_cfg: dict):
      required_params = {f.name for f in fields(cls)}
      provided_params = set(flags_cfg.keys())
      missing_params  = required_params - provided_params
      if missing_params:
          ...

      flags = {k: flags_cfg[k] for k in required_params}
      return cls(**flags)
    

class RMSNorm(nnx.Module):
  """nn.RMSNorm with scale param default at 0."""

  def __init__(
      self,
      dims: int,
      mesh: Optional[Mesh] = None,
      epsilon: float = 1e-6,
      with_scale: bool = True,
      dtype: Any = jnp.float32,
      scale_init: nnx.initializers.Initializer = nnx.initializers.ones,
      num_groups: int = 1
      quant: 
  ):
    self.dims = dims
    self.mesh = mesh
    self.epsilon = epsilon
    self.with_scale = with_scale
    self.dtype = dtype
    self.scale_init = scale_init
    self.num_groups = num_groups
    # default scale is 1
    self.scale = nnx.Param(self.scale_init((self.dims,), self.dtype))

    sharding(self.mesh)

  def __call__(self, x) -> jnp.ndarray:
    if self.num_groups == 1:
      x = jnp.asarray(x, jnp.float32)
      var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
      normed_x = x * jax.lax.rsqrt(var + self.epsilon, self.dtype)
      normed_x *= self.scale
      return normed_x
  
  def sharding(self, mesh):
    ...


class FFWConfig(Config):
  d_model: int
  hidden_size: int
  act: str
  dtype: Any = jnp.float32

class FFW(nnx.Module):
  """Dense Feed Forward Layer"""
  cfg: FFWConfig
  mesh: Mesh
  kernel_init: Initializer # TODO create factories for initializer(?)
  quant: Quantization | None = None
  sharding_cfg: ShardingConfig

  def __call__(self, x, op_mode):
    # TODO consider to create factories for einsum(?)
    x = jnp.asarray(x, jnp.float32)
    x = nnx.with_sharding_constraint(x, self.activation_sharding[op_mode]) 

    with jax.named_scope("wi_0"):
      gating_BSF = jnp.einsum('BSD,DF -> BSF' , x, self.kernel_gating_DF.value)
      activated_gating_BSF = modeling_flax_utils.ACT2FN[self.cfg.act](gating_BSF)
    with jax.named_scope("wi_1"):
      up_proj_BSF = jnp.einsum('BSD,DF -> BSF' , x, self.kernel_up_proj_DF.value)
    fuse_BSF = activated_gating_BSF * up_proj_BSF
    with jax.named_scope("wo"):
      output_BSD = jnp.einsum('BSF,FD -> BSD' ,fuse_BSF, self.kernel_down_proj_FD.value)
    
    return output_BSD


  def setup(self):
    self.create_sharding()
    self.kernel_gating_DF = nnx.Param(
      self.kernel_init((cfg.d_model, cfg.hidden_size), cfg.dtype),
      sharding=self.df_sharding)
    self.kernel_up_proj_DF = nnx.Param(
      self.kernel_init((cfg.d_model, cfg.hidden_size), cfg.dtype),
      sharding=self.df_sharding)
    self.kernel_down_proj_FD = nnx.Param(
      self.kernel_init((cfg.hidden_size, cfg.d_model), cfg.dtype),
      sharding=self.fd_sharding)

  def create_sharding():
    self.activation_sharding = dict()
    self.activation_sharding['prefill'] =  NamedSharding(
      self.mesh, P(self.sharding_cfg.activation_bsd.get_axes(OPERATION_MODE.PREFILL)))
    self.activation_sharding['decode'] =  NamedSharding(
      self.mesh, P(self.sharding_cfg.activation_bsd.get_axes(OPERATION_MODE.DECODE)))
    self.df_sharding =  NamedSharding(
      self.mesh, P(self.sharding_cfg.ffw_weight_df.get_axes()))
    self.fd_sharding =  NamedSharding(
      self.mesh, P(self.sharding_cfg.ffw_weight_fd.get_axes()))

    return 

# TODO to be implemented
class EmbedderConfig(Config):
  vocab_size: int
  d_model: int
  dtype: Any = jnp.float32
  normalize_embeddings: bool = False

class Embedder(nnx.Module):
  cfg: EmbedderConfig
  mesh: Mesh
  embedding_init: Initializer # TODO create factories for initializer(?)
  sharding_cfg: ShardingConfig
  quant: Quantization | None = None


  def setup(self):
    self.input_embedding_table_VD = nnx.Param(
      self.embedding_init((cfg.vocab_size, cfg.d_model), cfg.dtype),
      sharding=self.dv_sharding)
  def __call__(self, x, decode=False):
    if decode:
      return self.decode(x)
    else:
      return self.encode(x)
  def decode(self, x):
    x_BSD = nnx.with_sharding_constraint(x, self.prelogit_bsd)
    logits_BSV = jnp.einsum('BSD,DV -> BSV' , x_BSD, self.input_embedding_table)
    return logits_BSV

  def encode(self, x):
    x_BSD = nnx.with_sharding_constraint(x, self.prelogit_bsd)
    embedding_BSD = self.input_embedding_table_VD[(x, )]
    return embedding_BSD

  def create_sharding():
    self.prelogit_bsd =  NamedSharding(
      self.mesh, P(self.sharding_cfg.prelogit_bsd.get_axes()))
    self.dv_sharding =  NamedSharding(
      self.mesh, P(self.sharding_cfg.vocab_dv.get_axes()))


  