@dataclass
class RuntimeParams:
# A layer wised runtime parameters that may be needed when initializing the live blocks, i.e. Attention
# That way, if a new block need to be added, i.e. lora, we won't have to update the interfaces for all blocks
    kv_cache: Optional[KVCache] = None
    sharding_cfg: Optional[ShardingConfig] = None
    quantization: Optional[Quantization] = None 

class RMSNorm(nnx.Module):
  """nn.RMSNorm with scale param default at 0."""
  epsilon: float = 1e-6
  with_scale: bool = True
  param_dtype: Any = jnp.float32
  scale_init: nn.initializers.Initializer = nn.initializers.ones
  feature_axes: int | Sequence[int] = -1
  num_groups: int = 1
  scale_shift: float = 1.0

  def __call__(self, x):
    ...


class FFWConfig(Config):
  d_model: int
  hidden_size: int

  @classmethod
  def from_cfg(cls, flags_cfg: dict):
      required_params = {f.name for f in fields(cls)}
      provided_params = set(flags_cfg.keys())
      missing_params  = required_params - provided_params
      if missing_params:
          ...

      ffw_flags = {k: flags_cfg[k] for k in required_params}
      return cls(**ffw_flags)

  def make(self, name, runtime_param: Optional[layer.RuntimeParams] = None) -> FFW:
      ...
      return FFW(
          cfg=self,
          sharding_cfg=runtime_param.sharding_cfg,
          quantization=runtime_param.quantization,
      )

class FFW(nnx.Module):
  """Dense Feed Forward Layer"""
  cfg: Config = None
  kernel: 
  sharding_cfg: ShardingConfig = default_sharding()
  quantization: Quantization | None = None

  def setup(self):
    ...
  def __call__(self, x, op_mode):
    ...
  def sharding(op_mode):
    ...

class EmbedderConfig(Config):
  vocab_size: int
  d_model: int
  normalize_embeddings: bool = False

  @classmethod
  def from_cfg(cls, flags_cfg: dict):
      required_params = {f.name for f in fields(cls)}
      provided_params = set(flags_cfg.keys())
      missing_params  = required_params - provided_params
      if missing_params:
          ...

      embedder_flags = {k: flags_cfg[k] for k in required_params}
      return cls(**embedder_flags)

  def make(self, name, runtime_param: Optional[layer.RuntimeParams] = None) -> Embdder:
    ...
    return Embedder(
          cfg=self,
          sharding_cfg=runtime_param.sharding_cfg,
          quantization=runtime_param.quantization,      
    )

class Embedder(nnx.Module):
  cfg: EmbedderConfig
  vocab: 
  sharding_cfg: ShardingConfig = default_sharding()
  quantization: Quantization | None = None

  def setup(self):
    ...
  def __call__(self, x):
    ...
    return self.decode(x)
  def decode(self, x):
    ...
  def sharding():
    ...


  