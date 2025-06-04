class RMSNorm(nn.Module):
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


class FFWConfig:
  d_model: int
  hidden_size: int

  def __init__(self, yaml_config):
      ...
  def make(self, name, sharding_cfg=None, quantization=None) -> FFW:
      ...
      return FFW(
          d_model=self.d_model
          hidden_size=self.hidden_size
          cfg=self,
          sharding_cfg=sharding_cfg,
          quantization=quantization,
      )

class FFW(nn.Module):
  """Dense Feed Forward Layer"""
  d_model: int
  hidden_size: int
  cfg: Config = None
  sharding_cfg: ShardingConfig = default_sharding()
  quantization: Quantization | None = None

  def setup(self):
    ...
  def __call__(self, x):
    ...

class EmbedderConfig:
  vocab_size: int
  d_model: int
  normalize_embeddings: bool = False

  def __init__(self, flags_cfg):
    ...
  
  def make(self, name, sharding_cfg=None, quantization=None) -> Embdder:
    ...
    return Embdder(
          vocab_size=self.vocab_size
          d_model=self.d_model
          normalize_embeddings=self.normalize_embeddings
          cfg=self,
          sharding_cfg=sharding_cfg,
          quantization=quantization,      
    )

class Embdder(nn.Module):
  cfg: EmbedderConfig
  vocab_size: int
  d_model: int
  normalize_embeddings: bool = False
  sharding_cfg: ShardingConfig = default_sharding()
  quantization: Quantization | None = None

  def setup(self):
    ...
  def __call__(self, x):
    ...
    return self.decode(x)
  def decode(self, x):
    ...


  