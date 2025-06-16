
import jax
import jax.numpy as jnp
import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Optional
from jaxtyping import Float, Array, Int
from typing import Any, Callable

# Flax and JAX sharding imports
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Local or project-specific imports (placeholders, adjust as needed)
# It's assumed these would be defined elsewhere in your project.
# For example, you might have: from .shared_types import KVCache, ShardingConfig, Quantization
from tpu_commons.models.jax.common.constants import *
from tpu_commons.models.jax.common.sharding import *


class Quantization: pass
class KVCache: pass

# A dummy for modeling_flax_utils which might contain activation functions
class MockFlaxUtils:
    ACT2FN = {
        'silu': nnx.silu,
        'gelu': nnx.gelu,
        'relu': nnx.relu,
    }
modeling_flax_utils = MockFlaxUtils()


# Type alias for Initializer for cleaner type hints
Initializer = Callable[..., jax.Array]

@dataclasses.dataclass
class ParamFactory:
    """A factory for creating nnx.Param objects with shared RNGs and initializers."""
    rngs: nnx.Rngs
    initializer: Initializer

    def create_kernel_init(self, shape: tuple[int, ...], sharding: NamedSharding, dtype: Any = jnp.float32) -> nnx.Param:
        """Creates an nnx.Param using the factory's RNG stream and initializer."""
        param_data = self.initializer(self.rngs.params(), shape, dtype)
        return nnx.Param(param_data, sharding=sharding)

@dataclass
class RuntimeParams:
# A layer wised runtime parameters that may be needed when initializing the live blocks, i.e. Attention
# That way, if a new block need to be added, i.e. lora, we won't have to update the interfaces for all blocks
    kv_cache: Optional[KVCache] = None
    sharding_cfg: Optional[ShardingConfig] = None
    quantization: Optional[Quantization] = None 

@dataclass
class Config:
    """Base config class with a robust factory method."""

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **kwargs):
        if cfg is None:
            cfg = {}
        cfg.update(kwargs)

        required_params = {
            f.name
            for f in fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }

        # Check if any of the truly required parameters are missing from the provided config.
        missing_params = required_params - set(cfg.keys())
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {cls.__name__}: {', '.join(sorted(list(missing_params)))}"
            )

        known_params = {f.name for f in fields(cls)}
        filtered_cfg = {k: v for k, v in cfg.items() if k in known_params}

        return cls(**filtered_cfg)
    
@dataclass
class RMSNorm(nnx.Module):
  """nn.RMSNorm with scale param default at 0."""
  dims: int
  mesh: Mesh
  param_factory: ParamFactory
  sharding_cfg: ShardingConfig  # Kept for API consistency
  epsilon: float = 1e-6
  with_scale: bool = True
  dtype: Any = jnp.float32
  quant: Any | None = None

  def __post_init__(self):
    """Initializes the scale parameter."""
    # No Sharding
    scale_sharding = NamedSharding(self.mesh, PartitionSpec())
    self.scale = self.param_factory.create_kernel_init(
        shape=(self.dims,),
        sharding=scale_sharding,
        dtype=self.dtype
    )

  def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    """Applies RMS Normalization."""
    x = jnp.asarray(x, jnp.float32)
    
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_x = x * jax.lax.rsqrt(var + self.epsilon)
    
    return (normed_x * self.scale.value).astype(self.dtype)
  

@dataclass
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

  def __init__(
      self,
      cfg: FFWConfig,
      mesh: Mesh,
      sharding_cfg: ShardingConfig,
      kernel_init: Initializer = nnx.initializers.lecun_normal(),
      *,
      rngs: nnx.Rngs,
  ):
    self.cfg = cfg
    self.mesh = mesh
    self.sharding_cfg = sharding_cfg
    self.kernel_init = kernel_init

    self.create_sharding()
    self.kernel_gating_DF = nnx.Param(
      self.kernel_init(rngs.params(), (cfg.d_model, cfg.hidden_size), cfg.dtype),
      sharding=self.df_sharding)
    self.kernel_up_proj_DF = nnx.Param(
      self.kernel_init(rngs.params(), (cfg.d_model, cfg.hidden_size), cfg.dtype),
      sharding=self.df_sharding)
    self.kernel_down_proj_FD = nnx.Param(
      self.kernel_init(rngs.params(), (cfg.hidden_size, cfg.d_model), cfg.dtype),
      sharding=self.fd_sharding)

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

  def create_sharding(self):
    self.activation_sharding = dict()
    self.activation_sharding['prefill'] =  NamedSharding(
      self.mesh, P(self.sharding_cfg.activation_ffw_bsd))
    self.activation_sharding['decode'] =  NamedSharding(
      self.mesh, P(self.sharding_cfg.activation_ffw_bsd))
    self.df_sharding =  NamedSharding(
      self.mesh, P(self.sharding_cfg.ffw_weight_df))
    self.fd_sharding =  NamedSharding(
      self.mesh, P(self.sharding_cfg.ffw_weight_fd))

    return 

@dataclass
class EmbedderConfig(Config):
  vocab_size: int
  d_model: int
  dtype: Any = jnp.float32
  normalize_embeddings: bool = False

@dataclass
class Embedder(nnx.Module):
  cfg: EmbedderConfig
  mesh: Mesh
  param_factory: ParamFactory
  sharding_cfg: ShardingConfig
  quant: Any | None = None


  def __post_init__(self):
    self.create_sharding()
    self.input_embedding_table_VD = self.param_factory.create_kernel_init(
        shape=(self.cfg.vocab_size, self.cfg.d_model),
        sharding=self.dv_sharding,
        dtype=self.cfg.dtype
    )
  
  def __call__(self, x, decode=False):
    if decode:
      return self.decode(x)
    else:
      return self.encode(x)

  def decode(self, x: Float[Array, "B S D"]) -> Float[Array, "B S V"]:
    x_BSD = nnx.with_sharding_constraint(x, self.prelogit_bsd)
    logits_BSV = jnp.einsum('BSD,VD -> BSV' , x_BSD, self.input_embedding_table_VD.value)
    return logits_BSV

  def encode(self, x: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    embedding_BSD = self.input_embedding_table_VD.value[x]
    if self.cfg.normalize_embeddings:
      embedding_BSD *= jnp.sqrt(self.cfg.d_model).astype(self.cfg.dtype)
    return embedding_BSD

  def create_sharding(self):
    self.prelogit_bsd =  NamedSharding(self.mesh, PartitionSpec(self.sharding_cfg.prelogit_bsd))
    self.dv_sharding =  NamedSharding(self.mesh, PartitionSpec(self.sharding_cfg.vocab_dv))


  