import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Optional, Type

import jax
import jax.numpy as jnp
# Flax and JAX sharding imports
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Float, Int

from tpu_commons.models.jax.common.attention import Attention, AttentionConfig, AttentionMetadata
from tpu_commons.models.jax.common.kv_cache import (KVCacheConfig,
                                                    StandardUpdater)
from tpu_commons.models.jax.common.moe import MoE
from tpu_commons.models.jax.common.sharding import ShardingConfig


# A dummy for modeling_flax_utils which might contain activation functions
class FlaxUtils:
    """A dummy class to namespace activation functions, mimicking external utilities."""
    ACT2FN = {
        'silu': nnx.silu,
        'gelu': nnx.gelu,
        'relu': nnx.relu,
    }


modeling_flax_utils = FlaxUtils()

# Type alias for Initializer for cleaner type hints
Initializer = Callable[..., jax.Array]


@dataclasses.dataclass
class ParamFactory:
    """A factory for creating nnx.Param objects with shared RNGs and initializers.

    This class simplifies the creation of parameters by holding common
    configuration like the RNG stream and the weight initialization function.

    Attributes:
        rngs: An `nnx.Rngs` object to provide RNG streams for parameter initialization.
        initializer: A callable (e.g., a kernel initializer from JAX) used to
            generate parameter data.
    """
    rngs: nnx.Rngs
    initializer: Initializer

    def create_kernel_init(self,
                           shape: tuple[int, ...],
                           sharding: NamedSharding,
                           dtype: Any = jnp.float32) -> nnx.Param:
        """Creates an nnx.Param using the factory's RNG stream and initializer.

        Args:
            shape: The shape of the parameter tensor to create.
            sharding: The `NamedSharding` object that specifies how the parameter
                should be distributed across devices.
            dtype: The data type of the parameter.

        Returns:
            An `nnx.Param` instance containing the initialized data and sharding info.
        """
        param_data = self.initializer(self.rngs.params(), shape, dtype)
        return nnx.Param(param_data, sharding=sharding)


@dataclass
class RuntimeParams:
    """A container for runtime parameters needed by neural network blocks.

    This dataclass acts as a flexible container to pass objects that are only
    available at runtime (like a pre-allocated KV cache or dynamic sharding
    configurations) into the initialization of stateful modules. This avoids
    having to update the constructor signature of every module when a new
    runtime dependency is introduced.

    Attributes:
        kv_cache: The key-value cache object for attention layers.
        sharding_cfg: The configuration for tensor sharding.
        quantization: Configuration for quantization schemes.
    """
    kv_cache: Any = None
    sharding_cfg: Any = None
    quantization: Any = None


@dataclass
class Config:
    """Base configuration class with a robust factory method.

    This class provides a `from_cfg` classmethod that allows creating a config
    instance from a dictionary, ensuring that all required fields are present
    and ignoring any extraneous keys.
    """

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **kwargs):
        """Creates a config instance from a dictionary and/or keyword arguments.

        This factory method validates that all fields without default values
        are provided in the input dictionary or keyword arguments.

        Args:
            cfg: A dictionary of configuration parameters.
            **kwargs: Additional configuration parameters passed as keyword arguments.

        Returns:
            An instance of the configuration class.

        Raises:
            ValueError: If any required parameters are missing.
        """
        if cfg is None:
            cfg = {}
        cfg.update(kwargs)

        required_params = {
            f.name
            for f in fields(cls) if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
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
    """An implementation of Root Mean Square Layer Normalization.

    Attributes:
        dims: The feature dimension to normalize over.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        epsilon: A small float added to the variance to avoid division by zero.
        with_scale: If True, learns a multiplicative scale parameter.
        dtype: The data type for computations.
        quant: Optional configuration for quantization.
    """
    dims: int
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    epsilon: float = 1e-6
    with_scale: bool = True
    dtype: Any = jnp.float32
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the scale parameter."""
        # No Sharding on the scale parameter
        scale_sharding = NamedSharding(self.mesh, P())
        self.scale = self.param_factory.create_kernel_init(
            shape=(self.dims, ), sharding=scale_sharding, dtype=self.dtype)

    def __call__(self, x: Float) -> Float:
        """Applies RMS Normalization to the input tensor.

        Args:
            x: The input tensor. The normalization is applied over the last dimension.

        Returns:
            The normalized tensor with the same shape as the input.
        """
        x = jnp.asarray(x, jnp.float32)

        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_x = x * jax.lax.rsqrt(var + self.epsilon)

        return (normed_x * self.scale.value).astype(self.dtype)


@dataclass
class FFWConfig(Config):
    """Configuration for the Feed-Forward (FFW) layer.

    Attributes:
        d_model: The dimension of the model.
        hidden_size: The size of the intermediate hidden layer.
        act: The name of the activation function to use (e.g., 'silu').
        dtype: The data type for computations.
    """
    d_model: int
    hidden_size: int
    act: str
    dtype: Any = jnp.float32


@dataclass
class FFW(nnx.Module):
    """A Gated Feed-Forward Network (FFN) layer.

    This module consists of two linear projections (gating and up-projection),
    an element-wise multiplication of the activated gating projection and the
    up-projection, followed by a final downward projection.

    Attributes:
        cfg: The `FFWConfig` configuration object.
        mesh: The JAX device mesh.
        param_factory: The factory for creating parameters.
        sharding_cfg: The configuration for tensor sharding.
        quant: Optional configuration for quantization.
    """
    cfg: FFWConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the weight kernels for the feed-forward layer."""
        self.create_sharding()
        self.kernel_gating_DF = self.param_factory.create_kernel_init(
            shape=(self.cfg.d_model, self.cfg.hidden_size),
            dtype=self.cfg.dtype,
            sharding=self.df_sharding)
        self.kernel_up_proj_DF = self.param_factory.create_kernel_init(
            shape=(self.cfg.d_model, self.cfg.hidden_size),
            dtype=self.cfg.dtype,
            sharding=self.df_sharding)
        self.kernel_down_proj_FD = self.param_factory.create_kernel_init(
            shape=(self.cfg.hidden_size, self.cfg.d_model),
            dtype=self.cfg.dtype,
            sharding=self.fd_sharding)

    def __call__(self, x, op_mode):
        """Performs the forward pass of the FFW layer.

        Args:
            x: The input tensor of shape `(batch, sequence, d_model)`.
            op_mode: The operational mode ('prefill' or 'generate'), used for
                selecting sharding annotations.

        Returns:
            The output tensor of shape `(batch, sequence, d_model)`.
        """
        # TODO consider to create factories for einsum(?)
        x = jnp.asarray(x, jnp.float32)
        x = nnx.with_sharding_constraint(x, self.activation_ffw_btd[op_mode])

        with jax.named_scope("wi_0"):
            gating_BTF = jnp.einsum('BTD,DF -> BTF', x,
                                    self.kernel_gating_DF.value)
            activated_gating_BTF = modeling_flax_utils.ACT2FN[self.cfg.act](
                gating_BTF)
        with jax.named_scope("wi_1"):
            up_proj_BTF = jnp.einsum('BTD,DF -> BTF', x,
                                     self.kernel_up_proj_DF.value)
        fuse_BTF = activated_gating_BTF * up_proj_BTF
        with jax.named_scope("wo"):
            output_BTD = jnp.einsum('BTF,FD -> BTD', fuse_BTF,
                                    self.kernel_down_proj_FD.value)

        return output_BTD

    def create_sharding(self):
        """Creates and sets sharding attributes for weights and activations."""
        mode_dependent_attrs = [
            "activation_ffw_btd",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.df_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.ffw_weight_df))
        self.fd_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.ffw_weight_fd))

        return


@dataclass
class EmbedderConfig(Config):
    """Configuration for the Embedder module.

    Attributes:
        vocab_size: The size of the vocabulary.
        d_model: The dimension of the embeddings/the model.
        dtype: The data type for the embedding table.
        normalize_embeddings: If True, scale embeddings by `sqrt(d_model)`.
    """
    vocab_size: int
    d_model: int
    dtype: Any = jnp.float32
    normalize_embeddings: bool = False


@dataclass
class Embedder(nnx.Module):
    """A module for token embedding and, optionally, decoding (tied embeddings).

    This class handles both the "encoding" step of converting token IDs to dense
    vectors and the "decoding" step of projecting model outputs back to logits
    over the vocabulary.

    Attributes:
        cfg: The `EmbedderConfig` configuration object.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """
    cfg: EmbedderConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the embedding table."""
        self.create_sharding()
        self.input_embedding_table_VD = self.param_factory.create_kernel_init(
            shape=(self.cfg.vocab_size, self.cfg.d_model),
            sharding=self.dv_sharding,
            dtype=self.cfg.dtype)

    def __call__(self, x, decode=False):
        """Dispatches to either the encode or decode method.

        Args:
            x: The input tensor. Either token IDs for encoding or hidden states
                for decoding.
            decode: A boolean flag. If False (default), performs encoding. If
                True, performs decoding.

        Returns:
            Either embedding vectors or logit scores.
        """
        if decode:
            return self.decode(x)
        else:
            return self.encode(x)

    def decode(self, x: Float) -> Float:
        """Projects hidden states to vocabulary logits.

        Args:
            x: The input tensor of hidden states from the model backbone, with
                shape `(batch, sequence, d_model)`.

        Returns:
            The output logits over the vocabulary, with shape
            `(batch, sequence, vocab_size)`.
        """
        x_BTD = nnx.with_sharding_constraint(x, self.prelogit_btd)
        logits_BTV = jnp.einsum('BTD,VD -> BTV', x_BTD,
                                self.input_embedding_table_VD.value)
        return logits_BTV

    def encode(self, x: Int) -> Float:
        """Converts integer token IDs to dense embedding vectors.

        Args:
            x: The input tensor of token IDs, with shape `(batch, sequence)`.

        Returns:
            The corresponding embedding vectors, with shape
            `(batch, sequence, d_model)`.
        """
        embedding_BTD = self.input_embedding_table_VD.value[x]
        if self.cfg.normalize_embeddings:
            embedding_BTD *= jnp.sqrt(self.cfg.d_model).astype(self.cfg.dtype)
        return embedding_BTD

    def create_sharding(self):
        """Creates and sets sharding attributes for weights and activations."""
        self.prelogit_btd = NamedSharding(
            self.mesh, P(self.sharding_cfg.generate_sharding_cfg.prelogit_btd))
        self.dv_sharding = NamedSharding(
            self.mesh, PartitionSpec(self.sharding_cfg.vocab_dv))

    def __post_init__(self):
        self.create_sharding()
        self.input_embedding_table_VD = self.param_factory.create_kernel_init(
            shape=(self.cfg.vocab_size, self.cfg.d_model),
            sharding=self.dv_sharding,
            dtype=self.cfg.dtype)

class TransformerBlockConfig(Config):
    """
    light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    """
    attention: AttentionConfig
    kv_cache: KVCacheConfig
    ffw: FFWConfig = None
    block_type: str = None
    rmsnorm_epsilon: float = None
    overrides: Mapping[str, Any] = None

    def _block_type_cls(self) -> FFW:
        if self.block_type.lower() == "moe":
            return MoE
        elif self.block_type.lower() == "dense":
            return FFW
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

    def from_cfg(self, flags_cfg):
        self.attention = AttentionConfig.from_cfg(flags_cfg)
        self.kv_cache = KVCacheConfig.from_cfg(flags_cfg)
        block_class = self._block_type_cls()
        self.ffw = block_class.from_cfg(flags_cfg)
        self.cfg = flags_cfg


@dataclass
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving
    """
    cfg: TransformerBlockConfig
    block_type: str
    param_factory: ParamFactory
    mesh: Mesh
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def _create_module(self, module_cls: Type[nnx.Module], cfg: Any,
                       **overrides) -> nnx.Module:
        args = {
            "mesh": self.mesh,
            "param_factory": self.param_factory,
            "sharding_cfg": self.sharding_cfg,
            "quant": self.quant
        }
        args.update(overrides)
        return module_cls(cfg=cfg, **args)

    def __post_init__(self):

        self.d_model = self.cfg.attention.d_model
        self.attn = self._create_module(Attention, cfg=self.cfg.attention)
        self.kv_cache = KVCache(
            cfg=self.cfg.kv_cache,
            mesh=self.mesh,
            sharding_cfg=self.sharding_cfg,
            updater=StandardUpdater(),
        )

        self.mlp = self._create_module(FFW, cfg=self.cfg.ffw_cfg)

        if self.block_type == "moe":
            self.moe = self._create_module(MoE, cfg=self.cfg.ffw_cfg)

        self.post_attention_norm = self._create_module(
            RMSNorm,
            cfg={"dims": self.cfg.d_model},
        )
        self.post_mlp_norm = self._create_module(
            RMSNorm, cfg={"dims": self.cfg.d_model})

    # TODO:
    def __call__(
        self,
        x,
        op_mode,
        is_prefill: bool,
        do_sampling: bool,
        kv_caches: List[KVCache_type],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        temperatures: jax.Array = None,
        top_ps: jax.Array = None,
        top_ks: jax.Array = None,
        *args, **kwargs) -> Tuple[List[KVCache_type], jax.Array, jax.Array]:
        new_cache, score = self.self_attn(x, op_mode, self.kv_cache,
                                          attention_metadata)
        x = self.post_attention_norm(x + score)
        if self.block_type == "moe":
            y = self.moe(x, op_mode)
        elif self.block_type == "dense":
            y = self.mlp(x, op_mode)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")
        
        logits = self.post_mlp_norm(x + y)

        return new_cache, logits
