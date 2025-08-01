from dataclasses import dataclass, field, make_dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import DTypeLike, Float, Int
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.sharding import ShardingConfig


# A dummy for modeling_flax_utils which might contain activation functions
class FlaxUtils:
    """A dummy class to namespace activation functions, mimicking external utilities."""
    ACT2FN = {
        'silu': nnx.silu,
        'gelu': nnx.gelu,
        'relu': nnx.relu,
        'sigmoid': nnx.sigmoid,
        'softmax': nnx.softmax
    }


modeling_flax_utils = FlaxUtils()


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
        self.create_sharding()

    def __call__(self, x_TD: Float, op_mode='generate') -> Float:
        """Applies RMS Normalization to the input tensor.

        Args:
            x_TD: The input tensor. The normalization is applied over the last dimension.

        Returns:
            The normalized tensor with the same shape as the input.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD,
                                            self.activation_ffw_td[op_mode])

        with jax.named_scope("rms_norm_variance"):
            var_T1 = jnp.mean(jnp.square(x_TD), axis=-1, keepdims=True)
        with jax.named_scope("rms_norm_rsqrt"):
            normed_x_TD = x_TD * jax.lax.rsqrt(var_T1 + self.epsilon)

        with jax.named_scope("rms_norm_scale_apply"):
            normed_x_TD *= self.scale.value
        normed_x_TD = nnx.with_sharding_constraint(
            normed_x_TD, self.activation_ffw_td[op_mode])
        return normed_x_TD.astype(self.dtype)

    def generate_kernel(self, rngs: nnx.Rngs):
        self.scale = self.param_factory.create_scale_param(
            rngs,
            shape=(self.dims, ),
            sharding=self.scale_sharding,
            dtype=self.dtype)

    def create_sharding(self):
        """Creates and sets sharding attributes for weights and activations."""
        mode_dependent_attrs = [
            "activation_ffw_td",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_rules,
                                              attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(*generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # No Sharding on the scale parameter
        self.scale_sharding = NamedSharding(self.mesh, P())

        return


DenseFFWConfig = make_dataclass(
    "FFWConfig",
    [(HuggingFaceArgNames.HIDDEN_SIZE.value, int),
     (HuggingFaceArgNames.INTERMEDIATE_SIZE.value, int),
     (HuggingFaceArgNames.HIDDEN_ACT.value, str), ("dtype", DTypeLike),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))

DenseFFWConfig.__doc__ = f"""Configuration for the Dense Feed-Forward (FFW) layer.

     Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.INTERMEDIATE_SIZE.value}: The size of the intermediate hidden layer.
        {HuggingFaceArgNames.HIDDEN_ACT.value}: The name of the activation function to use (e.g., 'silu').
         dtype: The data type for computations.
         vllm_config: The VLLM config containing any overrides to apply.
     """


@dataclass
class DenseFFW(nnx.Module):
    """A Gated Feed-Forward Network (FFN) layer.

    This module consists of two linear projections (gating and up-projection),
    an element-wise multiplication of the activated gating projection and the
    up-projection, followed by a final downward projection.

    Attributes:
        cfg: The `DenseFFWConfig` configuration object.
        mesh: The JAX device mesh.
        param_factory: The factory for creating parameters.
        sharding_cfg: The configuration for tensor sharding.
        quant: Optional configuration for quantization.
    """
    cfg: DenseFFWConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        """Initializes the weight kernels for the feed-forward layer."""
        self.create_sharding()

    def __call__(self, x_TD, op_mode):
        """Performs the forward pass of the FFW layer.

        Args:
            x_TD: The input tensor of shape either `(sequence, d_model)`
            op_mode: The operational mode ('prefill' or 'generate'), used for
                selecting sharding annotations.

        Returns:
            The output tensor of shape `(batch, sequence, d_model)`.
        """
        # TODO consider to create factories for einsum(?)
        x_TD = jnp.asarray(x_TD, self.cfg.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD,
                                            self.activation_ffw_td[op_mode])
        act = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_ACT.value)
        with jax.named_scope("wi_0"):
            gating_TF = jnp.einsum('TD,DF -> TF', x_TD,
                                   self.kernel_gating_DF.value)
            activated_gating_TF = modeling_flax_utils.ACT2FN[act](gating_TF)
        with jax.named_scope("wi_1"):
            up_proj_TF = jnp.einsum('TD,DF -> TF', x_TD,
                                    self.kernel_up_proj_DF.value)
        fuse_TF = activated_gating_TF * up_proj_TF
        with jax.named_scope("wo"):
            output_TD = jnp.einsum('TF,FD -> TD', fuse_TF,
                                   self.kernel_down_proj_FD.value)

        return output_TD

    def generate_kernel(self, rngs: nnx.Rngs):
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        F = getattr(self.cfg, HuggingFaceArgNames.INTERMEDIATE_SIZE.value)

        self.kernel_gating_DF = self.param_factory.create_kernel_param(
            rngs,
            shape=(D, F),
            dtype=self.cfg.dtype,
            sharding=self.df_sharding)
        self.kernel_up_proj_DF = self.param_factory.create_kernel_param(
            rngs,
            shape=(D, F),
            dtype=self.cfg.dtype,
            sharding=self.df_sharding)
        self.kernel_down_proj_FD = self.param_factory.create_kernel_param(
            rngs,
            shape=(F, D),
            dtype=self.cfg.dtype,
            sharding=self.fd_sharding)

    def create_sharding(self):
        """Creates and sets sharding attributes for weights and activations."""
        mode_dependent_attrs = [
            "activation_ffw_td",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_rules,
                                              attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(*generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.df_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.ffw_weight_df))
        self.fd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.ffw_weight_fd))

        return


EmbedderConfig = make_dataclass(
    "EmbedderConfig",
    [(HuggingFaceArgNames.VOCAB_SIZE.value, int),
     (HuggingFaceArgNames.HIDDEN_SIZE.value, int), ("dtype", DTypeLike),
     ("normalize_embeddings", bool),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
EmbedderConfig.__doc__ = f"""Configuration for the Embedder module.

     Attributes:
         {HuggingFaceArgNames.VOCAB_SIZE.value}: The size of the vocabulary.
         {HuggingFaceArgNames.HIDDEN_SIZE.value}: The hidden dimension of the model.
         dtype: The data type for the embedding table.
         normalize_embeddings: If True, scale embeddings by `sqrt(d_model)`.
         vllm_config: The VLLM config containing any overrides to apply.
     """


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

    def generate_kernel(self, rngs: nnx.Rngs):
        V = getattr(self.cfg, HuggingFaceArgNames.VOCAB_SIZE.value)
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        self.input_embedding_table_VD = self.param_factory.create_kernel_param(
            rngs,
            shape=(V, D),
            sharding=self.vd_sharding,
            dtype=self.cfg.dtype)

    def decode(self, x_TD: Float) -> Float:
        """Projects hidden states to vocabulary logits.

        Args:
            x_TD: The input tensor of hidden states from the model backbone, with
                shape `(sequence, d_model)`.

        Returns:
            The output logits over the vocabulary, with shape
            `(sequence, vocab_size)`.
        """
        x_TD = jnp.asarray(x_TD, self.cfg.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.prelogit_td)

        with jax.named_scope("embedder_decode_projection"):
            logits_TV = jnp.einsum('VD,TD -> TV',
                                   self.input_embedding_table_VD.value, x_TD)
        return logits_TV

    def encode(self, x_T: Int) -> Float:
        """Converts integer token IDs to dense embedding vectors.

        Args:
            x_T: The input tensor of token IDs, with shape `(sequence, )`.

        Returns:
            The corresponding embedding vectors, with shape
            `(batch, sequence, d_model)`.
        """
        with jax.named_scope("embedder_encode_lookup"):
            embedding_TD = jnp.take(self.input_embedding_table_VD.value,
                                    x_T,
                                    axis=0)

        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        if self.cfg.normalize_embeddings:
            with jax.named_scope("embedder_normalize_embeddings"):
                embedding_TD *= jnp.sqrt(D).astype(self.cfg.dtype)
        return embedding_TD

    def create_sharding(self):
        """Creates and sets sharding attributes for weights and activations."""
        self.prelogit_td = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.prelogit_td))
        self.vd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.vocab_vd))


@dataclass
class LMhead(Embedder):
    """
    An Embedder that uses a (D, V) shaped embedding table, inheriting from
    the base Embedder class.

    This implementation overrides the kernel generation, encoding, and decoding
    methods to work with the transposed embedding matrix layout.
    """

    def generate_kernel(self, rngs: nnx.Rngs):
        V = getattr(self.cfg, HuggingFaceArgNames.VOCAB_SIZE.value)
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)

        self.input_embedding_table_DV = self.param_factory.create_kernel_param(
            rngs,
            shape=(D, V),
            sharding=self.dv_sharding,
            dtype=self.cfg.dtype)

    def __call__(self, x):
        """Dispatches to decode method.

        Args:
            x: The input tensor. Either token IDs for encoding or hidden states
                for decoding.
            decode: A boolean flag. If False (default), performs encoding. If
                True, performs decoding.

        Returns:
            Either embedding vectors or logit scores.
        """
        return self.decode(x)

    def decode(self, x_TD: Float) -> Float:
        """Projects hidden states to vocabulary logits.

        Args:
            x_TD: The input tensor of hidden states from the model backbone, with
                shape `(sequence, d_model)`.

        Returns:
            The output logits over the vocabulary, with shape
            `(sequence, vocab_size)`.
        """
        x_TD = jnp.asarray(x_TD, self.cfg.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.prelogit_td)

        with jax.named_scope("lmhead_decode_projection"):
            logits_TV = jnp.einsum('DV,TD -> TV',
                                   self.input_embedding_table_DV.value, x_TD)
        return logits_TV

    def create_sharding(self):
        """Creates and sets sharding attributes for weights and activations."""
        self.prelogit_td = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.prelogit_td))
        self.dv_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.vocab_dv))
