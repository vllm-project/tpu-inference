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
    activation_ffw_td: NamedSharding
    epsilon: float = 1e-6
    with_scale: bool = True
    dtype: Any = jnp.float32
    quant: Any | None = None

    def __call__(self, x_TD: Float, op_mode='generate') -> Float:
        """Applies RMS Normalization to the input tensor.

        Args:
            x_TD: The input tensor. The normalization is applied over the last dimension.

        Returns:
            The normalized tensor with the same shape as the input.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

        with jax.named_scope("rms_norm_variance"):
            var_T1 = jnp.mean(jnp.square(x_TD), axis=-1, keepdims=True)
        with jax.named_scope("rms_norm_rsqrt"):
            normed_x_TD = x_TD * jax.lax.rsqrt(var_T1 + self.epsilon)

        with jax.named_scope("rms_norm_scale_apply"):
            normed_x_TD *= self.scale.value
        normed_x_TD = nnx.with_sharding_constraint(normed_x_TD,
                                                   self.activation_ffw_td)
        return normed_x_TD.astype(self.dtype)

    def generate_kernel(self, rngs: nnx.Rngs):
        self.scale = self.param_factory.create_scale_param(
            rngs,
            shape=(self.dims, ),
            sharding=NamedSharding(self.mesh, P()),
            dtype=self.dtype)


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
    mesh: Mesh
    dtype: jnp.dtype
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    param_factory: ParamFactory
    df_sharding: NamedSharding
    fd_sharding: NamedSharding
    activation_ffw_td: NamedSharding
    quant: Any | None = None

    def __call__(self, x_TD):
        """Performs the forward pass of the FFW layer.

        Args:
            x_TD: The input tensor of shape either `(sequence, d_model)`

        Returns:
            The output tensor of shape `(batch, sequence, d_model)`.
        """
        # TODO consider to create factories for einsum(?)
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("wi_0"):
            gating_TF = jnp.einsum('TD,DF -> TF', x_TD,
                                   self.kernel_gating_DF.value)
            activated_gating_TF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TF)
        with jax.named_scope("wi_1"):
            up_proj_TF = jnp.einsum('TD,DF -> TF', x_TD,
                                    self.kernel_up_proj_DF.value)
        fuse_TF = activated_gating_TF * up_proj_TF
        with jax.named_scope("wo"):
            output_TD = jnp.einsum('TF,FD -> TD', fuse_TF,
                                   self.kernel_down_proj_FD.value)

        return output_TD

    def generate_kernel(self, rngs: nnx.Rngs):
        D = self.hidden_size
        F = self.intermediate_size

        self.kernel_gating_DF = self.param_factory.create_kernel_param(
            rngs, shape=(D, F), dtype=self.dtype, sharding=self.df_sharding)
        # FP8
        # TODO: update 128 to use config
        self.kernel_gating_scale_DF = self.param_factory.create_kernel_param(
            rngs,
            shape=(D // 128, F // 128),
            dtype=self.dtype,
            sharding=self.df_sharding)
        self.kernel_up_proj_DF = self.param_factory.create_kernel_param(
            rngs, shape=(D, F), dtype=self.dtype, sharding=self.df_sharding)
        # FP8
        # TODO: update 128 to use config
        self.kernel_up_proj_scale_DF = self.param_factory.create_kernel_param(
            rngs,
            shape=(D // 128, F // 128),
            dtype=self.dtype,
            sharding=self.df_sharding)
        self.kernel_down_proj_FD = self.param_factory.create_kernel_param(
            rngs, shape=(F, D), dtype=self.dtype, sharding=self.fd_sharding)
        # FP8
        # TODO: update 128 to use config
        self.kernel_down_proj_scale_FD = self.param_factory.create_kernel_param(
            rngs,
            shape=(F // 128, D // 128),
            dtype=self.dtype,
            sharding=self.fd_sharding)


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
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype
    mesh: Mesh
    param_factory: ParamFactory
    prelogit_td: NamedSharding
    vd_sharding: NamedSharding
    normalize_embeddings: bool = False

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
        self.input_embedding_table_VD = self.param_factory.create_kernel_param(
            rngs,
            shape=(self.vocab_size, self.hidden_size),
            sharding=self.vd_sharding,
            dtype=self.dtype)

    def decode(self, x_TD: Float) -> Float:
        """Projects hidden states to vocabulary logits.

        Args:
            x_TD: The input tensor of hidden states from the model backbone, with
                shape `(sequence, d_model)`.

        Returns:
            The output logits over the vocabulary, with shape
            `(sequence, vocab_size)`.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
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

        if self.normalize_embeddings:
            with jax.named_scope("embedder_normalize_embeddings"):
                embedding_TD *= jnp.sqrt(self.hidden_size).astype(self.dtype)
        return embedding_TD


@dataclass(kw_only=True)
class LMhead(Embedder):
    """
    An Embedder that uses a (D, V) shaped embedding table, inheriting from
    the base Embedder class.

    This implementation overrides the kernel generation, encoding, and decoding
    methods to work with the transposed embedding matrix layout.
    """
    dv_sharding: NamedSharding

    def generate_kernel(self, rngs: nnx.Rngs):
        self.input_embedding_table_DV = self.param_factory.create_kernel_param(
            rngs,
            shape=(self.hidden_size, self.vocab_size),
            sharding=self.dv_sharding,
            dtype=self.dtype)

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
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.prelogit_td)

        with jax.named_scope("lmhead_decode_projection"):
            logits_TV = jnp.einsum('DV,TD -> TV',
                                   self.input_embedding_table_DV.value, x_TD)
        return logits_TV
