# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import InitVar, dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jaxtyping import Float, Int

from tpu_inference.layers.jax.base import create_param


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


@dataclass(kw_only=True)
class RMSNorm(nnx.Module):
    """An implementation of Root Mean Square Layer Normalization.

    Attributes:
        dims: The feature dimension to normalize over.
        epsilon: A small float added to the variance to avoid division by zero.
        with_scale: If True, learns a multiplicative scale parameter.
        dtype: The data type for computations.
    """
    dims: int
    activation_ffw_td: Sharding = ()
    random_init: bool = False
    epsilon: float = 1e-6
    with_scale: bool = True
    dtype: Any = jnp.float32

    rngs: InitVar[nnx.Rngs]

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

    def __post_init__(self, rngs: nnx.Rngs):
        self.scale = create_param(rngs,
                                  shape=(self.dims, ),
                                  dtype=self.dtype,
                                  random_init=self.random_init)


@dataclass(kw_only=True)
class DenseFFW(nnx.Module):
    """A Gated Feed-Forward Network (FFN) layer.

    This module consists of two linear projections (gating and up-projection),
    an element-wise multiplication of the activated gating projection and the
    up-projection, followed by a final downward projection.

    Attributes:
        sharding_cfg: The configuration for tensor sharding.
    """
    dtype: jnp.dtype
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    df_sharding: Sharding = ()
    fd_sharding: Sharding = ()
    activation_ffw_td: Sharding = ()
    random_init: bool = False

    rngs: InitVar[nnx.Rngs]

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

    def __post_init__(self, rngs: nnx.Rngs):
        D = self.hidden_size
        F = self.intermediate_size

        self.kernel_gating_DF = create_param(rngs,
                                             shape=(D, F),
                                             dtype=self.dtype,
                                             sharding=self.df_sharding,
                                             random_init=self.random_init)
        self.kernel_up_proj_DF = create_param(rngs,
                                              shape=(D, F),
                                              dtype=self.dtype,
                                              sharding=self.df_sharding,
                                              random_init=self.random_init)
        self.kernel_down_proj_FD = create_param(rngs,
                                                shape=(F, D),
                                                dtype=self.dtype,
                                                sharding=self.fd_sharding,
                                                random_init=self.random_init)


@dataclass(kw_only=True)
class Embedder(nnx.Module):
    """A module for token embedding and, optionally, decoding (tied embeddings).

    This class handles both the "encoding" step of converting token IDs to dense
    vectors and the "decoding" step of projecting model outputs back to logits
    over the vocabulary.

    """
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype
    prelogit_td: Sharding = ()
    vd_sharding: Sharding = ()
    random_init: bool = False
    normalize_embeddings: bool = False

    rngs: InitVar[nnx.Rngs]

    def __post_init__(self, rngs: nnx.Rngs):
        self.input_embedding_table_VD = create_param(
            rngs,
            shape=(self.vocab_size, self.hidden_size),
            sharding=self.vd_sharding,
            dtype=self.dtype,
            random_init=self.random_init)

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
    dv_sharding: Sharding

    def __post_init__(self, rngs: nnx.Rngs):
        self.input_embedding_table_DV = create_param(
            rngs,
            shape=(self.hidden_size, self.vocab_size),
            sharding=self.dv_sharding,
            dtype=self.dtype,
            random_init=self.random_init)

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
